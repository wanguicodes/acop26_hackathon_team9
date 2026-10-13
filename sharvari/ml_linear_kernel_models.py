"""Elastic net, spline + ridge, and SVR (RBF) on the FULL covariate set, per arm.

Same data, covariates, outer CV folds, final-tuning CV, and test patients as the main run
(`ml_<arm>.ipynb` -> `outputs/<arm>/`), so the new models can be compared fold-by-fold with
RF / XGBoost. Nothing in `arm_config.py`, `ml_pipeline.py` or `outputs/` is changed; results go
to `outputs_extended_ml_models/<arm>/` (extended ML models tried beyond RF / XGBoost).

- Fold identity is checked: the null model is rerun and its outer-fold scores must equal the
  saved ones in `outputs/<arm>/cv_scores.csv` before RF/XGBoost scores are merged in.
- These models are not trees, so SHAP TreeExplainer does not apply. Covariate importance is
  out-of-fold permutation importance (increase in validation RMSE when a covariate is shuffled),
  computed on each outer validation fold. Elastic net also reports how often each covariate has
  a non-zero coefficient across the outer folds.
- Model choice should rest on the paired nested-CV comparison. The test set has already been
  used for the earlier runs; test scores here are reported for completeness only.

Run from the console:  %run sharvari/ml_linear_kernel_models.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arm_config as cfg  # noqa: E402
import ml_pipeline as mlp  # noqa: E402

OUTPUT_ROOT = HERE / "outputs_extended_ml_models"
FULL_OUTPUT_ROOT = cfg.OUTPUT_ROOT
N_BOOT = 2000
N_PERM_REPEATS = 10
NEW_MODELS = ["enet_tuned", "spline_ridge_tuned", "svr_tuned"]

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- models

MODELS = {
    "null": {"estimator": DummyRegressor(strategy="mean"), "grid": None, "spline": False},
    "enet_tuned": {
        "estimator": ElasticNet(max_iter=50_000),
        "grid": {
            "model__alpha": np.logspace(-3, 1, 13).tolist(),
            "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
        },
        "spline": False,
    },
    # Numeric covariates: median/IQR scaling -> cubic B-spline basis; ridge on all columns.
    "spline_ridge_tuned": {
        "estimator": Ridge(),
        "grid": {
            "pre__num__spline__n_knots": [4, 6],
            "model__alpha": np.logspace(-2, 3, 11).tolist(),
        },
        "spline": True,
    },
    # Outcome standardized inside the model so epsilon/C mean the same thing in every arm.
    "svr_tuned": {
        "estimator": TransformedTargetRegressor(regressor=SVR(kernel="rbf"), transformer=StandardScaler()),
        "grid": {
            "model__regressor__C": [0.3, 1, 3, 10, 30],
            "model__regressor__gamma": ["scale", 0.01, 0.03, 0.1],
            "model__regressor__epsilon": [0.05, 0.1, 0.3],
        },
        "spline": False,
    },
}


def build(model_name: str, data: mlp.ArmData) -> Pipeline:
    """Same preprocessing as ml_pipeline.build_pipeline, plus a spline basis where requested."""
    spec = MODELS[model_name]
    scaler = mlp.NUM_SCALERS[data.num_scaling]()
    if spec["spline"]:
        num = Pipeline([
            ("scale", scaler),
            ("spline", SplineTransformer(degree=3, n_knots=5, extrapolation="linear")),
        ])
    else:
        num = scaler
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(data.cat_cols)),
            ("num", num, list(data.num_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", clone(spec["estimator"]))])


def fit_model(model_name, X, y, data, cv, n_jobs=-1):
    pipe = build(model_name, data)
    grid = MODELS[model_name]["grid"]
    if not grid:
        return pipe.fit(X, y), {}
    search = GridSearchCV(pipe, grid, cv=cv, scoring=mlp.SCORING, n_jobs=n_jobs)
    search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_)


def enet_coefficients(pipe: Pipeline, data: mlp.ArmData) -> pd.DataFrame:
    """Elastic net coefficients per encoded column (numerics: effect per 1 IQR change)."""
    coefs = pd.Series(pipe.named_steps["model"].coef_, index=pipe.named_steps["pre"].get_feature_names_out())
    origin = mlp.feature_map(pipe, data)
    return pd.DataFrame({"covariate": origin, "coefficient": coefs}).rename_axis("feature").reset_index()


# --------------------------------------------------------------------------- nested CV

def nested_cv(data: mlp.ArmData, cv: dict, seed: int):
    X, y = data.X_train, data.y_train
    outer = mlp.stratified_splits(y, cv["outer_splits"], cv["outer_repeats"], cv["n_bins"], seed)
    scores, params, perm_rows, enet_rows = [], [], [], []
    t0 = time.time()
    for fold, (tr, va) in enumerate(outer):
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        inner = mlp.stratified_splits(y_tr, cv["inner_splits"], 1, cv["n_bins"], seed + fold + 1)
        for name in MODELS:
            est, best = fit_model(name, X_tr, y_tr, data, inner)
            scores.append({"fold": fold, "model": name, **mlp.regression_metrics(y_va, est.predict(X_va))})
            if best:
                params.append({"fold": fold, "model": name, **best})
            if name == "null":
                continue
            pi = permutation_importance(
                est, X_va, y_va, scoring=mlp.SCORING, n_repeats=N_PERM_REPEATS,
                random_state=seed + fold, n_jobs=1,
            )
            perm_rows.append(pd.DataFrame({
                "fold": fold, "model": name, "covariate": X_va.columns,
                "rmse_increase": pi.importances_mean,
            }))
            if name == "enet_tuned":
                c = enet_coefficients(est, data)
                nz = c.groupby("covariate")["coefficient"].apply(lambda s: bool((s.abs() > 1e-10).any()))
                enet_rows.append(nz.rename("selected").reset_index().assign(fold=fold))
        print(f"  outer fold {fold + 1}/{len(outer)} done ({time.time() - t0:.0f}s)")
    return (
        pd.DataFrame(scores), pd.DataFrame(params),
        pd.concat(perm_rows, ignore_index=True), pd.concat(enet_rows, ignore_index=True),
    )


def perm_summary(perm: pd.DataFrame) -> pd.DataFrame:
    """Per model x covariate: median out-of-fold RMSE increase, share of folds > 0, median rank."""
    perm = perm.assign(rank=perm.groupby(["model", "fold"])["rmse_increase"].rank(ascending=False))
    out = (
        perm.groupby(["model", "covariate"])
        .agg(
            median_rmse_increase=("rmse_increase", "median"),
            q25=("rmse_increase", lambda s: s.quantile(0.25)),
            q75=("rmse_increase", lambda s: s.quantile(0.75)),
            frac_folds_positive=("rmse_increase", lambda s: (s > 0).mean()),
            median_rank=("rank", "median"),
        )
        .reset_index()
        .sort_values(["model", "median_rmse_increase"], ascending=[True, False])
    )
    return out.round(4)


# --------------------------------------------------------------------------- plots

def plot_perm(perm: pd.DataFrame, summary: pd.DataFrame, title: str):
    fig, axes = plt.subplots(1, len(NEW_MODELS), figsize=(5.2 * len(NEW_MODELS), 6), sharex=False)
    for ax, model in zip(axes, NEW_MODELS):
        order = summary.loc[summary["model"] == model, "covariate"].tolist()
        sub = perm.loc[perm["model"] == model]
        sns.boxplot(data=sub, y="covariate", x="rmse_increase", order=order, ax=ax,
                    color="lightgrey", showfliers=False)
        sns.stripplot(data=sub, y="covariate", x="rmse_increase", order=order, ax=ax,
                      color="black", size=2.5, alpha=0.6)
        ax.axvline(0, color="firebrick", linestyle="--", linewidth=1)
        ax.set(title=model, xlabel="validation RMSE increase when shuffled", ylabel="")
    fig.suptitle(f"{title}\nout-of-fold permutation importance (15 outer folds)")
    fig.tight_layout()
    return fig


def plot_pdp(model, X_train, covariates, title: str):
    fig, ax = plt.subplots(figsize=(4.5 * len(covariates), 4))
    PartialDependenceDisplay.from_estimator(model, X_train, features=covariates, kind="average", ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- per arm

def run_arm(arm: str) -> pd.DataFrame:
    arm_cfg = cfg.ARMS[arm]
    slug = arm_cfg["slug"]
    cv = dict(cfg.CV_SETTINGS)
    out_dir = OUTPUT_ROOT / slug
    print(f"\n=== {arm}")

    data = mlp.load_arm_data(
        cfg.TRAIN_PATH, cfg.TEST_PATH, arm, arm_cfg["outcome"], arm_cfg["covariates"],
        cat_cols_all=cfg.CAT_COLS, leaky_cols=cfg.LEAKY_COLS, arm_col=cfg.ARM_COL,
        id_col=cfg.ID_COL, num_scaling=cfg.NUM_SCALING,
    )
    full_info = json.loads((FULL_OUTPUT_ROOT / slug / "run_info.json").read_text())
    assert data.covariates == full_info["covariates"], "covariates differ from the main run"

    scores, params, perm, enet_sel = nested_cv(data, cv, cfg.SEED)

    # Fold identity check against the main run, then merge its RF/XGBoost scores.
    old = pd.read_csv(FULL_OUTPUT_ROOT / slug / "cv_scores.csv", keep_default_na=False)
    old_null = old.loc[old["model"] == "null"].set_index("fold")["RMSE"]
    new_null = scores.loc[scores["model"] == "null"].set_index("fold")["RMSE"]
    assert np.allclose(old_null.sort_index(), new_null.sort_index()), "outer folds differ from main run"
    combined = pd.concat([old, scores.loc[scores["model"] != "null"]], ignore_index=True)
    order = list(dict.fromkeys(old["model"])) + NEW_MODELS
    cv_summary = mlp.summarize_cv(combined, order)
    best, paired = mlp.select_best(combined, metric="RMSE")

    # Final models: re-tuned on the full training set with the main run's final-CV design.
    final_cv = mlp.stratified_splits(data.y_train, cv["final_splits"], cv["final_repeats"], cv["n_bins"], cfg.SEED)
    final_models, final_rows = {}, []
    for name in NEW_MODELS:
        est, bp = fit_model(name, data.X_train, data.y_train, data, final_cv)
        final_models[name] = est
        final_rows.append({"model": name, **bp})
    test_new, preds = mlp.evaluate_test(final_models, data, n_boot=N_BOOT, seed=cfg.SEED)
    test_old = pd.read_csv(FULL_OUTPUT_ROOT / slug / "test_metrics.csv", index_col=0, keep_default_na=False)
    test_combined = pd.concat([test_old, test_new]).reindex(order)

    psum = perm_summary(perm)
    enet_freq = (
        enet_sel.groupby("covariate")["selected"].mean().rename("selection_freq_outer_folds")
        .sort_values(ascending=False).reset_index()
    )
    enet_final = enet_coefficients(final_models["enet_tuned"], data)

    figs = {
        "cv_scores_combined": mlp.plot_cv_scores(combined, order, title=f"{arm}: nested CV, all models (same folds)"),
        "permutation_importance": plot_perm(perm, psum, arm),
        "test_pred_vs_obs": mlp.plot_pred_vs_obs(
            preds, NEW_MODELS, arm_cfg["outcome_label"], title=f"{arm}: test set (n = {len(data.y_test)})"
        ),
    }
    for name in ["spline_ridge_tuned", "svr_tuned"]:
        top = [c for c in psum.loc[psum["model"] == name, "covariate"] if c in data.num_cols][:3]
        figs[f"partial_dependence_{name}"] = plot_pdp(
            final_models[name], data.X_train, top, title=f"{arm}: partial dependence ({name}, training set)"
        )

    tables = {
        "cv_scores_new_models": (scores, False),
        "cv_scores_combined": (combined, False),
        "cv_summary_combined": (cv_summary, True),
        "cv_paired_vs_best": (paired, True),
        "cv_tuned_params": (params, False),
        "final_tuned_params": (pd.DataFrame(final_rows), False),
        "test_metrics_combined": (test_combined, True),
        "test_predictions": (preds, False),
        "permutation_importance_folds": (perm, False),
        "permutation_importance_summary": (psum, False),
        "enet_selection_freq": (enet_freq, False),
        "enet_final_coefficients": (enet_final, False),
    }
    for name, (df, idx) in tables.items():
        mlp.save_table(df, out_dir, name, index=idx)
    for name, fig in figs.items():
        mlp.save_fig(fig, out_dir, name)
        plt.close(fig)

    mlp.save_run_info(
        out_dir, arm=arm, outcome=arm_cfg["outcome"], better=arm_cfg["better"],
        selected_model_by_cv=best, seed=cfg.SEED, cv=cv, n_boot=N_BOOT,
        n_perm_repeats=N_PERM_REPEATS, covariates=data.covariates, dropped=data.dropped,
        num_scaling=data.num_scaling, grids={k: v["grid"] for k, v in MODELS.items()},
        rf_xgb_scores_from=str(FULL_OUTPUT_ROOT / slug), fold_identity_checked=True,
        notes=[
            "Full covariate set (same as main run).",
            "Select models on paired nested CV; test set already used by earlier runs.",
            "enet coefficients are per 1 IQR change for numeric covariates.",
        ],
    )
    print(f"  best by mean outer-fold RMSE: {best}")

    cvs = combined.groupby("model")["RMSE"].agg(["mean", "std"])
    wide = combined.pivot(index="fold", columns="model", values="RMSE")
    ref = "xgb_tuned" if arm == "Tideglusib" else "rf_tuned"  # each arm's selected tree model
    return pd.DataFrame({
        "arm": arm, "model": order,
        "cv_RMSE_mean": cvs.loc[order, "mean"].round(3).to_numpy(),
        "cv_RMSE_sd": cvs.loc[order, "std"].round(3).to_numpy(),
        f"paired_diff_vs_tree": [(wide[m] - wide[ref]).mean().round(3) for m in order],
        "folds_better_than_tree": [int((wide[m] < wide[ref]).sum()) for m in order],
        "test_RMSE": test_combined["RMSE"].to_numpy(),
        "test_R2": test_combined["R2"].to_numpy(),
        "best_by_cv": [m == best for m in order],
        "tree_reference": ref,
    })


if __name__ == "__main__":
    summary = pd.concat([run_arm(arm) for arm in cfg.ARMS], ignore_index=True)
    mlp.save_table(summary, OUTPUT_ROOT, "summary_all_arms", index=False)
    print(f"\nSaved summary to {OUTPUT_ROOT / 'summary_all_arms.csv'}")
