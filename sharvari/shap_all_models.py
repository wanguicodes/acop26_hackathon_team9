"""SHAP covariate rankings for every ML model tried, in all three iterations, per arm.

Iterations (all on the shared split, same seed / CV design as the original runs):
  1. full covariate set     : rf_untuned, rf_tuned, xgb_untuned, xgb_tuned   (outputs/)
  2. reduced covariate set  : rf_untuned, rf_tuned, xgb_untuned, xgb_tuned   (outputs_reduced_covariates/)
  3. extended models        : enet_tuned, spline_ridge_tuned, svr_tuned on the full set,
                              svr_tuned on the reduced set                    (outputs_extended_ml_models/)

Final models were not saved, so each is refitted exactly as in its original run, and its test-set
predictions are checked against the saved `test_predictions.csv` before SHAP is computed. SHAP is
computed on the training set (as in the original pipeline):
  - RF / XGBoost: TreeExplainer, one-hot columns summed back to covariates (ml_pipeline.shap_explain).
  - Elastic net, spline + ridge, SVR: model-agnostic permutation SHAP on the original covariates,
    with a training-set background sample.

CV stability columns come from each run's saved out-of-fold results: fold SHAP for the tree models,
fold permutation importance for the extended models.

Outputs: `outputs_shap_all_models/` (nothing in the earlier output folders is changed).
Run from the console:  %run sharvari/shap_all_models.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arm_config as cfg  # noqa: E402
import ml_pipeline as mlp  # noqa: E402
import ml_linear_kernel_models as ext  # noqa: E402
from ml_reduced_covariates import REDUCED_COVARIATES  # noqa: E402

OUTPUT_ROOT = HERE / "outputs_shap_all_models"
BACKGROUND_N = 50
BACKGROUND_SEED = 6317
TREE_MODELS = ["rf_untuned", "rf_tuned", "xgb_untuned", "xgb_tuned"]
EXT_MODELS = ["enet_tuned", "spline_ridge_tuned", "svr_tuned"]

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

# (iteration label, covariate set, model, folder with saved predictions, label in that file, CV-stability source)
RUNS = (
    [("1_full", "full", m, "outputs", m, "fold_shap") for m in TREE_MODELS]
    + [("2_reduced", "reduced", m, "outputs_reduced_covariates", m, "fold_shap") for m in TREE_MODELS]
    + [("3_extended", "full", m, "outputs_extended_ml_models", m, "fold_perm") for m in EXT_MODELS]
    + [("3_extended", "reduced", "svr_tuned", "outputs_extended_ml_models/svr_reduced_covariates",
        "svr_tuned (reduced)", "fold_perm")]
)


def load(arm, covariates):
    a = cfg.ARMS[arm]
    return mlp.load_arm_data(
        cfg.TRAIN_PATH, cfg.TEST_PATH, arm, a["outcome"], covariates,
        cat_cols_all=cfg.CAT_COLS, leaky_cols=cfg.LEAKY_COLS, arm_col=cfg.ARM_COL,
        id_col=cfg.ID_COL, num_scaling=cfg.NUM_SCALING,
    )


def agnostic_shap(pipe, data: mlp.ArmData) -> pd.DataFrame:
    """Permutation SHAP on the original covariates; categoricals passed as integer codes."""
    X = data.X_train
    levels = {c: sorted(X[c].unique()) for c in data.cat_cols}

    def to_num(df):
        df = df[data.covariates].copy()
        for c in data.cat_cols:
            df[c] = df[c].map({lv: i for i, lv in enumerate(levels[c])})
        return df.to_numpy(float)

    def predict(arr):
        df = pd.DataFrame(arr, columns=data.covariates)
        for c in data.cat_cols:
            df[c] = df[c].round().astype(int).map(dict(enumerate(levels[c])))
        df[data.num_cols] = df[data.num_cols].astype(float)
        return pipe.predict(df)

    bg = X.sample(n=min(BACKGROUND_N, len(X)), random_state=BACKGROUND_SEED)
    explainer = shap.explainers.Permutation(predict, shap.maskers.Independent(to_num(bg)), seed=BACKGROUND_SEED)
    vals = explainer(to_num(X), max_evals=max(500, 2 * len(data.covariates) + 1)).values
    return pd.DataFrame(vals, columns=data.covariates, index=X.index)


def stability(source, folder, arm_slug, label, covariates):
    """Median rank / top-3 frequency across the 15 outer CV folds from the saved run."""
    base = HERE / folder / arm_slug
    if source == "fold_shap":
        f = pd.read_csv(base / "cv_fold_shap.csv").query("model == @label")
        piv = f.pivot(index="fold", columns="covariate", values="mean_abs_shap")
    else:
        f = pd.read_csv(base / "permutation_importance_folds.csv").query("model == @label.split(' ')[0]")
        piv = f.pivot(index="fold", columns="covariate", values="rmse_increase")
    ranks = piv[covariates].rank(axis=1, ascending=False)
    return pd.DataFrame({"cv_median_rank": ranks.median(), "cv_top3_freq": (ranks <= 3).mean()})


def run_arm(arm):
    a = cfg.ARMS[arm]
    slug = a["slug"]
    cv = cfg.CV_SETTINGS
    datasets = {"full": load(arm, a["covariates"]), "reduced": load(arm, REDUCED_COVARIATES[arm])}
    specs = mlp.default_model_specs(cfg.SEED)
    rows, checks = [], []

    for iteration, cov_set, model, folder, label, stab_src in RUNS:
        data = datasets[cov_set]
        final_cv = mlp.stratified_splits(data.y_train, cv["final_splits"], cv["final_repeats"], cv["n_bins"], cfg.SEED)
        if model in TREE_MODELS:
            pipe, _ = mlp.fit_spec(specs[model], data.X_train, data.y_train, data, final_cv)
            shap_df = mlp.shap_explain(pipe, data.X_train, data).by_covariate
            method = "TreeExplainer"
        else:
            pipe, _ = ext.fit_model(model, data.X_train, data.y_train, data, final_cv)
            shap_df = agnostic_shap(pipe, data)
            method = "Permutation (model-agnostic)"

        saved = pd.read_csv(HERE / folder / slug / "test_predictions.csv", keep_default_na=False)
        saved = saved.loc[saved["model"] == label].set_index("patient_id")["predicted"]
        new = pd.Series(pipe.predict(data.X_test), index=data.ids_test.to_numpy())
        max_diff = float((saved.loc[new.index] - new).abs().max())
        checks.append({"arm": arm, "iteration": iteration, "covariate_set": cov_set, "model": model,
                       "max_abs_diff_vs_saved_test_predictions": max_diff, "reproduced": max_diff < 1e-6})

        imp = shap_df.abs().mean()
        r = pd.DataFrame({"mean_abs_shap_train": imp, "share_of_total": imp / imp.sum()})
        r["direction"] = [mlp.effect_direction(c, data.X_train, shap_df, data.cat_cols) for c in r.index]
        r = r.join(stability(stab_src, folder, slug, label, data.covariates))
        r = r.sort_values("mean_abs_shap_train", ascending=False)
        r.insert(0, "rank", range(1, len(r) + 1))
        r = r.rename_axis("covariate").reset_index()
        r.insert(0, "shap_method", method)
        r.insert(0, "cv_stability_from", "fold SHAP" if stab_src == "fold_shap" else "fold permutation importance")
        r.insert(0, "model", model)
        r.insert(0, "covariate_set", cov_set)
        r.insert(0, "iteration", iteration)
        r.insert(0, "arm", arm)
        rows.append(r)
        print(f"  {iteration:10s} {cov_set:8s} {model:20s} reproduced={max_diff < 1e-6} top: {r['covariate'].head(3).tolist()}")

    long = pd.concat(rows, ignore_index=True).round(4)
    long["model_label"] = long["iteration"] + " | " + long["covariate_set"] + " | " + long["model"]
    rank_matrix = long.pivot(index="covariate", columns="model_label", values="rank")
    rank_matrix = rank_matrix[list(dict.fromkeys(long["model_label"]))]
    full_cols = [c for c in rank_matrix.columns if "| full |" in c]
    rank_matrix.insert(0, "median_rank_full_set_models", rank_matrix[full_cols].median(axis=1))
    rank_matrix = rank_matrix.sort_values("median_rank_full_set_models")

    out_dir = OUTPUT_ROOT / slug
    mlp.save_table(long.drop(columns="model_label"), out_dir, "shap_ranking_all_models", index=False)
    mlp.save_table(rank_matrix, out_dir, "shap_rank_matrix", index=True)

    heat = rank_matrix.drop(columns="median_rank_full_set_models")
    fig, ax = plt.subplots(figsize=(0.75 * heat.shape[1] + 4, 0.42 * heat.shape[0] + 2.5))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="viridis_r", cbar_kws={"label": "SHAP rank (1 = most important)"},
                linewidths=0.5, linecolor="white", ax=ax, mask=heat.isna())
    ax.set(xlabel="", ylabel="", title=f"{arm}: SHAP rank by model and iteration (blank = not in covariate set)")
    ax.tick_params(axis="x", rotation=60)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    mlp.save_fig(fig, out_dir, "shap_rank_heatmap")
    plt.close(fig)
    return long, pd.DataFrame(checks)


if __name__ == "__main__":
    all_long, all_checks = [], []
    for arm in cfg.ARMS:
        print(f"\n=== {arm}")
        lg, ck = run_arm(arm)
        all_long.append(lg)
        all_checks.append(ck)
    shap_all = pd.concat(all_long, ignore_index=True)
    repro = pd.concat(all_checks, ignore_index=True)
    mlp.save_table(shap_all.drop(columns="model_label"), OUTPUT_ROOT, "shap_ranking_all_arms_all_models", index=False)
    mlp.save_table(repro, OUTPUT_ROOT, "reproducibility_check", index=False)
    print(f"\nAll models reproduced saved test predictions: {bool(repro['reproduced'].all())}")
    print(f"Saved to {OUTPUT_ROOT}")
