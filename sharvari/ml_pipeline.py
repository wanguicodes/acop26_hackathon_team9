"""Shared ML pipeline for the DM1 hackathon (Team 9): RF and XGBoost on continuous outcomes.

The same steps run for every treatment arm:

1. load_arm_data   -- filter the shared split to one arm and guard against leakage
2. nested_cv       -- outer repeated stratified K-fold scores each model; an inner K-fold tunes it
3. select_best     -- pick the model with the lowest mean outer-fold RMSE
4. fit_final       -- re-tune every model on the full training set
5. evaluate_test   -- score once on the locked test set, with bootstrap CIs
6. shap_explain / shap_ranking -- SHAP-based covariate ranking for the selected model

All folds are stratified on quantile bins of the continuous outcome.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

SCORING = "neg_root_mean_squared_error"
METRICS = ["RMSE", "MAE", "R2"]


# --------------------------------------------------------------------------- data


@dataclass
class ArmData:
    """One arm's training and test data, restricted to the model inputs."""

    arm: str
    outcome: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series
    covariates: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    dropped: dict[str, str] = field(default_factory=dict)
    noise_cols: list[str] = field(default_factory=list)


def load_arm_data(
    train_path, test_path, arm, outcome, covariates, cat_cols_all, leaky_cols,
    arm_col="treatment_arm", id_col="patient_id",
) -> ArmData:
    """Read the shared split files, keep one arm, and validate the inputs.

    Raises if a covariate is a leaky column, if columns are missing, if values are missing,
    or if a patient appears in both train and test. Drops covariates that are constant in
    the training set (with a warning), since they carry no information.
    """
    covariates = list(dict.fromkeys(covariates))
    leaked = sorted((set(covariates) & set(leaky_cols)) | ({outcome} & set(covariates)))
    if leaked:
        raise ValueError(f"Leaky columns in the covariate list: {leaked}")

    frames = {}
    for name, path in {"train": train_path, "test": test_path}.items():
        df = pd.read_csv(path)
        missing = sorted(set(covariates + [outcome, arm_col, id_col]) - set(df.columns))
        if missing:
            raise KeyError(f"{name} file is missing columns: {missing}")
        df = df.loc[df[arm_col] == arm].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No {arm} rows in {path}")
        if df[[outcome, *covariates]].isna().any().any():
            raise ValueError(f"Missing values in {name} outcome/covariates for {arm}")
        frames[name] = df

    overlap = set(frames["train"][id_col]) & set(frames["test"][id_col])
    if overlap:
        raise ValueError(f"{len(overlap)} patients appear in both train and test")

    dropped = {
        c: "constant in training set" for c in covariates if frames["train"][c].nunique() <= 1
    }
    if dropped:
        warnings.warn(f"{arm}: dropping constant covariates {list(dropped)}", stacklevel=2)
    used = [c for c in covariates if c not in dropped]
    cat_cols = [c for c in used if c in cat_cols_all]
    num_cols = [c for c in used if c not in cat_cols_all]

    def split_xy(df):
        X = df[used].copy()
        X[num_cols] = X[num_cols].astype(float)
        X[cat_cols] = X[cat_cols].astype(str)
        return X, df[outcome].astype(float), df[id_col]

    X_tr, y_tr, id_tr = split_xy(frames["train"])
    X_te, y_te, id_te = split_xy(frames["test"])
    return ArmData(
        arm=arm, outcome=outcome, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        ids_train=id_tr, ids_test=id_te, covariates=used, num_cols=num_cols,
        cat_cols=cat_cols, dropped=dropped,
    )


def add_noise_probes(data: ArmData, n_probes: int, seed: int) -> ArmData:
    """Optionally append pure-noise columns as a reference level for covariate importance."""
    if n_probes <= 0:
        return data
    rng = np.random.default_rng(seed)
    X_tr, X_te = data.X_train.copy(), data.X_test.copy()
    names = [f"noise_{i + 1}" for i in range(n_probes)]
    for name in names:
        X_tr[name] = rng.standard_normal(len(X_tr))
        X_te[name] = rng.standard_normal(len(X_te))
    return replace(
        data, X_train=X_tr, X_test=X_te, covariates=data.covariates + names,
        num_cols=data.num_cols + names, noise_cols=names,
    )


def data_summary(data: ArmData) -> pd.DataFrame:
    """Train/test sizes and outcome distribution, as a quick sanity check."""
    rows = {}
    for name, y in {"train": data.y_train, "test": data.y_test}.items():
        rows[name] = {
            "n": len(y), "mean": y.mean(), "sd": y.std(), "min": y.min(),
            "median": y.median(), "max": y.max(),
        }
    return pd.DataFrame(rows).T.round(3)


# --------------------------------------------------------------------------- folds


def outcome_bins(y, n_bins: int = 4) -> np.ndarray:
    """Equal-count quantile bins of a continuous outcome (ties broken by order)."""
    return pd.qcut(pd.Series(np.asarray(y)).rank(method="first"), q=n_bins, labels=False).to_numpy()


def stratified_splits(y, n_splits=5, n_repeats=1, n_bins=4, seed=0) -> list:
    """Positional (train_idx, val_idx) pairs, stratified on outcome quantile bins."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(len(y)), outcome_bins(y, n_bins)))


# --------------------------------------------------------------------------- models


@dataclass
class ModelSpec:
    """An estimator plus an optional tuning grid (grid=None means fit with fixed settings)."""

    estimator: object
    grid: dict | None = None
    family: str = ""


def default_model_specs(seed: int) -> dict[str, ModelSpec]:
    """Null baseline, plus untuned and tuned versions of RF and XGBoost.

    Grids are deliberately small: each arm has ~130 training patients.
    """
    rf_kw = dict(n_estimators=300, random_state=seed, n_jobs=1)
    xgb_kw = dict(objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=1)
    return {
        "null": ModelSpec(DummyRegressor(strategy="mean"), family="null"),
        "rf_untuned": ModelSpec(RandomForestRegressor(**rf_kw), family="rf"),
        "rf_tuned": ModelSpec(
            RandomForestRegressor(**rf_kw),
            grid={
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": [0.33, 0.6, 1.0],
            },
            family="rf",
        ),
        "xgb_untuned": ModelSpec(XGBRegressor(**xgb_kw), family="xgb"),
        "xgb_tuned": ModelSpec(
            XGBRegressor(subsample=0.8, colsample_bytree=0.8, **xgb_kw),
            grid={
                "model__max_depth": [1, 2, 3],
                "model__learning_rate": [0.01, 0.03, 0.1],
                "model__n_estimators": [100, 300, 600],
                "model__min_child_weight": [1, 5],
            },
            family="xgb",
        ),
    }


def build_pipeline(estimator, num_cols, cat_cols) -> Pipeline:
    """One-hot encode categoricals, pass numerics through unscaled (trees don't need scaling)."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)),
            ("num", "passthrough", list(num_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def fit_spec(spec: ModelSpec, X, y, data: ArmData, cv, n_jobs=-1):
    """Fit one model; tune it with GridSearchCV over `cv` if it has a grid."""
    pipe = build_pipeline(clone(spec.estimator), data.num_cols, data.cat_cols)
    if not spec.grid:
        return pipe.fit(X, y), {}
    search = GridSearchCV(pipe, spec.grid, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    search.fit(X, y)
    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best


def regression_metrics(observed, predicted) -> dict[str, float]:
    obs, pred = np.asarray(observed, float), np.asarray(predicted, float)
    err = obs - pred
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else np.nan,
    }


# --------------------------------------------------------------------------- nested CV


@dataclass
class CVResult:
    scores: pd.DataFrame     # one row per outer fold x model
    params: pd.DataFrame     # tuned hyperparameters per outer fold
    fold_shap: pd.DataFrame  # mean |SHAP| per covariate on each outer validation fold


def nested_cv(
    data: ArmData, specs: dict[str, ModelSpec], outer_splits=5, outer_repeats=3,
    inner_splits=5, n_bins=4, seed=0, n_jobs=-1, fold_shap=True, verbose=True,
) -> CVResult:
    """Score every model on the same outer folds; tune inside each outer-training portion only.

    The outer validation fold never touches tuning, so outer scores are honest estimates of
    each modelling procedure. Fold-level SHAP importances show how stable the ranking is.
    """
    X, y = data.X_train, data.y_train
    outer = stratified_splits(y, outer_splits, outer_repeats, n_bins, seed)
    score_rows, param_rows, shap_rows = [], [], []
    t0 = time.time()
    for fold, (tr, va) in enumerate(outer):
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        inner = stratified_splits(y_tr, inner_splits, 1, n_bins, seed + fold + 1)
        for name, spec in specs.items():
            est, best = fit_spec(spec, X_tr, y_tr, data, inner, n_jobs)
            score_rows.append({"fold": fold, "model": name, **regression_metrics(y_va, est.predict(X_va))})
            if best:
                param_rows.append({"fold": fold, "model": name, **best})
            if fold_shap and spec.family != "null":
                imp = shap_explain(est, X_va, data).by_covariate.abs().mean()
                shap_rows.append(
                    imp.rename("mean_abs_shap").rename_axis("covariate").reset_index()
                    .assign(fold=fold, model=name)
                )
        if verbose:
            print(f"outer fold {fold + 1}/{len(outer)} done ({time.time() - t0:.0f}s)")
    return CVResult(
        scores=pd.DataFrame(score_rows),
        params=pd.DataFrame(param_rows),
        fold_shap=pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame(),
    )


def summarize_cv(scores: pd.DataFrame, order=None) -> pd.DataFrame:
    """Mean and SD of each metric across outer folds."""
    order = order or list(dict.fromkeys(scores["model"]))
    return scores.groupby("model")[METRICS].agg(["mean", "std"]).reindex(order).round(3)


def tuned_param_counts(params: pd.DataFrame) -> pd.DataFrame:
    """How often each hyperparameter combination was chosen across outer folds."""
    if params.empty:
        return pd.DataFrame()
    hp_cols = [c for c in params.columns if c not in ("fold", "model")]
    def fmt(v):
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

    combo = params[hp_cols].apply(
        lambda r: ", ".join(f"{k}={fmt(v)}" for k, v in r.dropna().items()), axis=1
    )
    return (
        params.assign(setting=combo)
        .groupby(["model", "setting"]).size().rename("n_folds")
        .reset_index().sort_values(["model", "n_folds"], ascending=[True, False])
        .reset_index(drop=True)
    )


def select_best(scores: pd.DataFrame, metric="RMSE", exclude=("null",)):
    """Pick the model with the lowest mean outer-fold metric; report paired fold differences.

    Paired differences use identical outer folds, so they are more informative than
    comparing the two means alone.
    """
    cand = scores.loc[~scores["model"].isin(exclude)]
    best = cand.groupby("model")[metric].mean().idxmin()
    wide = scores.pivot(index="fold", columns="model", values=metric)
    paired = pd.DataFrame({
        m: {
            f"mean_{metric}_diff_vs_best": (wide[m] - wide[best]).mean(),
            "sd_diff": (wide[m] - wide[best]).std(),
            "folds_where_best_is_better": int((wide[best] < wide[m]).sum()),
            "n_folds": len(wide),
        }
        for m in wide.columns if m != best
    }).T
    return best, paired.round(3)


# --------------------------------------------------------------------------- final fit + test


def fit_final(
    data: ArmData, specs: dict[str, ModelSpec], final_splits=5, final_repeats=2,
    n_bins=4, seed=0, n_jobs=-1,
):
    """Re-tune each model on the full training set and refit it there."""
    cv = stratified_splits(data.y_train, final_splits, final_repeats, n_bins, seed)
    models, rows = {}, []
    for name, spec in specs.items():
        est, best = fit_spec(spec, data.X_train, data.y_train, data, cv, n_jobs)
        models[name] = est
        if best:
            rows.append({"model": name, **best})
    return models, pd.DataFrame(rows)


def evaluate_test(models: dict, data: ArmData, n_boot=2000, seed=0):
    """Score each final model once on the test set; percentile bootstrap 95% CIs."""
    preds = pd.concat(
        [
            pd.DataFrame({
                "patient_id": data.ids_test.to_numpy(), "model": name,
                "observed": data.y_test.to_numpy(), "predicted": m.predict(data.X_test),
            })
            for name, m in models.items()
        ],
        ignore_index=True,
    )
    rng = np.random.default_rng(seed)
    n = len(data.y_test)
    boot_idx = rng.integers(0, n, size=(n_boot, n))  # same resamples for every model
    rows = []
    for name, g in preds.groupby("model", sort=False):
        obs, pred = g["observed"].to_numpy(), g["predicted"].to_numpy()
        boot = pd.DataFrame([regression_metrics(obs[i], pred[i]) for i in boot_idx])
        row = {"model": name}
        for k, v in regression_metrics(obs, pred).items():
            row[k] = v
            row[f"{k}_ci_low"] = boot[k].quantile(0.025)
            row[f"{k}_ci_high"] = boot[k].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(3), preds


# --------------------------------------------------------------------------- SHAP


@dataclass
class ShapResult:
    explanation: shap.Explanation  # per encoded feature (one-hot columns separate)
    encoded: pd.DataFrame          # encoded feature matrix
    by_covariate: pd.DataFrame     # SHAP summed back to original covariates


def encode(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    return pd.DataFrame(
        pre.transform(X), columns=pre.get_feature_names_out(), index=X.index
    ).astype(float)


def feature_map(pipe: Pipeline, data: ArmData) -> pd.Series:
    """Map each encoded column back to its original covariate."""
    pre = pipe.named_steps["pre"]
    origin = []
    if data.cat_cols:
        enc = pre.named_transformers_["cat"]
        origin += list(np.repeat(data.cat_cols, [len(c) for c in enc.categories_]))
    origin += list(data.num_cols)
    names = pre.get_feature_names_out()
    assert len(names) == len(origin), "feature map does not match encoded columns"
    return pd.Series(origin, index=names)


def shap_explain(pipe: Pipeline, X: pd.DataFrame, data: ArmData) -> ShapResult:
    """TreeExplainer SHAP values; one-hot columns are summed back to their covariate.

    Summing is valid because SHAP values are additive.
    """
    X_enc = encode(pipe, X)
    explanation = shap.TreeExplainer(pipe.named_steps["model"])(X_enc)
    vals = pd.DataFrame(explanation.values, columns=X_enc.columns, index=X.index)
    by_cov = vals.T.groupby(feature_map(pipe, data)).sum().T[data.covariates]
    return ShapResult(explanation=explanation, encoded=X_enc, by_covariate=by_cov)


def effect_direction(col: str, X: pd.DataFrame, shap_df: pd.DataFrame, cat_cols) -> str:
    """Plain-language direction of a covariate's effect on the model's prediction."""
    if col in cat_cols:
        order = shap_df[col].groupby(X[col]).mean().sort_values(ascending=False)
        return "levels by mean SHAP: " + " > ".join(map(str, order.index))
    rho = X[col].corr(shap_df[col], method="spearman")
    if np.isnan(rho):
        return "no variation"
    return f"higher value -> {'higher' if rho > 0 else 'lower'} prediction (Spearman {rho:+.2f})"


def shap_ranking(
    shap_train: ShapResult, data: ArmData, shap_test: ShapResult | None = None,
    fold_shap: pd.DataFrame | None = None, best: str | None = None,
) -> pd.DataFrame:
    """Rank covariates by mean |SHAP| of the final selected model (training set).

    Adds test-set mean |SHAP|, plus the median rank and top-3 frequency across the
    outer CV folds (stability). If noise probes are present, also the share of folds in
    which each covariate beats the strongest noise column.
    """
    imp = shap_train.by_covariate.abs().mean()
    out = pd.DataFrame({"mean_abs_shap_train": imp, "share_of_total": imp / imp.sum()})
    if shap_test is not None:
        out["mean_abs_shap_test"] = shap_test.by_covariate.abs().mean()
    if fold_shap is not None and not fold_shap.empty and best is not None:
        piv = fold_shap.loc[fold_shap["model"] == best].pivot(
            index="fold", columns="covariate", values="mean_abs_shap"
        )
        ranks = piv.rank(axis=1, ascending=False)
        out["cv_median_rank"] = ranks.median()
        out["cv_top3_freq"] = (ranks <= 3).mean()
        if data.noise_cols:
            noise_max = piv[data.noise_cols].max(axis=1)
            out["cv_beats_noise_freq"] = piv.gt(noise_max, axis=0).mean()
    out["direction"] = [
        effect_direction(c, data.X_train, shap_train.by_covariate, data.cat_cols)
        for c in out.index
    ]
    out = out.sort_values("mean_abs_shap_train", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.round(3)


# --------------------------------------------------------------------------- plots


def plot_cv_scores(scores: pd.DataFrame, order, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric in zip(axes, ["RMSE", "R2"]):
        sns.boxplot(data=scores, x="model", y=metric, order=order, ax=ax,
                    color="lightgrey", showfliers=False)
        sns.stripplot(data=scores, x="model", y=metric, order=order, ax=ax,
                      color="black", size=3, alpha=0.6)
        ax.set(xlabel="", title=f"Outer-fold {metric}")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pred_vs_obs(preds: pd.DataFrame, models, outcome_label: str, title=""):
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), squeeze=False)
    lims = [preds[["observed", "predicted"]].min().min(), preds[["observed", "predicted"]].max().max()]
    pad = 0.05 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]
    for ax, name in zip(axes.flat, models):
        g = preds.loc[preds["model"] == name]
        m = regression_metrics(g["observed"], g["predicted"])
        ax.scatter(g["observed"], g["predicted"], alpha=0.7)
        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
        ax.set(xlim=lims, ylim=lims, xlabel="Observed", ylabel="Predicted",
               title=f"{name}\nRMSE {m['RMSE']:.2f}, R² {m['R2']:.2f}")
        ax.set_aspect("equal")
    fig.suptitle(f"{title}\n{outcome_label}" if title else outcome_label)
    fig.tight_layout()
    return fig


def plot_shap_importance(ranking: pd.DataFrame, fold_shap: pd.DataFrame, best: str, title=""):
    """Left: mean |SHAP| of the final model. Right: spread across outer CV folds."""
    order = list(ranking.index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.35 * len(order) + 1.5), sharey=True)
    axes[0].barh(order, ranking["mean_abs_shap_train"], color="steelblue")
    axes[0].invert_yaxis()
    axes[0].set(xlabel="mean |SHAP| (final model, training set)", title="Final model")
    if fold_shap is not None and not fold_shap.empty:
        sub = fold_shap.loc[fold_shap["model"] == best]
        sns.boxplot(data=sub, y="covariate", x="mean_abs_shap", order=order, ax=axes[1],
                    color="lightgrey", showfliers=False)
        sns.stripplot(data=sub, y="covariate", x="mean_abs_shap", order=order, ax=axes[1],
                      color="black", size=3, alpha=0.6)
        axes[1].set(xlabel="mean |SHAP| on each outer validation fold", ylabel="",
                    title="Stability across CV folds")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(shap_res: ShapResult, max_display=15, title=""):
    plt.figure()
    shap.plots.beeswarm(shap_res.explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_shap_dependence(shap_res: ShapResult, data: ArmData, covariates, title=""):
    """SHAP value vs covariate value for the top covariates (original, un-encoded scale)."""
    X = data.X_train.loc[shap_res.by_covariate.index]
    fig, axes = plt.subplots(1, len(covariates), figsize=(4.5 * len(covariates), 4), squeeze=False)
    for ax, col in zip(axes.flat, covariates):
        s = shap_res.by_covariate[col]
        if col in data.cat_cols:
            sns.stripplot(x=X[col], y=s, ax=ax, color="black", alpha=0.6, size=4)
        else:
            ax.scatter(X[col], s, alpha=0.6)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set(xlabel=col, ylabel="SHAP value", title=col)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- saving


def save_table(df: pd.DataFrame, out_dir: Path, name: str, index=True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def save_fig(fig, out_dir: Path, name: str, dpi=200) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_run_info(out_dir: Path, **info) -> Path:
    """Record settings (seed, CV design, grids, covariates) next to the results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str))
    return path
