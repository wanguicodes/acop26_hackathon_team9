"""SVR (RBF) on the reduced, SHAP-informed covariate sets, per arm.

Sensitivity analysis: SVR did poorly on the full covariate set, plausibly because irrelevant
covariates dilute the RBF kernel's distances. This reruns it on the same short covariate lists
as `ml_reduced_covariates.py`, with the same outer CV folds, so it compares fold-by-fold with
the reduced-set RF/XGBoost (`outputs_reduced_covariates/`) and with the full-set SVR
(`outputs_extended_ml_models/`).

Caveat: the reduced lists were chosen from tree SHAP on the full training set, so CV scores on
them are optimistic. That bias applies equally to the reduced RF/XGBoost scores (fair
comparison), but not to the full-set runs. The test set has been used before; select on CV.

Outputs: `outputs_extended_ml_models/svr_reduced_covariates/<arm>/`.
Run from the console:  %run sharvari/ml_svr_reduced_covariates.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arm_config as cfg  # noqa: E402
import ml_pipeline as mlp  # noqa: E402
import ml_linear_kernel_models as ext  # noqa: E402  (model specs / pipeline builder)
from ml_reduced_covariates import REDUCED_COVARIATES  # noqa: E402

OUTPUT_ROOT = HERE / "outputs_extended_ml_models" / "svr_reduced_covariates"
REDUCED_RUN = HERE / "outputs_reduced_covariates"
EXTENDED_RUN = HERE / "outputs_extended_ml_models"
MODELS = ["null", "svr_tuned"]
N_BOOT = 2000

warnings.filterwarnings("ignore", category=FutureWarning)


def run_arm(arm: str) -> pd.DataFrame:
    arm_cfg = cfg.ARMS[arm]
    slug = arm_cfg["slug"]
    cv = dict(cfg.CV_SETTINGS)
    out_dir = OUTPUT_ROOT / slug
    covs = REDUCED_COVARIATES[arm]
    print(f"\n=== {arm}: {covs}")

    data = mlp.load_arm_data(
        cfg.TRAIN_PATH, cfg.TEST_PATH, arm, arm_cfg["outcome"], covs,
        cat_cols_all=cfg.CAT_COLS, leaky_cols=cfg.LEAKY_COLS, arm_col=cfg.ARM_COL,
        id_col=cfg.ID_COL, num_scaling=cfg.NUM_SCALING,
    )
    X, y = data.X_train, data.y_train
    outer = mlp.stratified_splits(y, cv["outer_splits"], cv["outer_repeats"], cv["n_bins"], cfg.SEED)

    scores, params, perm_rows = [], [], []
    for fold, (tr, va) in enumerate(outer):
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        inner = mlp.stratified_splits(y_tr, cv["inner_splits"], 1, cv["n_bins"], cfg.SEED + fold + 1)
        for name in MODELS:
            est, bp = ext.fit_model(name, X_tr, y_tr, data, inner)
            scores.append({"fold": fold, "model": name, **mlp.regression_metrics(y_va, est.predict(X_va))})
            if bp:
                params.append({"fold": fold, "model": name, **bp})
            if name != "null":
                pi = permutation_importance(
                    est, X_va, y_va, scoring=mlp.SCORING, n_repeats=ext.N_PERM_REPEATS,
                    random_state=cfg.SEED + fold, n_jobs=1,
                )
                perm_rows.append(pd.DataFrame({
                    "fold": fold, "model": name, "covariate": X_va.columns,
                    "rmse_increase": pi.importances_mean,
                }))
    scores = pd.DataFrame(scores)
    perm = pd.concat(perm_rows, ignore_index=True)

    # Fold identity check, then combine with reduced-set trees and full-set SVR / trees.
    red = pd.read_csv(REDUCED_RUN / slug / "cv_scores.csv", keep_default_na=False)
    full_ext = pd.read_csv(EXTENDED_RUN / slug / "cv_scores_combined.csv", keep_default_na=False)
    new_null = scores.loc[scores.model == "null"].set_index("fold")["RMSE"].sort_index()
    for ref in (red, full_ext):
        old_null = ref.loc[ref.model == "null"].set_index("fold")["RMSE"].sort_index()
        assert np.allclose(old_null, new_null), "outer folds differ"

    full_tree = "xgb_tuned" if arm == "Tideglusib" else "rf_tuned"
    combined = pd.concat([
        red.loc[red.model.isin(["null", "rf_tuned", "xgb_tuned"])].assign(model=lambda d: d.model + " (reduced)"),
        scores.loc[scores.model == "svr_tuned"].assign(model="svr_tuned (reduced)"),
        full_ext.loc[full_ext.model.isin([full_tree, "svr_tuned"])].assign(model=lambda d: d.model + " (full)"),
    ], ignore_index=True)
    order = list(dict.fromkeys(combined["model"]))
    best, paired = mlp.select_best(combined, metric="RMSE", exclude=("null (reduced)",))

    final_cv = mlp.stratified_splits(y, cv["final_splits"], cv["final_repeats"], cv["n_bins"], cfg.SEED)
    svr_final, svr_params = ext.fit_model("svr_tuned", X, y, data, final_cv)
    test_metrics, preds = mlp.evaluate_test({"svr_tuned (reduced)": svr_final}, data, n_boot=N_BOOT, seed=cfg.SEED)

    psum = ext.perm_summary(perm)
    top = [c for c in psum["covariate"] if c in data.num_cols][:3]
    figs = {
        "cv_scores_compare": mlp.plot_cv_scores(combined, order, title=f"{arm}: SVR reduced vs trees / full (same folds)"),
        "partial_dependence_svr": ext.plot_pdp(svr_final, X, top, title=f"{arm}: partial dependence (SVR, reduced set)"),
        "test_pred_vs_obs": mlp.plot_pred_vs_obs(
            preds, ["svr_tuned (reduced)"], arm_cfg["outcome_label"], title=f"{arm}: test set (n = {len(data.y_test)})"
        ),
    }
    tables = {
        "cv_scores_svr_reduced": (scores, False),
        "cv_scores_compare": (combined, False),
        "cv_summary_compare": (mlp.summarize_cv(combined, order), True),
        "cv_paired_vs_best": (paired, True),
        "cv_tuned_params": (pd.DataFrame(params), False),
        "final_tuned_params": (pd.DataFrame([{"model": "svr_tuned (reduced)", **svr_params}]), False),
        "test_metrics": (test_metrics, True),
        "test_predictions": (preds, False),
        "permutation_importance_folds": (perm, False),
        "permutation_importance_summary": (psum, False),
    }
    for name, (df, idx) in tables.items():
        mlp.save_table(df, out_dir, name, index=idx)
    for name, fig in figs.items():
        mlp.save_fig(fig, out_dir, name)
        plt.close(fig)
    mlp.save_run_info(
        out_dir, arm=arm, outcome=arm_cfg["outcome"], covariates=data.covariates,
        covariate_set="reduced (SHAP-informed, same as outputs_reduced_covariates)",
        selected_model_by_cv=best, seed=cfg.SEED, cv=cv, n_boot=N_BOOT,
        grid=ext.MODELS["svr_tuned"]["grid"], fold_identity_checked=True,
        caveats=[
            "Covariates chosen on full training set: CV optimistic (equally for reduced trees).",
            "Test set used by earlier runs: select on CV.",
        ],
    )

    wide = combined.pivot(index="fold", columns="model", values="RMSE")
    ref = f"{full_tree} (reduced)"
    stats = combined.groupby("model")["RMSE"].agg(["mean", "std"]).loc[order]
    return pd.DataFrame({
        "arm": arm, "model": order,
        "cv_RMSE_mean": stats["mean"].round(3).to_numpy(),
        "cv_RMSE_sd": stats["std"].round(3).to_numpy(),
        "paired_diff_vs_reduced_tree": [round((wide[m] - wide[ref]).mean(), 3) for m in order],
        "folds_better_than_reduced_tree": [int((wide[m] < wide[ref]).sum()) for m in order],
        "best_by_cv": [m == best for m in order],
        "svr_reduced_test_RMSE": test_metrics.loc["svr_tuned (reduced)", "RMSE"],
        "svr_reduced_test_R2": test_metrics.loc["svr_tuned (reduced)", "R2"],
    })


if __name__ == "__main__":
    summary = pd.concat([run_arm(arm) for arm in cfg.ARMS], ignore_index=True)
    mlp.save_table(summary, OUTPUT_ROOT, "summary_all_arms", index=False)
    print(f"\nSaved summary to {OUTPUT_ROOT / 'summary_all_arms.csv'}")
