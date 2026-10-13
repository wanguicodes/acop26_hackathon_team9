"""Rerun the per-arm ML pipeline on a reduced, SHAP-informed covariate set.

Identical to the arm notebooks (`ml_<arm>.ipynb`) except for the covariate list, which is
overridden per arm below. Nothing in `arm_config.py`, `ml_pipeline.py` or `outputs/` is
changed; results go to `outputs_reduced_covariates/<arm>/`, plus a full-vs-reduced comparison.

Reduced sets were chosen from the full-covariate run's SHAP ranking (`outputs/<arm>/shap_ranking.csv`):
covariates with a clear gap in mean |SHAP| and a high top-3 frequency across outer CV folds.
Tideglusib additionally keeps `age` (team decision).

Caveats:
- The subsets were chosen from SHAP on the full training set, so the nested-CV scores below are
  optimistic relative to a procedure that selects covariates inside each outer fold.
- The test set was already used to score the full-covariate models; these test scores are a
  second look and should be reported as a sensitivity analysis, not used to pick a winner.

Run from the console:  %run sharvari/ml_reduced_covariates.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arm_config as cfg  # noqa: E402
import ml_pipeline as mlp  # noqa: E402

REDUCED_COVARIATES = {
    "Mexiletine": ["myotonia_severity", "ctg_repeats"],
    "Oligonucleotide": [
        "ctg_repeats", "myotonia_severity", "somatic_mosaicism_index", "insulin_resistance",
    ],
    "Tideglusib": ["cognitive_function", "muscle_strength", "pr_interval", "age"],
}

FULL_OUTPUT_ROOT = cfg.OUTPUT_ROOT
REDUCED_OUTPUT_ROOT = HERE / "outputs_reduced_covariates"
N_NOISE_PROBES = 0
N_BOOT = 2000

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")


def run_arm(arm: str, covariates: list[str]) -> dict:
    arm_cfg = cfg.ARMS[arm]
    cv = dict(cfg.CV_SETTINGS)
    out_dir = REDUCED_OUTPUT_ROOT / arm_cfg["slug"]
    print(f"\n=== {arm}: {len(covariates)} covariates {covariates} -> {out_dir}")

    data = mlp.load_arm_data(
        cfg.TRAIN_PATH, cfg.TEST_PATH, arm, arm_cfg["outcome"], covariates,
        cat_cols_all=cfg.CAT_COLS, leaky_cols=cfg.LEAKY_COLS, arm_col=cfg.ARM_COL,
        id_col=cfg.ID_COL, num_scaling=cfg.NUM_SCALING,
    )
    data = mlp.add_noise_probes(data, N_NOISE_PROBES, cfg.SEED)

    specs = mlp.default_model_specs(cfg.SEED)
    model_order = list(specs)

    cv_res = mlp.nested_cv(
        data, specs, outer_splits=cv["outer_splits"], outer_repeats=cv["outer_repeats"],
        inner_splits=cv["inner_splits"], n_bins=cv["n_bins"], seed=cfg.SEED, verbose=False,
    )
    cv_summary = mlp.summarize_cv(cv_res.scores, model_order)
    best, paired = mlp.select_best(cv_res.scores, metric="RMSE")

    final_models, final_params = mlp.fit_final(
        data, specs, final_splits=cv["final_splits"], final_repeats=cv["final_repeats"],
        n_bins=cv["n_bins"], seed=cfg.SEED,
    )
    test_metrics, test_preds = mlp.evaluate_test(final_models, data, n_boot=N_BOOT, seed=cfg.SEED)

    shap_train = mlp.shap_explain(final_models[best], data.X_train, data)
    shap_test = mlp.shap_explain(final_models[best], data.X_test, data)
    ranking = mlp.shap_ranking(shap_train, data, shap_test, cv_res.fold_shap, best)

    title = f"{arm} (reduced covariates)"
    plot_models = list(dict.fromkeys(["rf_tuned", "xgb_tuned", best]))
    top_covs = [c for c in ranking.index if c not in data.noise_cols][:3]
    figures = {
        "cv_scores": mlp.plot_cv_scores(cv_res.scores, model_order, title=f"{title}: nested CV"),
        "test_pred_vs_obs": mlp.plot_pred_vs_obs(
            test_preds, plot_models, arm_cfg["outcome_label"],
            title=f"{title}: test set (n = {len(data.y_test)})",
        ),
        "shap_importance": mlp.plot_shap_importance(
            ranking, cv_res.fold_shap, best, title=f"{title}: SHAP importance ({best})"
        ),
        "shap_beeswarm": mlp.plot_shap_beeswarm(shap_train, title=f"{title}: SHAP values ({best})"),
        "shap_dependence": mlp.plot_shap_dependence(
            shap_train, data, top_covs, title=f"{title}: SHAP dependence"
        ),
    }

    tables = {
        "cv_scores": (cv_res.scores, False),
        "cv_summary": (cv_summary, True),
        "cv_tuned_params": (cv_res.params, False),
        "cv_paired_vs_best": (paired, True),
        "cv_fold_shap": (cv_res.fold_shap, False),
        "final_tuned_params": (final_params, False),
        "test_metrics": (test_metrics, True),
        "test_predictions": (test_preds, False),
        "shap_ranking": (ranking, True),
        "shap_values_train": (
            shap_train.by_covariate.assign(patient_id=data.ids_train.to_numpy()), False
        ),
        "scaling_params": (mlp.scaling_table(final_models[best], data), True),
    }
    for name, (df, write_index) in tables.items():
        mlp.save_table(df, out_dir, name, index=write_index)
    for name, fig in figures.items():
        mlp.save_fig(fig, out_dir, name)
        plt.close(fig)

    mlp.save_run_info(
        out_dir, arm=arm, outcome=arm_cfg["outcome"], better=arm_cfg["better"],
        selected_model=best, seed=cfg.SEED, cv=cv, n_boot=N_BOOT, quick_run=False,
        covariate_set="reduced (SHAP-informed, from outputs/<arm>/shap_ranking.csv)",
        covariates=data.covariates, dropped=data.dropped, noise_probes=data.noise_cols,
        num_scaling=data.num_scaling, grids={name: s.grid for name, s in specs.items()},
        train_file=str(cfg.TRAIN_PATH), test_file=str(cfg.TEST_PATH),
        caveats=[
            "Covariates selected on the full training set: nested-CV scores are optimistic.",
            "Test set already used for the full-covariate run: treat as sensitivity analysis.",
        ],
    )
    print(f"selected model: {best}")
    return {"best": best, "cv_summary": cv_summary, "test_metrics": test_metrics}


def compare_full_vs_reduced(arm: str, reduced: dict) -> pd.DataFrame:
    """Side-by-side CV and test metrics for the full-covariate run and the reduced run."""
    slug = cfg.ARMS[arm]["slug"]
    full_dir = FULL_OUTPUT_ROOT / slug
    full_info = json.loads((full_dir / "run_info.json").read_text())
    # keep_default_na=False: otherwise pandas parses the model name "null" as NaN
    full_cv = pd.read_csv(
        full_dir / "cv_summary.csv", header=[0, 1], index_col=0, keep_default_na=False
    )
    full_test = pd.read_csv(full_dir / "test_metrics.csv", index_col=0, keep_default_na=False)

    rows = []
    for label, cv_sum, test, n_cov, best in [
        ("full", full_cv, full_test, len(full_info["covariates"]), full_info["selected_model"]),
        ("reduced", reduced["cv_summary"], reduced["test_metrics"],
         len(REDUCED_COVARIATES[arm]), reduced["best"]),
    ]:
        for model in test.index:
            rows.append({
                "arm": arm, "covariate_set": label, "n_covariates": n_cov, "model": model,
                "selected": model == best,
                "cv_RMSE_mean": cv_sum.loc[model, ("RMSE", "mean")],
                "cv_RMSE_sd": cv_sum.loc[model, ("RMSE", "std")],
                "cv_R2_mean": cv_sum.loc[model, ("R2", "mean")],
                "test_RMSE": test.loc[model, "RMSE"],
                "test_RMSE_ci": f"[{test.loc[model, 'RMSE_ci_low']}, {test.loc[model, 'RMSE_ci_high']}]",
                "test_MAE": test.loc[model, "MAE"],
                "test_R2": test.loc[model, "R2"],
                "test_R2_ci": f"[{test.loc[model, 'R2_ci_low']}, {test.loc[model, 'R2_ci_high']}]",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    comparisons = []
    for arm, covs in REDUCED_COVARIATES.items():
        res = run_arm(arm, covs)
        comparisons.append(compare_full_vs_reduced(arm, res))
    comparison = pd.concat(comparisons, ignore_index=True)
    mlp.save_table(comparison, REDUCED_OUTPUT_ROOT, "comparison_full_vs_reduced", index=False)
    (REDUCED_OUTPUT_ROOT / "covariate_sets.json").write_text(
        json.dumps(REDUCED_COVARIATES, indent=2)
    )
    print(f"\nSaved comparison to {REDUCED_OUTPUT_ROOT / 'comparison_full_vs_reduced.csv'}")
