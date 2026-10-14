"""Best two model types per arm (full covariate set), tuned and untuned, with CV and test metrics.

Model types are ranked by mean outer-fold CV RMSE (best variant of each type). The table reports
test-set RMSE / MAE / R2 with bootstrap 95% CIs for both the tuned and untuned variant.

RF / XGBoost untuned and all tuned models come from earlier runs (`outputs/`,
`outputs_extended_ml_models/`). Elastic net and SVR were only run tuned before; their untuned
versions (sklearn defaults: ElasticNet alpha=1, l1_ratio=0.5; SVR C=1, gamma="scale",
epsilon=0.1 on the standardized outcome) are fitted here on the same outer folds and test set.

Output: `outputs_extended_ml_models/best_two_models_per_arm.csv`.
Run from the console:  %run sharvari/best_two_models_table.py
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arm_config as cfg  # noqa: E402
import ml_pipeline as mlp  # noqa: E402
import ml_linear_kernel_models as ext  # noqa: E402

EXT_DIR = HERE / "outputs_extended_ml_models"
UNTUNED = {"enet_untuned": "enet_tuned", "svr_untuned": "svr_tuned"}  # default settings, no grid
FAMILY = {
    "rf_untuned": "Random forest", "rf_tuned": "Random forest",
    "xgb_untuned": "XGBoost", "xgb_tuned": "XGBoost",
    "enet_untuned": "Elastic net", "enet_tuned": "Elastic net",
    "svr_untuned": "SVR (RBF)", "svr_tuned": "SVR (RBF)",
    "spline_ridge_tuned": "Spline + ridge",
}

warnings.filterwarnings("ignore", category=FutureWarning)


def fmt_ci(row, m):
    return f"{row[m]:.3f} [{row[f'{m}_ci_low']:.3f}, {row[f'{m}_ci_high']:.3f}]"


def run_arm(arm: str) -> pd.DataFrame:
    a = cfg.ARMS[arm]
    slug = a["slug"]
    cv = cfg.CV_SETTINGS
    data = mlp.load_arm_data(
        cfg.TRAIN_PATH, cfg.TEST_PATH, arm, a["outcome"], a["covariates"],
        cat_cols_all=cfg.CAT_COLS, leaky_cols=cfg.LEAKY_COLS, arm_col=cfg.ARM_COL,
        id_col=cfg.ID_COL, num_scaling=cfg.NUM_SCALING,
    )
    X, y = data.X_train, data.y_train

    # Untuned elastic net / SVR on the same outer folds as every other model
    outer = mlp.stratified_splits(y, cv["outer_splits"], cv["outer_repeats"], cv["n_bins"], cfg.SEED)
    rows = []
    for fold, (tr, va) in enumerate(outer):
        for name, base in UNTUNED.items():
            pipe = ext.build(base, data).fit(X.iloc[tr], y.iloc[tr])
            rows.append({"fold": fold, "model": name, **mlp.regression_metrics(y.iloc[va], pipe.predict(X.iloc[va]))})
    prev_cv = pd.read_csv(EXT_DIR / slug / "cv_scores_combined.csv", keep_default_na=False)
    cv_all = pd.concat([prev_cv.loc[prev_cv["model"] != "null"], pd.DataFrame(rows)], ignore_index=True)

    finals = {name: ext.build(base, data).fit(X, y) for name, base in UNTUNED.items()}
    test_new, _ = mlp.evaluate_test(finals, data, n_boot=ext.N_BOOT, seed=cfg.SEED)
    prev_test = pd.read_csv(EXT_DIR / slug / "test_metrics_combined.csv", index_col=0, keep_default_na=False)
    test_all = pd.concat([prev_test.drop(index="null"), test_new])

    cv_stats = cv_all.groupby("model")["RMSE"].agg(["mean", "std"])
    cv_stats["family"] = cv_stats.index.map(FAMILY)
    family_rank = cv_stats.groupby("family")["mean"].min().sort_values()
    top2 = list(family_rank.index[:2])

    out = []
    for rank, fam in enumerate(top2, start=1):
        for model in sorted([m for m in cv_stats.index if FAMILY[m] == fam], key=lambda m: "untuned" in m):
            t = test_all.loc[model]
            out.append({
                "arm": arm, "rank": rank, "model_type": fam, "model": model,
                "tuned": "untuned" not in model,
                "cv_RMSE_mean_sd": f"{cv_stats.loc[model, 'mean']:.3f} ± {cv_stats.loc[model, 'std']:.3f}",
                "test_RMSE": round(t["RMSE"], 3), "test_MAE": round(t["MAE"], 3), "test_R2": round(t["R2"], 3),
                "test_RMSE_95CI": fmt_ci(t, "RMSE"), "test_MAE_95CI": fmt_ci(t, "MAE"), "test_R2_95CI": fmt_ci(t, "R2"),
            })
    return pd.DataFrame(out)


if __name__ == "__main__":
    best_two = pd.concat([run_arm(arm) for arm in cfg.ARMS], ignore_index=True)
    mlp.save_table(best_two, EXT_DIR, "best_two_models_per_arm", index=False)
    print(f"Saved to {EXT_DIR / 'best_two_models_per_arm.csv'}")
