"""SHAP ranking figures for the slide deck: best 2 ML models per arm.

Selection rule: within each ML family (RF, XGBoost, SVR), take the best variant by mean
outer-CV RMSE on the full covariate set, then keep the two best families per arm.
CV (training folds only) is used so the test set plays no part in choosing what to show.
Elastic net and spline + ridge are treated as covariate-style models, not ML, and excluded.

Inputs (read-only):
  outputs_extended_ml_models/<arm>/cv_summary_combined.csv   CV RMSE for every model
  outputs_extended_ml_models/<arm>/test_metrics_combined.csv test RMSE (reported, not used to select)
  outputs_shap_all_models/shap_ranking_all_arms_all_models.csv SHAP rankings for every model

Outputs: sharvari/shap_slides/figures/shap_ranking_<arm>.png and selected_models.csv
Run from the console:  %run sharvari/shap_slides/make_shap_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
SHARVARI = HERE.parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

ARMS = {"Mexiletine": "mexiletine", "Oligonucleotide": "oligonucleotide", "Tideglusib": "tideglusib"}
FAMILIES = {
    "rf_untuned": "Random forest", "rf_tuned": "Random forest",
    "xgb_untuned": "XGBoost", "xgb_tuned": "XGBoost",
    "svr_tuned": "SVR",
}
LABELS = {
    "rf_untuned": "Random forest (untuned)", "rf_tuned": "Random forest (tuned)",
    "xgb_untuned": "XGBoost (untuned)", "xgb_tuned": "XGBoost (tuned)",
    "svr_tuned": "SVR (RBF, tuned)",
}
OUTCOMES = {
    "Mexiletine": "myotonia_change_12wk (negative = improvement)",
    "Oligonucleotide": "qmt_change_26wk (positive = improvement)",
    "Tideglusib": "tideglusib_composite (higher = improvement)",
}


def cv_rmse(slug):
    cv = pd.read_csv(SHARVARI / "outputs_extended_ml_models" / slug / "cv_summary_combined.csv",
                     header=[0, 1], index_col=0, keep_default_na=False)
    return pd.DataFrame({"cv_rmse_mean": cv[("RMSE", "mean")], "cv_rmse_sd": cv[("RMSE", "std")]})


def select_models():
    rows = []
    for arm, slug in ARMS.items():
        cv = cv_rmse(slug).loc[list(FAMILIES)]
        test = pd.read_csv(SHARVARI / "outputs_extended_ml_models" / slug / "test_metrics_combined.csv",
                           index_col="model", keep_default_na=False)
        cv["family"] = cv.index.map(FAMILIES)
        best_per_family = cv.sort_values("cv_rmse_mean").groupby("family").head(1)
        top2 = best_per_family.sort_values("cv_rmse_mean").head(2)
        for rank, (model, r) in enumerate(top2.iterrows(), start=1):
            rows.append({
                "arm": arm, "slot": rank, "model": model, "family": r["family"],
                "cv_rmse_mean": r["cv_rmse_mean"], "cv_rmse_sd": r["cv_rmse_sd"],
                "test_rmse": test.loc[model, "RMSE"], "test_r2": test.loc[model, "R2"],
                "null_test_rmse": test.loc["null", "RMSE"],
            })
    return pd.DataFrame(rows)


def short_direction(s):
    """Compress the pipeline's direction text to a short tag for bar labels."""
    if s.startswith("higher value -> higher"):
        return "↑ value → ↑ pred"
    if s.startswith("higher value -> lower"):
        return "↑ value → ↓ pred"
    if s.startswith("levels by mean SHAP: "):
        return s.replace("levels by mean SHAP: ", "")
    return s


def plot_arm(arm, chosen, shap):
    cmap = plt.get_cmap("viridis_r")
    norm = Normalize(0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.9), sharex=False)
    for ax, (_, m) in zip(axes, chosen.iterrows()):
        r = (
            shap.query("arm == @arm and covariate_set == 'full' and model == @m.model")
            .sort_values("rank", ascending=False)
        )
        y = np.arange(len(r))
        ax.barh(y, r["mean_abs_shap_train"], color=cmap(norm(r["cv_top3_freq"])), edgecolor="none")
        ax.set_yticks(y, r["covariate"], fontsize=9.5)
        xmax = r["mean_abs_shap_train"].max()
        ax.set_xlim(0, xmax * 1.55)
        for yi, (val, share, d) in enumerate(zip(r["mean_abs_shap_train"], r["share_of_total"], r["direction"])):
            ax.text(val + xmax * 0.015, yi, f"{share:.0%}  {short_direction(d)}", va="center", fontsize=7.5,
                    color="#333333")
        method = "TreeSHAP" if m.family != "SVR" else "permutation SHAP"
        ax.set_title(
            f"{LABELS[m.model]}\nCV RMSE {m.cv_rmse_mean:.2f} ± {m.cv_rmse_sd:.2f} | "
            f"test RMSE {m.test_rmse:.2f} (null {m.null_test_rmse:.2f})",
            fontsize=10.5, loc="left",
        )
        ax.set_xlabel(f"mean |SHAP| on training set ({method}, outcome units)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("CV stability: share of 15 outer folds\nwhere covariate ranks in top 3", fontsize=8.5)
    fig.suptitle(f"{arm}: SHAP covariate ranking, best 2 ML models   |   outcome: {OUTCOMES[arm]}",
                 fontsize=12.5, x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.135, right=0.89, top=0.84, bottom=0.10, wspace=0.6)
    out = FIG_DIR / f"shap_ranking_{ARMS[arm]}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    selected = select_models()
    selected.round(3).to_csv(HERE / "selected_models.csv", index=False)
    shap = pd.read_csv(SHARVARI / "outputs_shap_all_models" / "shap_ranking_all_arms_all_models.csv")
    for arm in ARMS:
        print(plot_arm(arm, selected.query("arm == @arm"), shap))
