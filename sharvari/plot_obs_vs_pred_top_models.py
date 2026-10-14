"""Observed vs predicted (test set) for the stepwise linear regression and the
top-2 tuned models per arm.

Inputs (all on the same shared test split):
  wangui/evaluate_on_test/<arm>_test_diagnostics.csv       stepwise linear regression
  sharvari/outputs/<arm>/test_predictions.csv              RF / XGBoost
  sharvari/outputs_extended_ml_models/<arm>/test_predictions.csv  Elastic net / SVR

Output:
  sharvari/figures/obs_vs_pred_top_models.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sharvari" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Top-2 model families per arm (from best_two_models_per_arm.csv), tuned versions.
ARMS = {
    "mexiletine": ("Mexiletine", "myotonia_change_12wk", ["enet_tuned", "rf_tuned"]),
    "oligonucleotide": ("Oligonucleotide", "qmt_change_26wk", ["enet_tuned", "svr_tuned"]),
    "tideglusib": ("Tideglusib", "tideglusib_composite", ["xgb_tuned", "rf_tuned"]),
}
LABELS = {
    "linreg": "Linear regression (stepwise)",
    "enet_tuned": "Elastic net, tuned",
    "rf_tuned": "Random forest, tuned",
    "svr_tuned": "SVR (RBF), tuned",
    "xgb_tuned": "XGBoost, tuned",
}


def load_arm(slug: str, models: list[str]) -> pd.DataFrame:
    lr = (
        pd.read_csv(ROOT / "wangui" / "evaluate_on_test" / f"{slug}_test_diagnostics.csv")
        .assign(model="linreg")
        .loc[:, ["patient_id", "model", "observed", "predicted"]]
    )
    ml = pd.concat(
        [
            pd.read_csv(ROOT / "sharvari" / "outputs" / slug / "test_predictions.csv"),
            pd.read_csv(
                ROOT / "sharvari" / "outputs_extended_ml_models" / slug / "test_predictions.csv"
            ),
        ]
    )
    ml = ml[ml["model"].isin(models)]
    out = pd.concat([lr, ml], ignore_index=True)
    # Every model must be scored on the same test patients.
    ids = out.groupby("model")["patient_id"].apply(frozenset)
    assert ids.nunique() == 1, f"{slug}: models scored on different test patients"
    return out


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    resid = obs - pred
    return {
        "RMSE": np.sqrt(np.mean(resid**2)),
        "MAE": np.mean(np.abs(resid)),
        "R2": 1 - np.sum(resid**2) / np.sum((obs - obs.mean()) ** 2),
    }


fig, axes = plt.subplots(3, 3, figsize=(12, 11.5))
rows = []
for i, (slug, (arm, outcome, models)) in enumerate(ARMS.items()):
    d = load_arm(slug, models)
    lo = d[["observed", "predicted"]].min().min()
    hi = d[["observed", "predicted"]].max().max()
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)
    for j, m in enumerate(["linreg", *models]):
        ax = axes[i, j]
        g = d[d["model"] == m]
        s = metrics(g["observed"].to_numpy(), g["predicted"].to_numpy())
        rows.append({"arm": arm, "model": LABELS[m], "n": len(g), **s})
        ax.plot(lims, lims, color="tab:blue", lw=1, ls="--")
        ax.scatter(g["observed"], g["predicted"], s=22, alpha=0.7, color="0.2")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title(LABELS[m], fontsize=10)
        ax.text(
            0.03, 0.97,
            f"RMSE {s['RMSE']:.2f}\nMAE {s['MAE']:.2f}\nR² {s['R2']:.2f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_xlabel("Observed" if i == 2 else "")
        if j == 0:
            ax.set_ylabel(f"{arm}\n({outcome})\nPredicted", fontsize=10)

fig.suptitle(
    "Test set: predicted vs observed (n = 35 per arm)\n"
    "Dashed line = perfect prediction",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT / "obs_vs_pred_top_models.png", dpi=200, bbox_inches="tight")

panel_metrics = pd.DataFrame(rows).round(3)
