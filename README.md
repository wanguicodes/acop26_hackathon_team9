# ACoP 2026 DM1 Hackathon: Team 9

**Identifying baseline predictors of treatment response in a simulated myotonic dystrophy type 1 (DM1) trial**

## Summary

The simulated trial has 500 patients, 18 baseline covariates, and 3 treatment arms. For each arm we modeled the
**continuous** primary outcome. We did not dichotomize into responders, because with about 170 patients per arm
that would throw away information. We then asked which baseline covariates drive response, and whether machine
learning finds anything that traditional covariate modeling misses.

- **Workflow:** we made one shared split for all methods: outliers were screened out first, then each arm was split
  80/20, stratified on the outcome, with seed 8742. Models were built and tuned on the training data only, and every
  method was scored once on the same held-out test patients (about 35 per arm) using RMSE, MAE and R².
- **Covariates:** we used 16 of the 18. We dropped `disease_duration` because it is exactly `age - age_at_onset`, and
  `phenotype` because it duplicates onset age. For Mexiletine we also dropped `albumin`, which is constant in that arm.
- **Methods:** we fit stepwise linear regression, elastic net, random forest, XGBoost and SVR. We used nested and
  repeated cross-validation, SHAP rankings, and noise-probe columns to check signal against noise.

### Key findings

| Arm | Outcome | Main driver (all methods agree) | Recommended model |
|---|---|---|---|
| Mexiletine | `myotonia_change_12wk` (negative = improvement) | `myotonia_severity`: worse baseline myotonia, larger improvement | Elastic net (ties with ML) |
| Oligonucleotide | `qmt_change_26wk` (positive = improvement) | `ctg_repeats`: longer repeats, larger QMT gain | Elastic net (ties with ML) |
| Tideglusib | `tideglusib_composite` (higher = improvement) | `cognitive_function`: worse baseline cognition, more improvement | XGBoost (nonlinear effects, e.g. muscle strength) |

ML found no hidden predictors. In each arm the top driver matches the drug's mechanism of action. ML improved
prediction only for Tideglusib. Oligonucleotide response is mostly not explained by baseline covariates (low test R²).
With only about 35 test patients per arm, confidence intervals are wide, so small differences between models should not
be over-read. The full discussion is in [sharvari/ML_key_takeaways.md](sharvari/ML_key_takeaways.md).

## Repository layout

| Path | Contents |
|---|---|
| `data/source/` | The original hackathon files, unedited: the trial dataset, the data dictionary, and the kickoff deck. |
| `data/derived/split/` | The shared train/test split used by every method (`training_shared.csv`, `testing_shared.csv`, `outlier_summary.csv`). |
| `data/derived/` | Outputs from the linear regression and from the cross-model evaluation (in `crossmodel_evaluation/`). |
| `wangui/` | `split_data.R` (makes the shared split), `eda.R` (EDA figures and tables on the full dataset), `linear_regression.R` (stepwise linear model per arm, fit on training data), and `evaluate_on_test.R` (scores those models on the test set). |
| `inez/` | EDA by arm (boxplots, within-arm correlation heatmaps) and a findings write-up (`findings_summary.Rmd` / `.html`). |
| `natiriva_code/` | Additional EDA and univariate screening of covariates against outcome (Spearman, Mann-Whitney). |
| `sharvari/` | The Python ML pipeline (`ml_pipeline.py`, `arm_config.py`, and the per-arm notebooks `ml_<arm>.ipynb`). It runs nested cross-validation with RF and XGBoost, plus SHAP and noise probes. Extended models (elastic net, SVR) and reduced-covariate sensitivity runs are in the other `ml_*.py` scripts. SHAP comparisons across models are in `shap_all_models.py`, and the slide figures are in `shap_slides/`. Each script writes to the matching `outputs*/` folder. |
| `crossmodel_evaluation.R` | Puts the linear regression and the top 2 ML models per arm side by side on the same test patients. It recomputes all metrics with one formula and checks them against the metrics each source reported. |
| `deliverable/` | The shareable outputs: a Quarto evaluation dashboard (`templates/dashboard/`), a Shiny version of it (`templates/shiny_app/`), a reference tidymodels workflow comparing elastic net and RF (`templates/quarto_workflow/`), per-arm results CSVs (`results/<arm>/`), and the covariate-importance comparison figure (`figures/`). See [deliverable/README.md](deliverable/README.md). |
| `AGENTS.md`, `.posit/` | Project context and the decisions log, plus a Posit Assistant skill that the team used while developing. |

## Reproducing the results

Run these steps in order from the repo root.

1. `wangui/split_data.R` makes the shared split. It is already committed, so you only need to rerun it to verify.
2. `wangui/linear_regression.R`, then `wangui/evaluate_on_test.R`, fit and test the linear regression.
3. The `sharvari/ml_<arm>.ipynb` notebooks (which call `ml_pipeline.py`) and the `sharvari/ml_*.py` scripts run the ML models.
4. `crossmodel_evaluation.R` produces the comparison across models on the test set.
5. `quarto render deliverable/templates/dashboard/index.qmd` builds the dashboard.

**Software:** you need R 4.6 with tidyverse, tidymodels, glmnet, ranger, broom, plotly, DT, bslib and rprojroot, plus
the Quarto CLI. For Python 3 you need numpy, pandas, scipy, statsmodels, scikit-learn, xgboost, shap, matplotlib,
seaborn and openpyxl.

## AI use

Posit Assistant (Claude) was used to summarize the kickoff deck, run structural data checks, review code against the
agreed workflow, and build the dashboard, Shiny app and reference workflow templates. All modeling decisions were made
and checked by the team.
