---
name: dm1-results-integration
description: Use when a team member has finished (or is working on) their per-arm covariate or ML model for the ACoP 2026 DM1 hackathon and wants their results reflected in the shared evaluation dashboard/app. Explains the exact file contract the dashboard reads and how to get your results into it, regardless of what language or method you used.
---

# Getting your results into the DM1 hackathon dashboard

This project has a shared evaluation dashboard (`templates/dashboard/index.qmd`,
with a Shiny alternative at `templates/shiny_app/app.R`) that displays, per arm:
predicted-vs-observed plots, a scoreboard (RMSE/MAE/R2), and a covariate
comparison table (coefficients vs. ML importance, side by side).

**You are free to use any model, any package, any language.** The only
requirement is writing your final numbers into the file contract below. The
dashboard reads these files; it does not care how they were produced.

## The results contract

For each arm, write three CSVs to `results/<arm_slug>/` where `arm_slug` is
`mexiletine`, `oligonucleotide`, or `tideglusib`:

### `results/<arm_slug>/scoreboard.csv`
One row per method you fit (covariate or ML, untuned or tuned).

| column | type | notes |
|---|---|---|
| `arm` | string | `"Mexiletine"`, `"Oligonucleotide"`, or `"Tideglusib"` (exact case) |
| `method` | string | e.g. `"Elastic Net"`, `"Stepwise"`, `"Random Forest"`, `"XGBoost"` -- your choice |
| `method_type` | string | `"covariate"` or `"ml"` |
| `tuned` | boolean | `TRUE`/`FALSE` |
| `rmse` | numeric | on the test set, locked, evaluated once |
| `mae` | numeric | same |
| `r2` | numeric | same |

### `results/<arm_slug>/covariates.csv`
One row per covariate per method (only covariates with a non-zero
coefficient/importance -- don't list every covariate that was zeroed out).

| column | type | notes |
|---|---|---|
| `arm` | string | same as above |
| `method` | string | must match a `method` in your scoreboard.csv |
| `covariate` | string | the covariate name (if you one-hot encoded, e.g. `phenotype_juvenile_onset` is fine) |
| `value` | numeric | the coefficient, importance score, or SHAP value |
| `value_type` | string | `"coefficient"`, `"importance"`, or `"shap"` |
| `rank` | integer | 1 = most important for that method, within that arm |

### `results/<arm_slug>/predictions.csv`
Test-set predictions only (never training predictions -- the dashboard's
predicted-vs-observed plot should reflect genuine held-out performance).

| column | type | notes |
|---|---|---|
| `arm` | string | same as above |
| `patient_id` | string | from the shared split files |
| `observed` | numeric | the true continuous outcome |
| `predicted` | numeric | your model's prediction |
| `method` | string | must match a `method` in your scoreboard.csv |

## How to use this (for your Posit Assistant session)

If someone asks you to "get my results into the dashboard," do this:

1. Find their fitted model object(s) and test-set predictions in their own
   working files (whatever arm/method they're working on).
2. Compute RMSE, MAE, R2 on the LOCKED test set
   (`data/derived/split/testing_shared.csv`, filtered to their arm) if not
   already computed. Never refit using the test set.
3. Extract their model's coefficients or importance scores, non-zero only,
   ranked by `abs(value)` descending within that method.
4. Write the three CSVs above to `results/<arm_slug>/`, using
   `write_csv()`/`to_csv()` (R or Python both fine) -- append or overwrite
   rows for their own method; don't delete other methods' rows already in
   the file. If the file already has rows for other methods, read it first,
   filter out any existing rows for THIS method+arm, then bind/concat the
   new rows and rewrite.
5. Re-render the dashboard to confirm it picks up the new data:
   `quarto render templates/dashboard/index.qmd`
6. Do not touch `templates/dashboard/index.qmd`, `templates/shiny_app/app.R`,
   or other files in `templates/` unless the user explicitly asks for a
   layout/content change to the app itself -- those are shared files other
   people also depend on.

## Context: what we agreed on as a team (see AGENTS.md for full detail)

- Continuous outcome only, no binary/responder modeling.
- `age` is dropped as a predictor everywhere (exact collinearity with
  `age_at_onset` + `disease_duration`).
- Use the shared split (`data/derived/split/training_shared.csv` /
  `testing_shared.csv`) -- do not re-split or use a different seed. Every
  method must be scored on the same test patients for the scoreboard to be
  valid.
- A reference implementation (elastic net + random forest) already exists
  at `templates/quarto_workflow/model_workflow.qmd` and has already written
  real baseline results to `results/`. Feel free to overwrite/replace this
  with your own, better methodology -- it's a fallback, not the final answer.
- You can deviate from any of the above if you have a good reason -- just
  say so (in your code comments or in Slack/standup) so the dashboard
  doesn't silently show inconsistent methodology across arms.
