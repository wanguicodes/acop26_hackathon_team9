# Project memory: ACoP 2026 DM1 Hackathon — Team 9

## Repo
- GitHub: https://github.com/wanguicodes/acop26_hackathon_team9
- Raw files from SharePoint (never edited): `data/source/`
  - `dm1_clinical_trial_dataset.csv` (500 patients x 26 columns)
  - `DM1_Data_Dictionary_Participant.xlsx` (sheets: "Column Dictionary", "Summary")
  - `Kick-offSession_final_participants.pdf`
- Saved train/test splits: `data/splits/`
- Quarto docs: `quarto/` (numbered, see Folder layout)
- Outputs: `output/figures/`, `output/tables/`
- Slides: `slides/`

## The challenge
- Simulated DM1 trial: 500 patients, 18 baseline covariates, 3 arms (one arm per patient).
- Goal: per arm, identify baseline covariates that drive response, separate signal
  from noise, and compare covariate models vs ML models on the same held-out test set.
- Judging: clinical (key predictors per arm, signal vs noise), technical (workflow,
  unbiased split, metrics), communication. Meaningful covariates > fancy models.
- Deliverables: 5-min talk + slides + reproducible code, uploaded to the team
  SharePoint folder by Tue Oct 13, 11:59 PM ET. Presentations Wed Oct 14, 9:00 AM.
- Hackathon's own framing question (slide 9): "Can ML identify additional predictors
  and improve patient stratification beyond traditional covariate modeling?" — treat
  covariate modeling as the baseline, ML as the stress test against it.

## Outcomes per arm
| Arm | Continuous | Direction | Binary label |
|---|---|---|---|
| Mexiletine | `myotonia_change_12wk` | negative = improvement | `responder_phenotype_mex` |
| Oligonucleotide | `qmt_change_26wk` | positive = improvement | `responder_phenotype_oligo` |
| Tideglusib | `tideglusib_composite` (0-20, wk 26) | higher = improvement | `responder_phenotype_tidge` |

Chosen outcome: Continuous (regression), all 3 arms — preserves power on ~130-170
patients/arm vs. dichotomizing. Binary responder label used only as a secondary,
illustrative cutoff derived from model predictions, not a separate modeling track.

## Data facts (confirmed via structural check, 2026-10-11)
- 500 rows, 26 columns, matches data dictionary exactly.
- Per-arm sizes: Mexiletine 161, Oligonucleotide 170, Tideglusib 169.
- Zero missing values in any of the 18 baseline covariates.
- Outcome/responder columns populated only for a patient's assigned arm (NA elsewhere) — expected.
- **Exact collinearity confirmed:** `age = age_at_onset + disease_duration` for all 500 rows
  (not just correlated — identical). Drop `age` from models; keep `age_at_onset` and
  `disease_duration` separately (they capture distinct clinical information).
- Numeric ranges and categorical levels all match the dictionary; no implausible values.
- `phenotype` split (61.8% / 20.0% / 18.2%) matches slide 40 exactly — confirms this is the
  real dataset referenced at kickoff.
- Responder label balance (reference only, not the modeling target): Mexiletine 114/47,
  Oligonucleotide 100/70, Tideglusib 88/81.

## Covariates (18)
- Numeric (13): ctg_repeats, somatic_mosaicism_index, age, age_at_onset, disease_duration,
  myotonia_severity, muscle_strength, pr_interval, cognitive_function, ck_level,
  insulin_resistance, albumin, bmi
- Categorical (5): phenotype (3), sex (2), family_history (3), cataracts (2),
  smoking_status (3)
- Drop `age` from modeling (exact collinearity with age_at_onset + disease_duration, confirmed above).

## Workflow rules
1. Split each arm separately, stratified by outcome (bin continuous outcome into groups), fixed seed.
2. Seed: 7461 (proposed; confirm/override before running the actual split).
3. Test set is locked. Use it once, for final evaluation only.
4. All EDA after the split uses training data only.
5. Preprocessing is fit on train only and then applied to test.
6. One scoreboard per arm, with all methods evaluated on the same test patients.
   Regression: RMSE, MAE, R². Show untuned and tuned results side by side.
7. Report the covariates selected or ranked by each method, and compare across methods.

## Folder layout
- `quarto/` — numbered Quarto docs: `01_eda.qmd`, `02_split.qmd`, `03_models.qmd`
  (parameterized by `arm`, rendered 3x — proposed default, confirm/override),
  `04_scoreboard.qmd`, `05_report.qmd`.
- `data/splits/` — saved `.rds` train/test objects per arm.
- `output/figures/`, `output/tables/` — rendered artifacts.
- `slides/` — final presentation.
- Later: a `bslib`-based Shiny app wrapping the final per-arm results (scoreboard,
  covariate importance, maybe an interactive prediction panel). Built last, after
  modeling is settled — not before.

## Technical plan — tidymodels
1. Split: `initial_split(strata = outcome_bin)` per arm, fixed seed, saved to
   `data/splits/*.rds`.
2. Recipes (tailored per model, not one-size-fits-all):
   - Elastic net: `step_normalize(all_numeric_predictors())` + `step_dummy(all_nominal_predictors())`.
   - Random forest: minimal recipe (dummy encoding only; no normalization needed).
3. Models:
   - Covariate baseline: `linear_reg(penalty = tune(), mixture = tune()) %>% set_engine("glmnet")`.
   - ML stress test: `rand_forest(trees = tune(), min_n = tune()) %>% set_engine("ranger") %>% set_mode("regression")`.
     Chosen over XGBoost: more robust near defaults on ~100-130 training patients/arm, fewer
     hyperparameters to tune, lower overfitting risk at this sample size. Always call
     `set_mode()` explicitly — tidymodels does not infer it.
   - Keep tuning grids coarse given the small per-arm sample size.
4. Metrics: `metric_set(rmse, mae, rsq)` — define once, reuse for every model/arm.
5. Covariate extraction:
   - Elastic net: `tidy()` on the finalized workflow for non-zero coefficients.
   - Random forest: `vip::vi(method = "permute")`.
   - Key visual: top covariates per method, side by side — overlap = cross-validated signal.
6. AI disclosure: name the actual tool (Claude via Posit Assistant, noting which model tier
   did what), not a generic placeholder.

## Pre-submission checklist (from kickoff slide 49)
- [ ] Code is complete and reproduces all reported results end to end.
- [ ] Split, noise handling, and metrics are explained and justified in slides/code.
- [ ] Key covariates are separated from noise, per arm, with rationale.
- [ ] AI tools used are reported, with the purpose of each.
- [ ] Slides tell one story: workflow -> key covariates -> interpretation -> conclusion.

## Strategic approach
- Start with covariate modeling (interpretable baseline), use ML as a stress test against
  it — not the other way around. Matches the hackathon's own framing question.
- Differentiate arms by mechanism, not just by statistics:
  - Mexiletine (Na+ channel blocker, symptomatic) → myotonia_severity / muscle_strength plausible.
  - Oligonucleotide (DMPK-targeting ASO) → ctg_repeats / somatic_mosaicism_index plausible
    (direct measures of toxic RNA burden the drug targets).
  - Tideglusib (dual-pathway, motor + cognition) → may draw from multiple covariate domains.
- Be visibly honest about noise: with ~20-30 test patients per arm, don't over-read small
  score gaps between methods.
- Visuals: coefficient/importance bar charts, a method-agreement grid, one scoreboard per arm.
  No raw `summary()` dumps or unexplained SHAP/importance plots.
- End with an actionable stratification statement, not just "X was significant."

## Language / tooling
- R (v4.6.1). tidymodels, glmnet, ranger, broom, vip.
- Quarto for reproducible docs (CLI available in this environment).

## Decisions log
- 2026-10-11: Reviewed kickoff deck, set up project scope and workflow rules.
- 2026-10-11: Confirmed R + tidymodels as the stack.
- 2026-10-11: Confirmed data files, ran structural pre-split checks (see Data facts above).
- 2026-10-11: Decided continuous outcome (regression) over binary, for all 3 arms.
- 2026-10-11: Decided random forest (ranger) over XGBoost as the ML comparator.
- 2026-10-11: Connected to team repo (wanguicodes/acop26_hackathon_team9); cloned locally,
  confirmed push access, scaffolding on branch `setup/project-scaffold`.

## AI-use log (for disclosure slide)
- 2026-10-11: Posit Assistant (Claude, mixed model tiers — Sonnet for workflow/code,
  used for summarizing the kickoff deck, project setup, data structural checks, and
  reviewing/correcting an externally-drafted tidymodels technical plan).

## My task scope (Shiny evaluation app) — 2026-10-11
- My deliverable is specifically the evaluation/scoring Shiny app, not the EDA or
  per-arm modeling — other team members own that and will produce the model outputs.
- Built last, once teammates' per-arm results (scoreboard tables, fitted coefficients/
  importance) are ready to plug in.
- Scope follows slides 70-82: ML tail (permutation importance, SHAP, slide 71-73) +
  full Assessment & Evaluation Metrics section (slides 75-82).
- App panels:
  1. Scoreboard per arm: RMSE/MAE/R², untuned vs tuned, same test patients (slide 80).
  2. Covariates included: covariate-model coefficients vs ML importance, side by side,
     overlap = cross-validated signal (slides 72-73, 81).
  3. ROC curve as a secondary panel only: continuous outcome is primary (RMSE/MAE/R²,
     slide 76); ROC/AUC doesn't apply directly to a continuous outcome, so it's shown
     on the *derived* binary responder cutoff from the continuous model's predictions,
     clearly labeled as secondary/illustrative (ties to the binary-label decision above).
- Not pushing to the shared repo until explicitly told to. Branch `setup/project-scaffold`
  stays local for now; connection to the repo is confirmed working (clone + push dry-run OK).
- Plan to publish (e.g. Posit Connect) later, deferred a few days per user.

## App format superseded — 2026-10-11 (later same day)
- Switched from Shiny app to a **Quarto Dashboard** (plotly for interactivity, likely
  Observable JS for the arm selector — no R/Shiny backend needed). Simpler to publish
  as a static site; no live server required for judges to view it.
- ROC curve dropped entirely — continuous outcome only, no binary/responder view anywhere
  in this app. (Supersedes the earlier "ROC as secondary panel" note above.)
- Final scope, two tabs:
  1. Model Evaluation — slide 76 (predicted-vs-observed plot) + RMSE/MAE/R² (all three)
     + slide 81 (covariate comparison table: coefficients vs. ML importance, side by side).
  2. Data/EDA — exploratory, arm-selectable, deliberately minimal (not a pile of filter
     widgets). Exact content TBD (pending user confirmation: likely outcome distribution
     + top 2-3 covariates per arm, not all 18).
- Still waiting on teammates' per-arm model outputs before any dashboard code is written.
