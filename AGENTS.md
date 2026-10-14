# Project memory: ACoP 2026 DM1 Hackathon — Team 9

## Repo
- GitHub: https://github.com/wanguicodes/acop26_hackathon_team9
- Raw files from SharePoint (never edited): `data/source/`
  - `dm1_clinical_trial_dataset.csv` (500 patients x 26 columns)
  - `DM1_Data_Dictionary_Participant.xlsx` (sheets: "Column Dictionary", "Summary")
  - `Kick-offSession_final_participants.pdf`
- Shared train/test split (everyone uses this, do not re-split): `data/derived/split/`
  - `training_shared.csv`, `testing_shared.csv`, `outlier_summary.csv`
  - Produced by `wangui/split_data.R`. Current version: outliers screened out first
    (IQR method, 3x threshold), then 80/20 per arm, stratified on 5 quantile bins
    of that arm's CONTINUOUS outcome, seed 8742.
- Individual team members' working folders: `wangui/`, `inez/`, `natiriva_code/`, `sharvari/`.
- **`deliverable/` — what actually gets shared/presented at the end.** Slides, the
  evaluation dashboard/app, and the code behind them. See `deliverable/README.md`.
- Project skill (auto-discovered by Posit Assistant at the repo root, regardless of
  which part of the repo you're working in): `.posit/assistant/skills/dm1-results-integration/SKILL.md`
  — explains how to get YOUR model's results into the shared dashboard/app.

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

## Outcomes per arm — CONTINUOUS ONLY, no binary/responder modeling
| Arm | Outcome column | Direction |
|---|---|---|
| Mexiletine | `myotonia_change_12wk` | negative = improvement |
| Oligonucleotide | `qmt_change_26wk` | positive = improvement |
| Tideglusib | `tideglusib_composite` (0-20, wk 26) | higher = improvement |

Decided against the binary responder label (`responder_phenotype_*`) entirely:
dichotomizing a continuous outcome throws away power/variance we can't afford to
lose on ~130-170 patients/arm. No ROC curves, no classification metrics anywhere
in this project.

## Data facts (confirmed via structural check)
- 500 rows, 26 columns in the source file, matches the data dictionary exactly.
- Zero missing values in any of the 18 baseline covariates.
- Outcome/responder columns populated only for a patient's assigned arm (NA elsewhere).
- **Exact collinearity: `age = age_at_onset + disease_duration` for all 500 rows**
  (not just correlated — identical). **Team decision (Oct 13): keep `age` and
  `age_at_onset`; drop `disease_duration`** (fully determined by the other two).
  Note for interpretation: with both in a linear model, the `age` coefficient is the
  effect of age at fixed onset age, i.e. effectively a disease-duration effect.
- `albumin` is constant within the Mexiletine arm specifically (confirmed independently
  by Sharvari's `arm_config.py`) — drop it as a predictor for that arm, or rely on a
  zero-variance recipe step (`step_zv()`) to catch it automatically.
- `phenotype` split (61.8%/20.0%/18.2%) matches kickoff slide 40 exactly.

## Covariates (18, minus `disease_duration` and `phenotype` = 16 used)
Team decision (Oct 13), implemented identically in `wangui/linear_regression.R` and
`sharvari/arm_config.py`:
- Numeric (12 used): ctg_repeats, somatic_mosaicism_index, age, age_at_onset,
  myotonia_severity, muscle_strength, pr_interval, cognitive_function, ck_level,
  insulin_resistance, albumin, bmi
- Categorical (4 used): sex (2), family_history (3), cataracts (2), smoking_status (3)
- Excluded: `disease_duration` (= age − age_at_onset) and `phenotype` (onset-age bands,
  overlapping with age_at_onset).
- Mexiletine additionally drops `albumin` (constant in that arm) → 15 covariates there.

## Workflow rules
1. Use the shared split (`data/derived/split/*.csv`) — do not re-split or use a
   different seed. Every method must be scored on the same test patients for the
   scoreboard to be valid.
2. All EDA/model development uses training data only; the test set is touched once,
   for final evaluation.
3. Preprocessing (recipes) is fit on train only, applied to test.
4. One scoreboard per arm: RMSE, MAE, R2. Show untuned and tuned results side by side
   where applicable.
5. Report the covariates selected/ranked by each method, and compare across methods —
   agreement across methods is the strongest evidence of real signal.

## Strategic approach
- Start with covariate modeling (interpretable baseline), use ML as a stress test
  against it — matches the hackathon's own framing question.
- Differentiate arms by mechanism, not just by statistics. Confirmed by the reference
  model's actual results:
  - Mexiletine (Na+ channel blocker, symptomatic) → myotonia_severity ranked #1. Matches.
  - Oligonucleotide (DMPK-targeting ASO) → ctg_repeats ranked #1. Matches (direct measure
    of the toxic RNA burden the drug targets).
  - Tideglusib (dual-pathway, motor + cognition) → cognitive_function ranked #1. Matches.
- Be visibly honest about noise: with ~20-35 test patients per arm, don't over-read
  small score gaps between methods.
- End with an actionable stratification statement, not just "X was significant."

## Technical plan — tidymodels (reference implementation)
See `deliverable/templates/quarto_workflow/model_workflow.qmd` for the actual, running
code. Summary:
- Elastic net (`linear_reg(penalty = tune(), mixture = tune()) %>% set_engine("glmnet")`)
  as the covariate baseline.
- Random forest (`rand_forest(trees = tune(), min_n = tune()) %>% set_engine("ranger",
  importance = "permutation") %>% set_mode("regression")`) as the ML stress test.
  Chosen over XGBoost: more robust near defaults on ~100-130 training patients/arm.
- `metric_set(rmse, mae, rsq)`, defined once, reused for every model/arm.
- Elastic net covariates: `tidy(extract_fit_parsnip(fit), penalty = best_penalty)`.
- RF covariates: `extract_fit_engine(fit)$variable.importance` (ranger's built-in
  permutation importance — simpler and more robust than a custom vip wrapper).
- Results written to `deliverable/results/<arm_slug>/*.csv` — see SKILL.md for the
  exact contract any team member's results need to follow.
- This reference implementation already ran end-to-end for all 3 arms and produced
  real (if simple) baseline results. It's a fallback, not the final answer — anyone's
  own better methodology should replace it.
- **Out of date:** the reference workflow was built under the old covariate rule (drop
  `age`, keep `disease_duration` + `phenotype`). Rerun it with the agreed 16-covariate set
  above so the scoreboard compares methods on the same covariates.

## Team repo state — division of labor
- **wangui/**: shared split + full-dataset EDA figures/tables + stepwise linear regression
  per arm (`wangui/linear_regression.R`; partial F-tests, training-only metrics written to
  `data/derived/linear_regression_training_*.csv`, per-arm outputs in
  `wangui/linear_regression/`). Test-set predictions not yet written to `deliverable/results/`.
- **inez/**: EDA by arm (boxplots, within-arm correlation heatmaps) + `findings_summary.Rmd`/html.
- **natiriva_code/**: more EDA + univariate screening (Mann-Whitney, Spearman). Note:
  her script reads `data/dm1_clinical_trial_dataset.csv` (should be `data/source/...`) —
  a path bug on her end, not ours to fix.
- **sharvari/**: ML pipeline framework (Python/scikit-learn + xgboost) — nested CV,
  Random Forest + XGBoost vs. a dummy baseline, SHAP, noise-probe columns for signal
  validation. Well-built, matches our metrics/leakage discipline. Executed for all 3 arms
  (full run, not quick run); outputs committed in `sharvari/outputs/<arm>/`
  (`test_metrics.csv`, `test_predictions.csv`, `shap_ranking.csv`, CV summaries).

## Pre-submission checklist (from kickoff slide 49)
- [ ] Code is complete and reproduces all reported results end to end.
- [ ] Split, noise handling, and metrics are explained and justified in slides/code.
- [ ] Key covariates are separated from noise, per arm, with rationale.
- [ ] AI tools used are reported, with the purpose of each (see AI-use log below).
- [ ] Slides tell one story: workflow -> key covariates -> interpretation -> conclusion.

## Language / tooling
- R (R 4.6.1) for the dashboard/app/workflow templates: tidymodels, glmnet, ranger,
  broom, plotly, DT, bslib, rprojroot, Quarto (CLI available in this environment).
- Python (scikit-learn, xgboost, shap) for Sharvari's ML pipeline — the hackathon
  explicitly allows mixed languages across a team; the `deliverable/results/*.csv`
  contract is how R and Python outputs meet in one place.
- Explicitly decided against vetiver/plumber: no one needs a live model-serving API
  for this deliverable, and it would add setup complexity/confusion for no visible
  benefit in judging.

## Decisions log
- Reviewed kickoff deck, set up project scope and workflow rules.
- Confirmed R + tidymodels as the stack for the dashboard/app side.
- Confirmed data files, ran structural pre-split checks (see Data facts above).
- Decided continuous outcome (regression) over binary, for all 3 arms — final, confirmed twice.
- Decided random forest (ranger) over XGBoost as our own reference ML comparator.
- Connected to team repo; adopted the team's shared split rather than deriving our own.
- Scoped personal task to the evaluation dashboard/app, built last — but ended up also
  building a reference covariate+ML implementation to unblock the dashboard and fill
  an observed gap (no one else had covariate-model numbers yet).
- Decided against vetiver/plumber for the same reason as above (no deployment need here).
- Restructured all shareable output into `deliverable/` (templates, results, README)
  so it's obvious to the team what's "the submission" vs. working files. Skill file
  stays at the repo root (`.posit/assistant/skills/`) since that's where Posit
  Assistant auto-discovers project skills — moving it into `deliverable/` would break
  auto-discovery for teammates working elsewhere in the repo.
- Pushed directly to `main`, matching how the rest of the team has been working
  (no branch/PR workflow in use).
- Covariate set changed (Oct 13, team decision): keep `age` + `age_at_onset`, drop
  `disease_duration` and `phenotype`. Supersedes the earlier "drop `age`" rule.

## AI-use log (for disclosure slide)
- Posit Assistant (Claude, mixed model tiers — used for: summarizing the kickoff deck;
  project setup; data structural checks; reviewing/correcting an externally-drafted
  tidymodels technical plan; building and testing the Quarto dashboard, Shiny app, and
  reference modeling workflow templates; writing the project skill file that helps
  teammates' own Posit Assistant sessions wire their results into the shared dashboard;
  reviewing teammates' code for alignment with the agreed workflow).
