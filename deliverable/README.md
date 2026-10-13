# deliverable/

This is what gets shared/presented at the end: the slides, the evaluation
dashboard, and the code behind them. If you only look at one folder in this
repo to understand "what are we actually submitting," this is it.

## What's here

- `templates/dashboard/index.qmd` -- the main Quarto dashboard (predicted-vs-
  observed plots, scoreboard, covariate comparison table, per arm). Render with
  `quarto render deliverable/templates/dashboard/index.qmd`.
- `templates/shiny_app/app.R` -- a Shiny alternative to the dashboard, same
  data, live reactivity. Run with `shiny::runApp("deliverable/templates/shiny_app")`.
- `templates/quarto_workflow/model_workflow.qmd` -- a parameterized, runnable
  reference model (elastic net vs. random forest) for one arm at a time. This
  is also a fallback: it already wrote real results below, so the dashboard
  has something to show even before everyone's final models are in.
- `results/<arm_slug>/{scoreboard,covariates,predictions}.csv` -- the actual
  numbers the dashboard/app read. `arm_slug` is `mexiletine`, `oligonucleotide`,
  or `tideglusib`.

## If you just finished your model and want it in the dashboard

Ask your Posit Assistant session (in this repo) to "get my results into the
dashboard." It should automatically find and follow
`.posit/assistant/skills/dm1-results-integration/SKILL.md`, which has the
exact file format expected. You don't need to touch the dashboard/app code
yourself -- just get your numbers into `results/<arm_slug>/*.csv` in that
format and re-render.

See the repo root `AGENTS.md` for the full project context (data facts,
workflow rules, strategic approach).
