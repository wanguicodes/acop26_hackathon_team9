################################################################################
# linear_regression.R  --  Stepwise linear regression per treatment arm
#                          ACoP 2026 DM1 Hackathon (Team 9)
#
# SCOPE
#   For each arm, fit a linear model for that arm's continuous efficacy
#   endpoint using ONLY the training set (data/derived/split/training_shared.csv).
#
#   Arm              Outcome                 Candidate covariates
#   ---------------  ----------------------  -------------------------------
#   Mexiletine       myotonia_change_12wk    15 (no albumin)
#   Oligonucleotide  qmt_change_26wk         16
#   Tideglusib       tideglusib_composite    16
#
# COVARIATE SELECTION
#   1. Forward inclusion (p < p_enter): starting from the intercept-only model,
#      each remaining candidate is added individually and tested with a
#      partial F-test (add1). The covariate with the smallest p-value is added
#      if p < p_enter; repeat until no candidate qualifies.
#   2. Backward elimination (p < p_stay): starting from the forward model, each
#      covariate is dropped individually and tested with a partial F-test
#      (drop1). The covariate with the largest p-value is removed if
#      p >= 0.001; repeat until all remaining covariates have p < p_stay.
#   Categorical covariates are tested as whole terms (one F-test across all
#   levels), not as individual dummy variables.
#
# OUTPUTS
#   wangui/linear_regression/
#     <arm>_coefficients.csv          estimates, 95% CI, p-values (final model)
#     <arm>_stepwise_log.csv          every forward/backward step and p-value
#     <arm>_diagnostics.csv           per-patient fitted, residuals, studentized
#                                     residuals, leverage, Cook's distance
#     <arm>_diagnostics.png           residual / QQ / studentized / leverage /
#                                     Cook's distance plots
#   data/derived/
#     linear_regression_training_stats.csv       RMSE, MAE, R-squared per arm
#     linear_regression_training_covariates.csv  final covariates per arm
#     linear_regression_training_pred_obs.csv    observed vs predicted
################################################################################

set.seed(1234)

### Packages -------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(broom)
  library(gridExtra)
})

### Paths ----------------------------------------------------------------------
root <- tryCatch(here::here(), error = function(e) getwd())
train_path <- file.path(root, "data", "derived", "split", "training_shared.csv")
if (!file.exists(train_path)) {
  stop("Could not find ", train_path, ". Run wangui/split_data.R first.")
}
out_dir     <- file.path(root, "wangui", "linear_regression")
derived_dir <- file.path(root, "data", "derived")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

theme_set(theme_bw(base_size = 10))

### Data -----------------------------------------------------------------------
train <- read_csv(train_path, show_col_types = FALSE)

cat_cov <- c("sex", "family_history", "cataracts", "smoking_status")
train <- train |> mutate(across(all_of(cat_cov), factor))

### Model specification per arm ------------------------------------------------
base_cov <- c("age", "age_at_onset", "ctg_repeats", "somatic_mosaicism_index",
              "myotonia_severity", "muscle_strength", "cognitive_function",
              "ck_level", "pr_interval", "insulin_resistance")
tail_cov <- c("bmi", "sex", "family_history", "cataracts", "smoking_status")

spec <- list(
  Mexiletine = list(
    outcome    = "myotonia_change_12wk",
    candidates = c(base_cov, tail_cov)
  ),
  Oligonucleotide = list(
    outcome    = "qmt_change_26wk",
    candidates = c(base_cov, "albumin", tail_cov)
  ),
  Tideglusib = list(
    outcome    = "tideglusib_composite",
    candidates = c(base_cov, "albumin", tail_cov)
  )
)

p_enter <- 0.5 
p_stay  <- 0.1 

### Stepwise selection ---------------------------------------------------------
# Formulas are built in the caller's environment so add1()/drop1(), which
# re-evaluate the model data, can find `d`.
make_formula <- function(outcome, terms, env = parent.frame()) {
  rhs <- if (length(terms) == 0) "1" else paste(terms, collapse = " + ")
  as.formula(paste(outcome, "~", rhs), env = env)
}

stepwise_select <- function(d, outcome, candidates) {
  log <- list()
  selected <- character(0)

  # Forward inclusion
  repeat {
    remaining <- setdiff(candidates, selected)
    if (length(remaining) == 0) break
    fit <- lm(make_formula(outcome, selected), data = d)
    tests <- add1(fit, scope = make_formula(outcome, c(selected, remaining)),
                  test = "F") # F-test compares residual sum of squares between models
    tests <- tests[rownames(tests) != "<none>", , drop = FALSE]
    p <- setNames(tests[["Pr(>F)"]], rownames(tests))
    best <- names(which.min(p))
    added <- p[[best]] < p_enter
    log[[length(log) + 1]] <- tibble(
      phase = "forward", step = sum(log_phase(log) == "forward") + 1,
      covariate = best, p_value = p[[best]], threshold = p_enter,
      action = if (added) "added" else "stop (none p < 0.01)"
    )
    if (!added) break
    selected <- c(selected, best)
  }
  forward_terms <- selected

  # Backward elimination from the forward model
  repeat {
    if (length(selected) == 0) break
    fit <- lm(make_formula(outcome, selected), data = d)
    tests <- drop1(fit, test = "F")
    tests <- tests[rownames(tests) != "<none>", , drop = FALSE]
    p <- setNames(tests[["Pr(>F)"]], rownames(tests))
    worst <- names(which.max(p))
    removed <- p[[worst]] >= p_stay
    log[[length(log) + 1]] <- tibble(
      phase = "backward", step = sum(log_phase(log) == "backward") + 1,
      covariate = worst, p_value = p[[worst]], threshold = p_stay,
      action = if (removed) "removed" else "stop (all p < 0.001)"
    )
    if (!removed) break
    selected <- setdiff(selected, worst)
  }

  list(forward_terms = forward_terms, final_terms = selected,
       log = bind_rows(log))
}

log_phase <- function(log) vapply(log, function(x) x$phase, character(1))

### Diagnostics ----------------------------------------------------------------
diagnostics_table <- function(fit, d) {
  tibble(
    patient_id    = d$patient_id,
    observed      = model.response(model.frame(fit)),
    fitted        = fitted(fit),
    residual      = residuals(fit),
    std_residual  = rstandard(fit),
    stud_residual = rstudent(fit),
    leverage      = hatvalues(fit),
    cooks_d       = cooks.distance(fit)
  )
}

diagnostics_plot <- function(diag, arm, outcome, n_par) {
  n <- nrow(diag)
  lev_cut  <- 2 * n_par / n
  cook_cut <- 4 / n

  p1 <- ggplot(diag, aes(fitted, residual)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE, linewidth = 0.6) +
    labs(x = "Fitted", y = "Residual", title = "Residuals vs fitted")

  p2 <- ggplot(diag, aes(sample = std_residual)) +
    stat_qq(alpha = 0.6) +
    stat_qq_line(linetype = "dashed") +
    labs(x = "Theoretical quantiles", y = "Standardized residual",
         title = "Normal Q-Q")

  p3 <- ggplot(diag, aes(fitted, stud_residual)) +
    geom_hline(yintercept = c(-3, 3), linetype = "dashed") +
    geom_hline(yintercept = 0) +
    geom_point(alpha = 0.6) +
    labs(x = "Fitted", y = "Studentized residual",
         title = "Studentized residuals (|r| > 3 dashed)")

  p4 <- ggplot(diag, aes(seq_len(n), leverage)) +
    geom_hline(yintercept = lev_cut, linetype = "dashed") +
    geom_point(alpha = 0.6) +
    labs(x = "Observation index", y = "Leverage (hat value)",
         title = sprintf("Leverage (2p/n = %.3f dashed)", lev_cut))

  p5 <- ggplot(diag, aes(seq_len(n), cooks_d)) +
    geom_hline(yintercept = cook_cut, linetype = "dashed") +
    geom_segment(aes(xend = seq_len(n), yend = 0)) +
    labs(x = "Observation index", y = "Cook's distance",
         title = sprintf("Cook's distance (4/n = %.3f dashed)", cook_cut))

  p6 <- ggplot(diag, aes(leverage, stud_residual, size = cooks_d)) +
    geom_hline(yintercept = c(-3, 3), linetype = "dashed") +
    geom_vline(xintercept = lev_cut, linetype = "dashed") +
    geom_point(alpha = 0.5) +
    scale_size_area(max_size = 5, name = "Cook's D") +
    labs(x = "Leverage", y = "Studentized residual",
         title = "Residuals vs leverage")

  arrangeGrob(p1, p2, p3, p4, p5, p6, ncol = 3,
              top = paste0(arm, " (", outcome, "): final model diagnostics, n = ", n))
}

### Fit per arm ----------------------------------------------------------------
results <- imap(spec, function(s, arm) {
  d <- train |>
    filter(treatment_arm == arm) |>
    select(patient_id, all_of(c(s$outcome, s$candidates))) |>
    drop_na()
  message(arm, ": n = ", nrow(d), " complete cases")

  sel <- stepwise_select(d, s$outcome, s$candidates)
  fit <- lm(make_formula(s$outcome, sel$final_terms), data = d)

  coefs <- tidy(fit, conf.int = TRUE, conf.level = 0.95) |>
    mutate(treatment_arm = arm, outcome = s$outcome, .before = 1)
  diag <- diagnostics_table(fit, d)

  stem <- tolower(arm)
  write_csv(coefs, file.path(out_dir, paste0(stem, "_coefficients.csv")))
  write_csv(sel$log |> mutate(treatment_arm = arm, .before = 1),
            file.path(out_dir, paste0(stem, "_stepwise_log.csv")))
  write_csv(diag, file.path(out_dir, paste0(stem, "_diagnostics.csv")))
  ggsave(file.path(out_dir, paste0(stem, "_diagnostics.png")),
         diagnostics_plot(diag, arm, s$outcome, length(coef(fit))),
         width = 14, height = 8, dpi = 150)

  list(arm = arm, outcome = s$outcome, fit = fit, d = d, sel = sel,
       coefs = coefs, diag = diag)
})

### Derived outputs ------------------------------------------------------------
stats <- map_dfr(results, function(r) {
  obs <- r$diag$observed
  pred <- r$diag$fitted
  tibble(
    treatment_arm = r$arm,
    outcome       = r$outcome,
    n             = length(obs),
    n_covariates  = length(r$sel$final_terms),
    RMSE          = sqrt(mean((obs - pred)^2)),
    MAE           = mean(abs(obs - pred)),
    R_squared     = summary(r$fit)$r.squared
  )
})

covariates <- map_dfr(results, function(r) {
  terms <- r$sel$final_terms
  tibble(
    treatment_arm = r$arm,
    outcome       = r$outcome,
    covariate     = if (length(terms) == 0) NA_character_ else terms
  )
})

pred_obs <- map_dfr(results, function(r) {
  tibble(
    treatment_arm = r$arm,
    patient_id    = r$diag$patient_id,
    outcome       = r$outcome,
    observed      = r$diag$observed,
    predicted     = r$diag$fitted
  )
})

### Predicted vs observed plot (one panel per treatment arm) ------------------
# Panels use free scales (each arm's outcome is on a different scale), but
# within a panel the x- and y-axis limits are forced to match so the line of
# unity is a true diagonal. This is done by adding invisible corner points
# (geom_blank) at the shared min/max of observed and predicted per panel.
pred_obs <- pred_obs |> mutate(panel = paste0(treatment_arm, "\n(", outcome, ")"))
panel_lims <- pred_obs |>
  group_by(panel) |>
  summarise(lo = min(c(observed, predicted)), hi = max(c(observed, predicted)), .groups = "drop") |>
  pivot_longer(cols = c(lo, hi), values_to = "val")

pred_obs_plot <- ggplot(pred_obs, aes(observed, predicted)) +
  geom_blank(data = panel_lims, aes(x = val, y = val)) +
  geom_abline(slope = 1, intercept = 0, colour = "blue", linewidth = 0.6) +
  geom_point(alpha = 0.6) +
  facet_wrap(~panel, nrow = 1, scales = "free") +
  labs(x = "Observed", y = "Predicted",
       title = "Training set: predicted vs. observed, by treatment arm",
       subtitle = "Blue line = line of unity (predicted = observed)")
ggsave(file.path(out_dir, "pred_vs_obs.png"), pred_obs_plot,
       width = 12, height = 4.5, dpi = 150)

write_csv(stats,      file.path(derived_dir, "linear_regression_training_stats.csv"))
write_csv(covariates, file.path(derived_dir, "linear_regression_training_covariates.csv"))
write_csv(pred_obs,   file.path(derived_dir, "linear_regression_training_pred_obs.csv"))

### Console summary ------------------------------------------------------------
walk(results, function(r) {
  message("\n== ", r$arm, " (", r$outcome, ") ==")
  message("Forward model: ",
          if (length(r$sel$forward_terms)) paste(r$sel$forward_terms, collapse = ", ") else "(none)")
  message("Final model:   ",
          if (length(r$sel$final_terms)) paste(r$sel$final_terms, collapse = ", ") else "(intercept only)")
})
print(stats)
