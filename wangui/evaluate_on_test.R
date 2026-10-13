################################################################################
# evaluate_on_test.R  --  Score the final stepwise linear models on the test set
#                         ACoP 2026 DM1 Hackathon (Team 9)
#
# SCOPE
#   Sources wangui/linear_regression.R (which selects and fits the final model
#   per arm on the TRAINING set only), then applies each arm's final model,
#   unchanged, to that arm's TEST patients (data/derived/split/testing_shared.csv).
#   No refitting or reselection happens on the test data.
#
# METRICS (per arm, test set)
#   RMSE, MAE, and R_squared = 1 - SSE/SST on the test outcomes. Unlike the
#   training R-squared this can be negative, which means the model predicts
#   worse than the test-set mean.
#
# OUTPUTS
#   wangui/evaluate_on_test/
#     <arm>_test_diagnostics.csv   per-patient observed, predicted, residual
#     <arm>_test_diagnostics.png   residuals vs predicted, residual Q-Q,
#                                  residuals vs observed, residual histogram
#     pred_vs_obs_test.png         predicted vs observed, one panel per arm
#   data/derived/
#     linear_regression_test_stats.csv           RMSE, MAE, R_squared per arm
#     linear_regression_test_pred_obs.csv        observed vs predicted per patient
#     linear_regression_train_vs_test_stats.csv  training and test metrics together
################################################################################

set.seed(1234)

### Fit final models on training data ------------------------------------------
# Sourced into its own environment so its objects don't clutter this script.
root <- tryCatch(here::here(), error = function(e) getwd())
lr <- new.env()
source(file.path(root, "wangui", "linear_regression.R"), local = lr)

suppressPackageStartupMessages({
  library(tidyverse)
  library(gridExtra)
})
theme_set(theme_bw(base_size = 10))

### Paths ----------------------------------------------------------------------
test_path <- file.path(root, "data", "derived", "split", "testing_shared.csv")
if (!file.exists(test_path)) {
  stop("Could not find ", test_path, ". Run wangui/split_data.R first.")
}
out_dir     <- file.path(root, "wangui", "evaluate_on_test")
derived_dir <- file.path(root, "data", "derived")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

### Test data ------------------------------------------------------------------
# Categorical covariates use the training factor levels so predict() codes
# them the same way the model was fit.
test <- read_csv(test_path, show_col_types = FALSE)
for (v in lr$cat_cov) {
  test[[v]] <- factor(test[[v]], levels = levels(lr$train[[v]]))
}

### Predict and evaluate per arm -----------------------------------------------
metrics <- function(obs, pred) {
  tibble(
    n         = length(obs),
    RMSE      = sqrt(mean((obs - pred)^2)),
    MAE       = mean(abs(obs - pred)),
    R_squared = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
  )
}

test_diag_plot <- function(diag, arm, outcome) {
  p1 <- ggplot(diag, aes(predicted, residual)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(alpha = 0.6) +
    labs(x = "Predicted", y = "Residual (observed - predicted)",
         title = "Residuals vs predicted")
  p2 <- ggplot(diag, aes(sample = residual)) +
    stat_qq(alpha = 0.6) +
    stat_qq_line(linetype = "dashed") +
    labs(x = "Theoretical quantiles", y = "Residual", title = "Residual Q-Q")
  p3 <- ggplot(diag, aes(observed, residual)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(alpha = 0.6) +
    labs(x = "Observed", y = "Residual", title = "Residuals vs observed")
  p4 <- ggplot(diag, aes(residual)) +
    geom_histogram(bins = 15, colour = "white") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    labs(x = "Residual", y = "Count", title = "Residual distribution")
  arrangeGrob(p1, p2, p3, p4, ncol = 2,
              top = paste0(arm, " (", outcome, "): test set residuals, n = ", nrow(diag)))
}

test_results <- imap(lr$results, function(r, arm) {
  d <- test |> filter(treatment_arm == arm)
  pred <- predict(r$fit, newdata = d)
  diag <- tibble(
    treatment_arm = arm,
    patient_id    = d$patient_id,
    outcome       = r$outcome,
    observed      = d[[r$outcome]],
    predicted     = unname(pred),
    residual      = observed - predicted
  )
  if (anyNA(diag$predicted)) warning(arm, ": some test predictions are NA")

  stem <- tolower(arm)
  write_csv(diag, file.path(out_dir, paste0(stem, "_test_diagnostics.csv")))
  ggsave(file.path(out_dir, paste0(stem, "_test_diagnostics.png")),
         test_diag_plot(diag, arm, r$outcome), width = 10, height = 8, dpi = 150)

  list(diag = diag,
       stats = metrics(diag$observed, diag$predicted) |>
         mutate(treatment_arm = arm, outcome = r$outcome,
                n_covariates = length(r$sel$final_terms), .before = 1))
})

test_stats    <- map_dfr(test_results, "stats")
test_pred_obs <- map_dfr(test_results, ~ .x$diag |> select(-residual))

train_vs_test <- bind_rows(
  lr$stats |> mutate(set = "train"),
  test_stats |> mutate(set = "test")
) |>
  select(treatment_arm, outcome, set, n, n_covariates, RMSE, MAE, R_squared) |>
  arrange(treatment_arm, desc(set))

write_csv(test_stats,    file.path(derived_dir, "linear_regression_test_stats.csv"))
write_csv(test_pred_obs, file.path(derived_dir, "linear_regression_test_pred_obs.csv"))
write_csv(train_vs_test, file.path(derived_dir, "linear_regression_train_vs_test_stats.csv"))

### Predicted vs observed plot (same layout as the training version) -----------
plot_df <- test_pred_obs |> mutate(panel = paste0(treatment_arm, "\n(", outcome, ")"))
panel_lims <- plot_df |>
  group_by(panel) |>
  summarise(lo = min(c(observed, predicted)), hi = max(c(observed, predicted)),
            .groups = "drop") |>
  pivot_longer(cols = c(lo, hi), values_to = "val")

pred_obs_test_plot <- ggplot(plot_df, aes(observed, predicted)) +
  geom_blank(data = panel_lims, aes(x = val, y = val)) +
  geom_abline(slope = 1, intercept = 0, colour = "blue", linewidth = 0.6) +
  geom_point(alpha = 0.6) +
  facet_wrap(~panel, nrow = 1, scales = "free") +
  labs(x = "Observed", y = "Predicted",
       title = "Test set: predicted vs. observed, by treatment arm",
       subtitle = "Blue line = line of unity (predicted = observed)")
ggsave(file.path(out_dir, "pred_vs_obs_test.png"), pred_obs_test_plot,
       width = 12, height = 4.5, dpi = 150)

print(train_vs_test)
