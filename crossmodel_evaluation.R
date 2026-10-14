################################################################################
# crossmodel_evaluation.R  --  Compare test-set performance across models
#                              ACoP 2026 DM1 Hackathon (Team 9)
#
# SCOPE
#   Three models per treatment arm, all scored on the same held-out test
#   patients (data/derived/split/testing_shared.csv, 35 per arm):
#     - Linear regression (stepwise)   -> wangui/evaluate_on_test.R
#     - The top-2 tuned models per arm -> sharvari/ (see top_models below,
#       taken from sharvari/outputs_extended_ml_models/best_two_models_per_arm.csv)
#         Mexiletine:      Elastic net (tuned), Random forest (tuned)
#         Oligonucleotide: Elastic net (tuned), SVR (tuned)
#         Tideglusib:      XGBoost (tuned),     Random forest (tuned)
#   Other models in Sharvari's outputs (null, *_untuned, ...) are ignored.
#
# METRICS
#   RMSE, MAE and R_squared (1 - SSE/SST) are recomputed here from each
#   model's observed/predicted values with one formula, so every model is
#   scored identically. They are checked against the metrics each source
#   reported; any disagreement beyond rounding triggers a warning.
#
# INPUTS
#   data/derived/linear_regression_test_pred_obs.csv   (run wangui/evaluate_on_test.R first)
#   data/derived/linear_regression_test_stats.csv
#   sharvari/outputs/<arm>/test_predictions.csv                       RF, XGBoost
#   sharvari/outputs_extended_ml_models/<arm>/test_predictions.csv    Elastic net, SVR
#   sharvari/outputs_extended_ml_models/<arm>/test_metrics_combined.csv
#
# OUTPUTS (data/derived/crossmodel_evaluation/)
#   crossmodel_test_stats.csv           long: arm x model, RMSE / MAE / R_squared
#   crossmodel_test_stats_wide.csv      one row per arm x metric, model roles as columns
#   crossmodel_test_pred_obs.csv        long: per patient x model, observed / predicted
#   crossmodel_test_pred_obs_wide.csv   per patient, one predicted column per model role
#   crossmodel_reported_vs_recomputed.csv  check against source-reported metrics
#   pred_vs_obs_test.png                predicted vs observed, colored by method
################################################################################

suppressPackageStartupMessages({
  library(tidyverse)
})

root <- tryCatch(here::here(), error = function(e) getwd())
out_dir <- file.path(root, "data", "derived", "crossmodel_evaluation")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

arms <- c("Mexiletine", "Oligonucleotide", "Tideglusib")
outcome_of <- c(Mexiletine      = "myotonia_change_12wk",
                Oligonucleotide = "qmt_change_26wk",
                Tideglusib      = "tideglusib_composite")

# Top-2 tuned models per arm (rank 1, rank 2).
top_models <- list(Mexiletine      = c("enet_tuned", "rf_tuned"),
                   Oligonucleotide = c("enet_tuned", "svr_tuned"),
                   Tideglusib      = c("xgb_tuned",  "rf_tuned"))

model_labels <- c(linear_regression = "Linear regression (stepwise)",
                  enet_tuned        = "Elastic net (tuned)",
                  rf_tuned          = "Random forest (tuned)",
                  svr_tuned         = "SVR, RBF (tuned)",
                  xgb_tuned         = "XGBoost (tuned)")

# One colour per method, consistent across arms (Okabe-Ito, colour-blind safe).
model_colours <- setNames(c("#000000", "#E69F00", "#0072B2", "#CC79A7", "#009E73"),
                          model_labels)
role_levels   <- c("Linear regression", "Rank 1 model", "Rank 2 model")

model_roles <- imap_dfr(top_models, \(m, arm) tibble(
  treatment_arm = arm,
  model = c("linear_regression", m),
  role  = role_levels
))

### Read predictions -----------------------------------------------------------
lr_path <- file.path(root, "data", "derived", "linear_regression_test_pred_obs.csv")
if (!file.exists(lr_path)) stop("Missing ", lr_path, ". Run wangui/evaluate_on_test.R first.")

lr_pred <- read_csv(lr_path, show_col_types = FALSE) |>
  transmute(treatment_arm, patient_id, outcome, model = "linear_regression",
            observed, predicted)

sharvari_dirs <- c("outputs", "outputs_extended_ml_models")

read_sharvari <- function(arm, file, dirs = sharvari_dirs) {
  paths <- file.path(root, "sharvari", dirs, tolower(arm), file)
  paths <- paths[file.exists(paths)]
  if (length(paths) == 0) stop("Missing ", file, " for ", arm)
  map_dfr(paths, read_csv, show_col_types = FALSE) |>
    filter(model %in% top_models[[arm]]) |>
    distinct() |>
    mutate(treatment_arm = arm, .before = 1)
}

ml_pred <- map_dfr(arms, read_sharvari, file = "test_predictions.csv") |>
  mutate(outcome = unname(outcome_of[treatment_arm])) |>
  select(treatment_arm, patient_id, outcome, model, observed, predicted)

pred_obs <- bind_rows(lr_pred, ml_pred) |>
  left_join(model_roles, by = c("treatment_arm", "model")) |>
  mutate(treatment_arm = factor(treatment_arm, levels = arms),
         role = factor(role, levels = role_levels),
         model_label = unname(model_labels[model])) |>
  arrange(treatment_arm, role, patient_id)

# Every model must be scored on the same test patients with the same observed values,
# and each (arm, patient, model) must appear exactly once.
if (anyDuplicated(select(pred_obs, treatment_arm, patient_id, model)))
  stop("Duplicate predictions for the same patient and model.")
consistency <- pred_obs |>
  group_by(treatment_arm, patient_id) |>
  summarise(n_models = n_distinct(model),
            obs_range = max(observed) - min(observed), .groups = "drop")
if (any(consistency$n_models != length(role_levels)))
  stop("Not every test patient has predictions from all three models.")
if (any(consistency$obs_range > 1e-8))
  stop("Observed values differ between models for the same patient.")

### Metrics (recomputed uniformly) ---------------------------------------------
stats <- pred_obs |>
  group_by(treatment_arm, outcome, role, model, model_label) |>
  summarise(
    n         = n(),
    RMSE      = sqrt(mean((observed - predicted)^2)),
    MAE       = mean(abs(observed - predicted)),
    R_squared = 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2),
    .groups = "drop"
  )

stats_wide <- stats |>
  pivot_longer(c(RMSE, MAE, R_squared), names_to = "metric") |>
  mutate(value = sprintf("%.3f  %s", value, model_label)) |>
  select(treatment_arm, outcome, metric, role, value) |>
  pivot_wider(names_from = role, values_from = value) |>
  mutate(metric = factor(metric, levels = c("RMSE", "MAE", "R_squared"))) |>
  arrange(treatment_arm, metric)

pred_obs_wide <- pred_obs |>
  select(treatment_arm, patient_id, outcome, observed, model, predicted) |>
  pivot_wider(names_from = model, values_from = predicted, names_prefix = "predicted_")

### Check against source-reported metrics --------------------------------------
lr_reported <- read_csv(file.path(root, "data", "derived", "linear_regression_test_stats.csv"),
                        show_col_types = FALSE) |>
  transmute(treatment_arm, model = "linear_regression", RMSE, MAE, R_squared)
ml_reported <- map_dfr(arms, read_sharvari, file = "test_metrics_combined.csv",
                       dirs = "outputs_extended_ml_models") |>
  transmute(treatment_arm, model, RMSE, MAE, R_squared = R2)

reported_check <- bind_rows(lr_reported, ml_reported) |>
  pivot_longer(c(RMSE, MAE, R_squared), names_to = "metric", values_to = "reported") |>
  left_join(stats |>
              mutate(treatment_arm = as.character(treatment_arm)) |>
              pivot_longer(c(RMSE, MAE, R_squared), names_to = "metric", values_to = "recomputed") |>
              select(treatment_arm, model, metric, recomputed),
            by = c("treatment_arm", "model", "metric")) |>
  mutate(abs_diff = abs(reported - recomputed))
# Sharvari's metrics are rounded to 3 decimals.
if (any(is.na(reported_check$abs_diff)) || any(reported_check$abs_diff > 0.001))
  warning("Some reported metrics are missing or disagree with recomputed values; ",
          "see crossmodel_reported_vs_recomputed.csv")

### Write tables ---------------------------------------------------------------
write_csv(stats,          file.path(out_dir, "crossmodel_test_stats.csv"))
write_csv(stats_wide,     file.path(out_dir, "crossmodel_test_stats_wide.csv"))
write_csv(pred_obs,       file.path(out_dir, "crossmodel_test_pred_obs.csv"))
write_csv(pred_obs_wide,  file.path(out_dir, "crossmodel_test_pred_obs_wide.csv"))
write_csv(reported_check, file.path(out_dir, "crossmodel_reported_vs_recomputed.csv"))

### Predicted vs observed plot -------------------------------------------------
# Free scales across panels, but matching x/y limits within each panel so the
# line of unity is a true diagonal (invisible corner points via geom_blank).
panel_levels <- paste0(arms, "\n(", outcome_of[arms], ")")
plot_df <- pred_obs |>
  mutate(panel = factor(paste0(treatment_arm, "\n(", outcome, ")"), levels = panel_levels),
         model_label = factor(model_label, levels = model_labels))

panel_range <- plot_df |>
  group_by(panel) |>
  summarise(lo = min(c(observed, predicted)), hi = max(c(observed, predicted)),
            .groups = "drop")
panel_lims <- panel_range |> pivot_longer(c(lo, hi), values_to = "val")

r2_labels <- stats |>
  mutate(panel = factor(paste0(treatment_arm, "\n(", outcome, ")"), levels = panel_levels),
         model_label = factor(model_label, levels = model_labels),
         label = sprintf("R\u00b2 = %.2f, RMSE = %.2f", R_squared, RMSE)) |>
  left_join(panel_range, by = "panel") |>
  mutate(x = lo + 0.03 * (hi - lo),
         y = hi - (as.integer(role) - 1) * 0.07 * (hi - lo))

pred_obs_plot <- ggplot(plot_df, aes(observed, predicted, colour = model_label)) +
  geom_blank(data = panel_lims, aes(x = val, y = val), inherit.aes = FALSE) +
  geom_abline(slope = 1, intercept = 0, colour = "grey40", linetype = "dashed") +
  geom_point(alpha = 0.7, size = 1.8) +
  geom_label(data = r2_labels, aes(x = x, y = y, label = label),
             hjust = 0, vjust = 1, size = 3.1, fontface = "bold", show.legend = FALSE,
             fill = scales::alpha("white", 0.85), linewidth = 0,
             label.padding = unit(0.1, "lines")) +
  facet_wrap(~panel, nrow = 1, scales = "free") +
  scale_colour_manual(values = model_colours) +
  guides(colour = guide_legend(override.aes = list(size = 3.5, alpha = 1))) +
  labs(x = "Observed", y = "Predicted", colour = NULL,
       title = "Test set: predicted vs. observed, linear regression vs. top-2 tuned models per arm",
       subtitle = "Dashed line = line of unity (predicted = observed); n = 35 test patients per arm") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", legend.text = element_text(size = 12))

ggsave(file.path(out_dir, "pred_vs_obs_test.png"), pred_obs_plot,
       width = 14, height = 5.5, dpi = 150)

print(stats_wide, width = Inf)
