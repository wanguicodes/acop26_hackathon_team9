library(tidyverse)

df <- read_csv("data/dm1_clinical_trial_dataset.csv")

glimpse(df)

# ---------------------------------------------------------------------------
# EDA by treatment arm: continuous covariates vs responder status / outcome
# ---------------------------------------------------------------------------

continuous_covariates <- c(
  "age", "age_at_onset", "albumin", "bmi", "ck_level", "cognitive_function",
  "ctg_repeats", "disease_duration", "insulin_resistance", "muscle_strength",
  "myotonia_severity", "pr_interval", "somatic_mosaicism_index"
)

categorical_covariates <- c("phenotype", "sex", "family_history", "cataracts", "smoking_status")

# Each arm has its own continuous outcome + derived binary responder label
arm_config <- list(
  Mexiletine = list(outcome = "myotonia_change_12wk", responder = "responder_phenotype_mex"),
  Oligonucleotide = list(outcome = "qmt_change_26wk", responder = "responder_phenotype_oligo"),
  Tideglusib = list(outcome = "tideglusib_composite", responder = "responder_phenotype_tidge")
)

# 1) Boxplots: continuous covariates by responder status, one arm at a time
plot_boxplots <- function(arm_name) {
  cfg <- arm_config[[arm_name]]
  df |>
    filter(treatment_arm == arm_name, !is.na(.data[[cfg$responder]])) |>
    select(all_of(continuous_covariates), responder = all_of(cfg$responder)) |>
    pivot_longer(all_of(continuous_covariates), names_to = "covariate", values_to = "value") |>
    ggplot(aes(x = responder, y = value)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 0.8) +
    facet_wrap(~covariate, scales = "free_y") +
    labs(
      title = paste0(arm_name, ": continuous covariates by responder status"),
      x = NULL, y = NULL
    )
}

# 2) Scatter plots: continuous covariates vs the arm's continuous outcome
plot_scatter <- function(arm_name) {
  cfg <- arm_config[[arm_name]]
  df |>
    filter(treatment_arm == arm_name, !is.na(.data[[cfg$outcome]])) |>
    select(all_of(continuous_covariates), outcome = all_of(cfg$outcome)) |>
    pivot_longer(all_of(continuous_covariates), names_to = "covariate", values_to = "value") |>
    ggplot(aes(x = value, y = outcome)) +
    geom_point(alpha = 0.4, size = 0.9) +
    facet_wrap(~covariate, scales = "free_x") +
    labs(
      title = paste0(arm_name, ": continuous covariates vs ", cfg$outcome),
      x = NULL, y = cfg$outcome
    )
}

# 3) Statistical tests
# Mann-Whitney U: continuous covariate vs binary responder status (non-parametric,
# since no normality assumption is checked/assumed here)
# Spearman correlation: continuous covariate vs continuous outcome (robust to
# non-linearity/outliers vs Pearson)
run_continuous_tests <- function(arm_name) {
  cfg <- arm_config[[arm_name]]
  arm_df <- df |> filter(treatment_arm == arm_name)

  mw_results <- map_dfr(continuous_covariates, function(covar) {
    sub <- arm_df |> filter(!is.na(.data[[cfg$responder]]))
    responders <- sub |> filter(.data[[cfg$responder]] == "Responder") |> pull(.data[[covar]])
    non_responders <- sub |> filter(.data[[cfg$responder]] == "Non-responder") |> pull(.data[[covar]])
    wt <- wilcox.test(responders, non_responders, exact = FALSE)
    tibble(
      covariate = covar,
      test = "Mann-Whitney U (responder vs non-responder)",
      statistic = unname(wt$statistic),
      p_value = wt$p.value,
      n_responder = length(responders),
      n_non_responder = length(non_responders)
    )
  })

  corr_results <- map_dfr(continuous_covariates, function(covar) {
    sub <- arm_df |> filter(!is.na(.data[[cfg$outcome]]), !is.na(.data[[covar]]))
    ct <- cor.test(sub[[covar]], sub[[cfg$outcome]], method = "spearman")
    tibble(
      covariate = covar,
      test = "Spearman correlation vs continuous outcome",
      statistic = unname(ct$estimate),
      p_value = ct$p.value,
      n = nrow(sub)
    )
  })

  bind_rows(mw_results, corr_results) |> mutate(treatment_arm = arm_name, .before = 1)
}

# Fisher's exact test: categorical covariate vs binary responder status
run_categorical_tests <- function(arm_name) {
  cfg <- arm_config[[arm_name]]
  arm_df <- df |> filter(treatment_arm == arm_name, !is.na(.data[[cfg$responder]]))

  map_dfr(categorical_covariates, function(covar) {
    tab <- table(arm_df[[covar]], arm_df[[cfg$responder]])
    ft <- fisher.test(tab, simulate.p.value = TRUE, B = 10000)
    tibble(
      covariate = covar,
      test = "Fisher's exact test (responder vs non-responder)",
      p_value = ft$p.value,
      n = sum(tab)
    )
  }) |>
    mutate(treatment_arm = arm_name, .before = 1)
}

arms <- names(arm_config)

continuous_tests <- map_dfr(arms, run_continuous_tests) |>
  group_by(test) |>
  mutate(p_adj_fdr = p.adjust(p_value, method = "fdr")) |>
  ungroup() |>
  arrange(treatment_arm, test, p_value)

categorical_tests <- map_dfr(arms, run_categorical_tests) |>
  group_by(test) |>
  mutate(p_adj_fdr = p.adjust(p_value, method = "fdr")) |>
  ungroup() |>
  arrange(treatment_arm, test, p_value)

# ---------------------------------------------------------------------------
# Generate and save all plots (6 total: boxplot + scatter per arm)
# ---------------------------------------------------------------------------

dir.create("plots", showWarnings = FALSE)

walk(arms, function(arm_name) {
  ggsave(
    file.path("plots", paste0(arm_name, "_boxplot_by_responder.png")),
    plot_boxplots(arm_name),
    width = 9, height = 7, dpi = 150
  )
  ggsave(
    file.path("plots", paste0(arm_name, "_scatter_vs_outcome.png")),
    plot_scatter(arm_name),
    width = 9, height = 7, dpi = 150
  )
})

# To view a plot interactively instead of just saving it, call e.g.:
# plot_boxplots("Mexiletine")
# plot_scatter("Tideglusib")

# ---------------------------------------------------------------------------
# Combine all 6 plots into a single multi-page PDF for easy sharing
# ---------------------------------------------------------------------------

pdf(file.path("plots", "dm1_eda_all_arms.pdf"), width = 9, height = 7)
walk(arms, function(arm_name) {
  print(plot_boxplots(arm_name))
  print(plot_scatter(arm_name))
})
dev.off()

# ---------------------------------------------------------------------------
# Pre-split diagnostics: variability by arm, numeric distribution checks,
# correlation structure, and responder class balance
# ---------------------------------------------------------------------------

# 1) Every covariate, summarized within each arm
continuous_summary_by_arm <- df |>
  select(treatment_arm, all_of(continuous_covariates)) |>
  pivot_longer(-treatment_arm, names_to = "covariate", values_to = "value") |>
  group_by(treatment_arm, covariate) |>
  summarise(
    n = n(),
    mean = mean(value),
    sd = sd(value),
    median = median(value),
    iqr = IQR(value),
    cv_pct = round(100 * sd(value) / mean(value), 1),
    .groups = "drop"
  ) |>
  arrange(covariate, treatment_arm)

categorical_summary_by_arm <- df |>
  select(treatment_arm, all_of(categorical_covariates)) |>
  pivot_longer(-treatment_arm, names_to = "covariate", values_to = "level") |>
  count(treatment_arm, covariate, level) |>
  group_by(treatment_arm, covariate) |>
  mutate(prop = round(n / sum(n), 3)) |>
  ungroup() |>
  arrange(covariate, treatment_arm, level)

# 2) Range, skew, and outliers for the 13 numeric covariates (whole sample;
# baseline covariates should be comparable across arms under randomization)
numeric_diagnostics <- df |>
  select(all_of(continuous_covariates)) |>
  pivot_longer(everything(), names_to = "covariate", values_to = "value") |>
  group_by(covariate) |>
  summarise(
    n = n(),
    min = min(value),
    max = max(value),
    range = max - min,
    mean = mean(value),
    sd = sd(value),
    median = median(value),
    q1 = quantile(value, 0.25),
    q3 = quantile(value, 0.75),
    iqr = q3 - q1,
    skewness = mean(((value - mean(value)) / sd(value))^3),
    lower_fence = q1 - 1.5 * iqr,
    upper_fence = q3 + 1.5 * iqr,
    n_outliers = sum(value < lower_fence | value > upper_fence),
    pct_outliers = round(100 * n_outliers / n, 1),
    .groups = "drop"
  ) |>
  arrange(desc(abs(skewness)))

# 3a) Correlation among the 13 numeric covariates (Spearman: robust to the
# skew/outliers flagged above). Reporting unique pairs above |rho| = 0.3.
numeric_corr_matrix <- df |>
  select(all_of(continuous_covariates)) |>
  cor(method = "spearman")

numeric_corr_pairs <- as.data.frame(as.table(numeric_corr_matrix)) |>
  rename(var1 = Var1, var2 = Var2, correlation = Freq) |>
  filter(var1 != var2) |>
  mutate(pair_key = map2_chr(as.character(var1), as.character(var2), ~ paste(sort(c(.x, .y)), collapse = "|"))) |>
  distinct(pair_key, .keep_all = TRUE) |>
  select(var1, var2, correlation) |>
  filter(abs(correlation) >= 0.3) |>
  arrange(desc(abs(correlation)))

# 3b) Responder/Non-responder class balance within each arm
responder_balance <- map_dfr(names(arm_config), function(arm) {
  cfg <- arm_config[[arm]]
  df |>
    filter(treatment_arm == arm) |>
    mutate(responder_status = .data[[cfg$responder]]) |>
    count(responder_status) |>
    mutate(arm = arm, prop = round(n / sum(n), 3), .before = 1)
})

# ---------------------------------------------------------------------------
# Train / test split, done separately per arm (outcomes are arm-specific),
# stratified on the categorical responder label, with a fixed seed. Once
# created, each arm's test set is written to disk and locked: re-running
# this script will NOT regenerate or overwrite it, so it stays untouched
# until final evaluation.
# ---------------------------------------------------------------------------

library(rsample)

SPLIT_SEED <- 5182
TEST_PROP <- 0.2
lock_dir <- "locked_test_sets"
dir.create(lock_dir, showWarnings = FALSE)

split_one_arm <- function(arm_name) {
  cfg <- arm_config[[arm_name]]

  arm_data <- df |>
    filter(treatment_arm == arm_name) |>
    mutate(responder_status = factor(
      .data[[cfg$responder]], levels = c("Non-responder", "Responder")
    )) |>
    select(
      patient_id, treatment_arm, all_of(continuous_covariates),
      all_of(categorical_covariates), responder_status
    )

  stopifnot(sum(is.na(arm_data$responder_status)) == 0)

  set.seed(SPLIT_SEED)
  arm_split <- initial_split(arm_data, prop = 1 - TEST_PROP, strata = responder_status)
  train_new <- training(arm_split)
  test_new <- testing(arm_split)

  test_lock_path <- file.path(lock_dir, paste0(arm_name, "_test.rds"))
  if (file.exists(test_lock_path)) {
    # Test set already locked from a previous run: reuse it as-is, and
    # rebuild the matching training set as "everything else" so train/test
    # stay complementary even if this script is re-run later.
    test_final <- readRDS(test_lock_path)
    train_final <- arm_data |> filter(!patient_id %in% test_final$patient_id)
  } else {
    saveRDS(test_new, test_lock_path)
    test_final <- test_new
    train_final <- train_new
  }

  list(train = train_final, test = test_final)
}

splits_by_arm <- map(arms, split_one_arm) |> set_names(arms)

train_mexiletine <- splits_by_arm$Mexiletine$train
test_mexiletine <- splits_by_arm$Mexiletine$test
train_oligonucleotide <- splits_by_arm$Oligonucleotide$train
test_oligonucleotide <- splits_by_arm$Oligonucleotide$test
train_tideglusib <- splits_by_arm$Tideglusib$train
test_tideglusib <- splits_by_arm$Tideglusib$test

# Check: n and responder-class proportions preserved train vs. test, per arm
map_dfr(arms, function(arm_name) {
  bind_rows(
    splits_by_arm[[arm_name]]$train |> mutate(set = "train"),
    splits_by_arm[[arm_name]]$test |> mutate(set = "test")
  ) |>
    mutate(treatment_arm = arm_name) |>
    count(treatment_arm, set, responder_status) |>
    group_by(treatment_arm, set) |>
    mutate(prop = round(n / sum(n), 3)) |>
    ungroup()
})

# ---------------------------------------------------------------------------
# Logistic regression per arm (training data only), Non-responder = 0,
# Responder = 1. One model per arm, since predictors of response can
# plausibly differ by mechanism of action.
# ---------------------------------------------------------------------------

library(broom)

# Zero-variance numeric covariates (e.g. albumin in Mexiletine) would make
# the model rank-deficient, so drop them per arm before fitting.
fit_arm_logit <- function(train_df) {
  train_df <- train_df |>
    mutate(response_01 = if_else(responder_status == "Responder", 1L, 0L))

  usable_numeric <- continuous_covariates[
    map_lgl(continuous_covariates, ~ sd(train_df[[.x]], na.rm = TRUE) > 0)
  ]
  dropped <- setdiff(continuous_covariates, usable_numeric)
  if (length(dropped) > 0) {
    message("Dropping zero-variance predictor(s): ", paste(dropped, collapse = ", "))
  }

  predictors <- c(usable_numeric, categorical_covariates)
  form <- as.formula(paste("response_01 ~", paste(predictors, collapse = " + ")))
  glm(form, data = train_df, family = binomial())
}

logit_models <- list(
  Mexiletine = fit_arm_logit(train_mexiletine),
  Oligonucleotide = fit_arm_logit(train_oligonucleotide),
  Tideglusib = fit_arm_logit(train_tideglusib)
)

# Tidy coefficients as odds ratios with 95% CI, for all three models
logit_results <- map_dfr(names(logit_models), function(arm_name) {
  tidy(logit_models[[arm_name]], exponentiate = TRUE, conf.int = TRUE) |>
    mutate(treatment_arm = arm_name, .before = 1)
})

# Model fit diagnostics: n events per predictor, pseudo-R2, convergence
logit_diagnostics <- map_dfr(names(logit_models), function(arm_name) {
  mod <- logit_models[[arm_name]]
  n_events <- sum(mod$y == 1)
  n_predictors <- length(coef(mod)) - 1
  null_dev <- mod$null.deviance
  resid_dev <- mod$deviance
  tibble(
    treatment_arm = arm_name,
    n_obs = nrow(mod$data),
    n_events = n_events,
    n_predictors = n_predictors,
    events_per_predictor = round(n_events / n_predictors, 1),
    mcfadden_r2 = round(1 - resid_dev / null_dev, 3),
    converged = mod$converged
  )
})

# ---------------------------------------------------------------------------
# Elastic net logistic regression per arm (training data only), tuned via
# cross-validation. Elastic net is chosen over plain logistic regression to
# handle the multicollinearity among baseline covariates and the low
# events-per-predictor ratio flagged above (it shrinks/selects coefficients
# instead of letting them blow up, as happened with phenotype in Mexiletine).
# ---------------------------------------------------------------------------

library(glmnet)

ENET_CV_SEED <- 2694
ALPHA_GRID <- seq(0, 1, by = 0.1) # 0 = ridge, 1 = lasso

# Stratified fold assignment (by outcome class) so each CV fold keeps a
# similar responder/non-responder ratio, important given the class imbalance
# already noted (e.g. ~29% responders in Mexiletine).
make_stratified_foldid <- function(y, k = 5, seed) {
  set.seed(seed)
  foldid <- integer(length(y))
  for (cls in unique(y)) {
    idx <- which(y == cls)
    foldid[idx] <- sample(rep(seq_len(k), length.out = length(idx)))
  }
  foldid
}

fit_arm_enet <- function(train_df, arm_name) {
  train_df <- train_df |>
    mutate(response_01 = if_else(responder_status == "Responder", 1L, 0L))

  usable_numeric <- continuous_covariates[
    map_lgl(continuous_covariates, ~ sd(train_df[[.x]], na.rm = TRUE) > 0)
  ]
  predictors <- c(usable_numeric, categorical_covariates)

  x_train <- model.matrix(
    as.formula(paste("~", paste(predictors, collapse = " + "))),
    data = train_df
  )[, -1]
  y_train <- train_df$response_01

  foldid <- make_stratified_foldid(y_train, k = 5, seed = ENET_CV_SEED)

  # Scan alpha (0 = ridge ... 1 = lasso); for each, let cv.glmnet tune lambda
  alpha_scan <- map_dfr(ALPHA_GRID, function(a) {
    set.seed(ENET_CV_SEED)
    cvfit <- cv.glmnet(
      x_train, y_train,
      family = "binomial", alpha = a, foldid = foldid, type.measure = "deviance"
    )
    tibble(alpha = a, lambda_min = cvfit$lambda.min, cv_deviance = min(cvfit$cvm))
  })

  best_alpha <- alpha_scan$alpha[which.min(alpha_scan$cv_deviance)]

  set.seed(ENET_CV_SEED)
  final_cv <- cv.glmnet(
    x_train, y_train,
    family = "binomial", alpha = best_alpha, foldid = foldid, type.measure = "deviance"
  )

  list(
    arm = arm_name,
    alpha_scan = alpha_scan |> mutate(treatment_arm = arm_name, .before = 1),
    best_alpha = best_alpha,
    cv_fit = final_cv,
    x_train = x_train,
    y_train = y_train,
    predictors = predictors
  )
}

enet_fits <- map(arms, function(a) fit_arm_enet(splits_by_arm[[a]]$train, a)) |> set_names(arms)

# Summary: chosen alpha/lambda, CV deviance, and how many coefficients
# elastic net kept non-zero vs. the ~20 predictors in the plain logistic fit
enet_summary <- map_dfr(arms, function(arm_name) {
  fit <- enet_fits[[arm_name]]
  coefs <- as.matrix(coef(fit$cv_fit, s = "lambda.min"))
  tibble(
    treatment_arm = arm_name,
    best_alpha = fit$best_alpha,
    lambda_min = fit$cv_fit$lambda.min,
    cv_deviance_min = min(fit$cv_fit$cvm),
    n_predictors_total = length(fit$predictors),
    n_nonzero_coefs = sum(coefs[-1, 1] != 0) # exclude intercept
  )
})

# Non-zero coefficients retained per arm, as odds ratios
enet_coefs <- map_dfr(arms, function(arm_name) {
  fit <- enet_fits[[arm_name]]
  coefs <- as.matrix(coef(fit$cv_fit, s = "lambda.min"))
  tibble(
    treatment_arm = arm_name,
    term = rownames(coefs),
    log_odds = coefs[, 1],
    odds_ratio = exp(coefs[, 1])
  ) |>
    filter(term != "(Intercept)", log_odds != 0) |>
    arrange(desc(abs(log_odds)))
})

# ---------------------------------------------------------------------------
# Elastic net logistic regression per arm (glmnet), with cv.glmnet choosing
# the optimal lambda on the training set only. Addresses the multicollinearity
# and low events-per-predictor issues flagged for the plain logistic models.
# ---------------------------------------------------------------------------

library(glmnet)

ENET_ALPHA <- 0.5 # 0 = ridge, 1 = lasso; 0.5 balances both given correlated predictors
ENET_SEED <- 8821
N_FOLDS <- 10

fit_arm_enet <- function(train_df) {
  train_df <- train_df |>
    mutate(response_01 = if_else(responder_status == "Responder", 1L, 0L))

  usable_numeric <- continuous_covariates[
    map_lgl(continuous_covariates, ~ sd(train_df[[.x]], na.rm = TRUE) > 0)
  ]
  predictors <- c(usable_numeric, categorical_covariates)
  form <- as.formula(paste("~", paste(predictors, collapse = " + "), "- 1"))

  x <- model.matrix(form, data = train_df)
  y <- train_df$response_01

  set.seed(ENET_SEED)
  cvfit <- cv.glmnet(
    x, y,
    family = "binomial", alpha = ENET_ALPHA,
    nfolds = N_FOLDS, type.measure = "deviance", standardize = TRUE
  )

  list(cvfit = cvfit, x = x, y = y)
}

enet_fits <- map(
  list(Mexiletine = train_mexiletine, Oligonucleotide = train_oligonucleotide, Tideglusib = train_tideglusib),
  fit_arm_enet
)

# Lambda choice + model complexity summary, at both the CV-minimum and the
# more parsimonious 1-SE lambda (safer given the small per-arm sample sizes)
enet_lambda_summary <- map_dfr(names(enet_fits), function(arm_name) {
  cvfit <- enet_fits[[arm_name]]$cvfit
  tibble(
    treatment_arm = arm_name,
    lambda_min = cvfit$lambda.min,
    lambda_1se = cvfit$lambda.1se,
    n_nonzero_min = cvfit$nzero[cvfit$lambda == cvfit$lambda.min],
    n_nonzero_1se = cvfit$nzero[cvfit$lambda == cvfit$lambda.1se]
  )
})

# Coefficients (odds ratios) at lambda.1se, the default reported model
extract_enet_coefs <- function(cvfit, s = "lambda.1se") {
  cm <- as.matrix(coef(cvfit, s = s))
  tibble(term = rownames(cm), log_odds = cm[, 1]) |>
    filter(term != "(Intercept)", log_odds != 0) |>
    mutate(odds_ratio = exp(log_odds)) |>
    arrange(desc(abs(log_odds)))
}

enet_results <- map_dfr(names(enet_fits), function(arm_name) {
  extract_enet_coefs(enet_fits[[arm_name]]$cvfit) |>
    mutate(treatment_arm = arm_name, .before = 1)
})

# ---------------------------------------------------------------------------
# Linear regression per arm (training data only), on the arm's continuous
# outcome: univariate (one predictor at a time) and multivariate (all
# predictors together). The locked test sets are not touched here.
# ---------------------------------------------------------------------------

library(broom)

# Train-set patient membership was fixed when the splits were created and
# locked; we just re-attach each arm's continuous outcome column (dropped
# from train_mexiletine/etc. earlier because that split kept only the
# binary responder outcome) without changing who is in train vs. test.
build_train_with_continuous_outcome <- function(arm_name) {
  train_ids <- splits_by_arm[[arm_name]]$train$patient_id
  cfg <- arm_config[[arm_name]]
  df |>
    filter(patient_id %in% train_ids) |>
    select(
      patient_id, all_of(continuous_covariates), all_of(categorical_covariates),
      outcome = all_of(cfg$outcome)
    )
}

train_lm_data <- map(arms, build_train_with_continuous_outcome) |> set_names(arms)

# disease_duration is dropped: it is an exact linear combination
# (disease_duration = age - age_at_onset), so it is redundant once age and
# age_at_onset are both in the model and would otherwise break VIF/aliasing.
lm_predictors_for_arm <- function(train_df) {
  usable_numeric <- continuous_covariates[
    map_lgl(continuous_covariates, ~ sd(train_df[[.x]], na.rm = TRUE) > 0)
  ]
  usable_numeric <- setdiff(usable_numeric, "disease_duration")
  c(usable_numeric, categorical_covariates)
}

# --- Univariate linear regression: one predictor at a time ---
run_univariate_lm <- function(train_df, arm_name) {
  predictors <- lm_predictors_for_arm(train_df)

  map_dfr(predictors, function(p) {
    mod <- lm(as.formula(paste("outcome ~", p)), data = train_df)
    g <- glance(mod)
    tidy(mod, conf.int = TRUE) |>
      filter(term != "(Intercept)") |>
      mutate(
        treatment_arm = arm_name,
        predictor = p,
        r_squared = g$r.squared,
        .before = 1
      )
  })
}

univariate_lm_results <- map_dfr(arms, function(a) run_univariate_lm(train_lm_data[[a]], a))

# --- Multivariate linear regression: all predictors together ---
fit_multivariate_lm <- function(train_df) {
  predictors <- lm_predictors_for_arm(train_df)
  lm(as.formula(paste("outcome ~", paste(predictors, collapse = " + "))), data = train_df)
}

multivariate_lm_models <- map(arms, function(a) fit_multivariate_lm(train_lm_data[[a]])) |> set_names(arms)

multivariate_lm_results <- map_dfr(names(multivariate_lm_models), function(a) {
  tidy(multivariate_lm_models[[a]], conf.int = TRUE) |> mutate(treatment_arm = a, .before = 1)
})

multivariate_lm_glance <- map_dfr(names(multivariate_lm_models), function(a) {
  glance(multivariate_lm_models[[a]]) |> mutate(treatment_arm = a, .before = 1)
})

# Diagnostics: multicollinearity (VIF) and residual normality (Shapiro-Wilk),
# since OLS inference depends on these assumptions
library(car)

multivariate_lm_diagnostics <- map_dfr(names(multivariate_lm_models), function(a) {
  mod <- multivariate_lm_models[[a]]
  vif_vals <- vif(mod)
  # vif() returns a vector for numeric terms and a matrix row for factors
  # with >1 df; take the generalized VIF column in that case.
  if (is.matrix(vif_vals)) vif_vals <- vif_vals[, 1]
  shapiro_p <- shapiro.test(residuals(mod))$p.value
  tibble(
    treatment_arm = a,
    max_vif = round(max(vif_vals), 1),
    term_max_vif = names(vif_vals)[which.max(vif_vals)],
    n_vif_above_5 = sum(vif_vals > 5),
    shapiro_p_value = round(shapiro_p, 3)
  )
})
