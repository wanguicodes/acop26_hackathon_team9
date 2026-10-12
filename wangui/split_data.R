################################################################################
# split_data.R  --  Train/test split for the ACoP 2026 DM1 Hackathon (Team 9)
#
# SCOPE
#   Split the full dataset into an 80/20 training/testing split, performed
#   SEPARATELY within each treatment arm (Mexiletine, Oligonucleotide,
#   Tideglusib) so each arm is proportionally represented in both splits.
#
#   Within each arm, the split is stratified on that arm's CONTINUOUS efficacy
#   outcome, binned into 5 quantile-based groups (~20% of the arm each). This
#   ensures training and testing have matching proportions of individuals in
#   each outcome bin -- i.e. a representative spread of outcome severity in
#   both sets, not just a representative mean. The binary responder label is
#   intentionally ignored for this split.
#
#   Each arm's continuous outcome lives in a different column:
#     Mexiletine      -> myotonia_change_12wk
#     Oligonucleotide  -> qmt_change_26wk
#     Tideglusib       -> tideglusib_composite
#
#   The per-arm training/testing subsets are then stacked into two shared
#   files spanning all three arms.
#
# OUTLIER HANDLING (before splitting)
#   Every continuous variable (13 baseline covariates + each arm's continuous
#   efficacy outcome) is screened for extreme values with the IQR method:
#   values below Q1 - 3xIQR or above Q3 + 3xIQR are flagged. Flagged patients
#   are excluded from BOTH the training and test set before the split is
#   performed. All flagged values are written to a single summary table along
#   with their fold-change relative to the IQR (i.e. how many IQRs beyond the
#   nearest quartile the value falls -- values at the 3x threshold itself have
#   a fold-change of exactly 3). Baseline covariates are screened across all
#   patients; each arm's outcome is screened only within that arm (outcome
#   columns are NA outside their arm).
#
# OUTPUTS
#   data/derived/split/training_shared.csv
#   data/derived/split/testing_shared.csv
#   data/derived/split/outlier_summary.csv
################################################################################

### Packages -------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidymodels)  # provides rsample::initial_split() etc.
  library(dplyr)
  library(purrr)
  library(readr)
})

### Paths ----------------------------------------------------------------------
root <- tryCatch(here::here(), error = function(e) getwd())
data_path <- file.path(root, "data", "source", "dm1_clinical_trial_dataset.csv")
if (!file.exists(data_path)) {
  hits <- list.files(root, pattern = "dm1_clinical_trial_dataset\\.csv$",
                      recursive = TRUE, full.names = TRUE)
  if (length(hits) == 0)
    stop("Could not locate dm1_clinical_trial_dataset.csv under: ", root)
  data_path <- hits[[1]]
}
out_dir <- file.path(root, "data", "derived", "split")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
message("Data: ", data_path)
message("Out:  ", out_dir)

### Read data -------------------------------------------------------------------
dat <- read_csv(data_path, show_col_types = FALSE)

arms <- c("Mexiletine", "Oligonucleotide", "Tideglusib")

# Continuous efficacy outcome column for each arm (populated only within arm)
outcome_of <- c(Mexiletine      = "myotonia_change_12wk",
                Oligonucleotide = "qmt_change_26wk",
                Tideglusib      = "tideglusib_composite")

# Continuous baseline covariates (screened across all patients, regardless of arm)
cont_cov <- c("age", "age_at_onset", "disease_duration", "ctg_repeats",
              "somatic_mosaicism_index", "myotonia_severity", "muscle_strength",
              "cognitive_function", "ck_level", "pr_interval",
              "insulin_resistance", "albumin", "bmi")

n_bins <- 5

### Outlier screening (IQR method), before splitting ---------------------------
# Flag values below Q1 - 3*IQR or above Q3 + 3*IQR for each continuous
# variable. Outcome columns are evaluated on their own (non-NA, i.e.
# within-arm) values; covariates are evaluated across all patients.
flag_outliers <- function(var) {
  x <- dat[[var]]
  q <- quantile(x, c(0.25, 0.75), na.rm = TRUE)
  iqr <- unname(q[2] - q[1])
  lower <- unname(q[1] - 3 * iqr)
  upper <- unname(q[2] + 3 * iqr)
  idx <- which(!is.na(x) & (x < lower | x > upper))
  if (length(idx) == 0) return(NULL)
  tibble(
    patient_id    = dat$patient_id[idx],
    treatment_arm = dat$treatment_arm[idx],
    variable      = var,
    value         = x[idx],
    q1            = unname(q[1]),
    q3            = unname(q[2]),
    iqr           = iqr,
    lower_bound   = lower,
    upper_bound   = upper,
    direction     = ifelse(x[idx] < lower, "below Q1-3xIQR", "above Q3+3xIQR"),
    # How many IQRs beyond the nearest quartile the value falls; the
    # Q1-3xIQR/Q3+3xIQR threshold itself corresponds to a fold-change of 3.
    fold_change_iqr = ifelse(x[idx] < lower,
                              (q[1] - x[idx]) / iqr,
                              (x[idx] - q[2]) / iqr)
  )
}

outlier_summary <- map_dfr(c(cont_cov, unname(outcome_of)), flag_outliers) %>%
  arrange(desc(fold_change_iqr))

write_csv(outlier_summary, file.path(out_dir, "outlier_summary.csv"))

flagged_ids <- unique(outlier_summary$patient_id)
message("\nOutlier screening (IQR method, 3x threshold): ",
        nrow(outlier_summary), " flagged value(s) across ",
        length(flagged_ids), " patient(s).")
if (length(flagged_ids) > 0) print(outlier_summary, n = Inf)

dat <- dat %>% filter(!patient_id %in% flagged_ids)
message("Patients remaining after outlier exclusion: ", nrow(dat))

### Split within each arm, stratified on binned continuous outcome -------------
set.seed(8742)  # reproducible 80/20 split

split_one_arm <- function(arm) {
  sub <- dat %>% filter(treatment_arm == arm)
  outcome_col <- outcome_of[[arm]]

  # Quantile-based bins (~equal-sized groups) on the arm's continuous outcome.
  sub <- sub %>%
    mutate(outcome_bin = ntile(.data[[outcome_col]], n_bins) %>% factor())

  split <- initial_split(sub, prop = 0.8, strata = outcome_bin)
  list(train = training(split) %>% select(-outcome_bin),
       test  = testing(split)  %>% select(-outcome_bin))
}

splits <- lapply(arms, split_one_arm)
names(splits) <- arms

training_shared <- bind_rows(lapply(splits, `[[`, "train"))
testing_shared  <- bind_rows(lapply(splits, `[[`, "test"))

### Sanity check: outcome-bin proportions per arm, train vs test ---------------
combined <- bind_rows(
  training_shared %>% mutate(set = "train"),
  testing_shared  %>% mutate(set = "test")
)
check <- purrr::map_dfr(arms, function(arm) {
  outcome_col <- outcome_of[[arm]]
  combined %>%
    filter(treatment_arm == arm) %>%
    mutate(outcome_val = .data[[outcome_col]],
           outcome_bin = ntile(outcome_val, n_bins)) %>%
    count(treatment_arm, set, outcome_bin) %>%
    group_by(set) %>%
    mutate(prop = n / sum(n)) %>%
    ungroup()
})
print(check, n = Inf)

### Export -----------------------------------------------------------------------
write_csv(training_shared, file.path(out_dir, "training_shared.csv"))
write_csv(testing_shared, file.path(out_dir, "testing_shared.csv"))
message("\nSaved: ", file.path(out_dir, "training_shared.csv"),
        " (n = ", nrow(training_shared), ")")
message("Saved: ", file.path(out_dir, "testing_shared.csv"),
        " (n = ", nrow(testing_shared), ")")
