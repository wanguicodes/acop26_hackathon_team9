################################################################################
# split_data.R  --  Train/test split for the ACoP 2026 DM1 Hackathon (Team 9)
#
# SCOPE
#   Split the full dataset into an 80/20 training/testing split, performed
#   SEPARATELY within each treatment arm (Mexiletine, Oligonucleotide,
#   Tideglusib) so each arm is proportionally represented in both splits.
#   Within each arm, the split is stratified on that arm's binary responder
#   label so training and testing contain the same proportion of responders
#   (rsample::initial_split rounds strata sizes identically, so the
#   responder:non-responder ratio -- and therefore the number of responders
#   relative to split size -- matches across arms' train/test sets).
#
#   Each arm's responder label lives in its own column (NA outside that arm):
#     Mexiletine      -> responder_phenotype_mex
#     Oligonucleotide  -> responder_phenotype_oligo
#     Tideglusib       -> responder_phenotype_tidge
#   These are coalesced into one shared `responder` column before splitting
#   and stratifying.
#
#   The per-arm training/testing subsets are then stacked into two shared
#   files spanning all three arms.
#
# OUTPUTS
#   data/derived/split/training_shared.csv
#   data/derived/split/testing_shared.csv
################################################################################

### Packages -------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidymodels)  # provides rsample::initial_split() etc.
  library(dplyr)
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

# Each arm's responder label is stored in a separate column (NA outside that
# arm). Coalesce into one shared `responder` column for stratified splitting.
dat <- dat %>%
  mutate(responder = coalesce(responder_phenotype_mex,
                               responder_phenotype_oligo,
                               responder_phenotype_tidge),
         responder = factor(responder))

### Split within each arm, stratified on responder status ----------------------
set.seed(1234)  # reproducible 80/20 split

split_one_arm <- function(arm) {
  sub <- dat %>% filter(treatment_arm == arm)
  split <- initial_split(sub, prop = 0.8, strata = responder)
  list(train = training(split), test = testing(split))
}

splits <- lapply(arms, split_one_arm)
names(splits) <- arms

training_shared <- bind_rows(lapply(splits, `[[`, "train"))
testing_shared  <- bind_rows(lapply(splits, `[[`, "test"))

### Sanity check: responder counts/proportions per arm, train vs test ----------
check <- bind_rows(
  training_shared %>% mutate(set = "train"),
  testing_shared  %>% mutate(set = "test")
) %>%
  count(treatment_arm, set, responder) %>%
  group_by(treatment_arm, set) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()
print(check)

### Export -----------------------------------------------------------------------
write_csv(training_shared, file.path(out_dir, "training_shared.csv"))
write_csv(testing_shared, file.path(out_dir, "testing_shared.csv"))
message("\nSaved: ", file.path(out_dir, "training_shared.csv"),
        " (n = ", nrow(training_shared), ")")
message("Saved: ", file.path(out_dir, "testing_shared.csv"),
        " (n = ", nrow(testing_shared), ")")
