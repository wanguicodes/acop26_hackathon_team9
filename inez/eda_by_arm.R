## EDA: baseline covariate distributions and correlation structure by treatment arm
## Author: Inez
##
## Purpose: before any modeling, check (1) whether baseline covariates differ
## systematically across the three (non-randomly assigned) treatment arms, and
## (2) whether candidate covariates are collinear within each arm -- collinear
## covariates will "share credit" under permutation importance, which the
## hackathon's suggested methodology (see kickoff deck, p.72) relies on to
## separate real drivers from noise.
##
## Inputs:  data/source/dm1_clinical_trial_dataset.csv
## Outputs: inez/figures/01_boxplots_by_arm.png
##          inez/figures/02_correlation_mexiletine.png
##          inez/figures/03_correlation_oligonucleotide.png
##          inez/figures/04_correlation_tideglusib.png

library(tidyverse)

df <- read_csv("data/source/dm1_clinical_trial_dataset.csv", show_col_types = FALSE)

mex   <- df |> filter(treatment_arm == "Mexiletine")
oligo <- df |> filter(treatment_arm == "Oligonucleotide")
tidge <- df |> filter(treatment_arm == "Tideglusib")

## ---- 1. Boxplots of key covariates by arm --------------------------------

key_covs <- c("ctg_repeats", "muscle_strength", "myotonia_severity", "somatic_mosaicism_index",
              "cognitive_function", "pr_interval", "ck_level", "age", "disease_duration")

df_long <- df |>
  select(treatment_arm, all_of(key_covs)) |>
  pivot_longer(-treatment_arm, names_to = "covariate", values_to = "value")

p_box <- ggplot(df_long, aes(x = treatment_arm, y = value)) +
  geom_boxplot() +
  facet_wrap(~ covariate, scales = "free_y") +
  labs(x = NULL, y = NULL, title = "Baseline covariate distributions by treatment arm")

ggsave("inez/figures/01_boxplots_by_arm.png", p_box, width = 9, height = 7, dpi = 150)

## ---- 2. Within-arm correlation heatmaps -----------------------------------
## Collinear covariates (e.g., ctg_repeats / myotonia_severity / muscle_strength /
## somatic_mosaicism_index) are expected to split credit under permutation
## importance; flagging that cluster here informs how results get interpreted.

numeric_all <- c("age", "age_at_onset", "albumin", "bmi", "ck_level", "cognitive_function",
                  "ctg_repeats", "disease_duration", "insulin_resistance", "muscle_strength",
                  "myotonia_severity", "pr_interval", "somatic_mosaicism_index")

plot_corr <- function(data, arm_name) {
  m <- data |> select(all_of(numeric_all)) |> cor(use = "pairwise.complete.obs")
  m_long <- as.data.frame(m) |>
    rownames_to_column("var1") |>
    pivot_longer(-var1, names_to = "var2", values_to = "r")
  ggplot(m_long, aes(var1, var2, fill = r)) +
    geom_tile() +
    geom_text(aes(label = round(r, 2)), size = 2.3) +
    scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick", limits = c(-1, 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste("Covariate correlations —", arm_name), x = NULL, y = NULL)
}

p_corr_mex   <- plot_corr(mex, "Mexiletine")
p_corr_oligo <- plot_corr(oligo, "Oligonucleotide")
p_corr_tidge <- plot_corr(tidge, "Tideglusib")

ggsave("inez/figures/02_correlation_mexiletine.png", p_corr_mex, width = 7, height = 6, dpi = 150)
ggsave("inez/figures/03_correlation_oligonucleotide.png", p_corr_oligo, width = 7, height = 6, dpi = 150)
ggsave("inez/figures/04_correlation_tideglusib.png", p_corr_tidge, width = 7, height = 6, dpi = 150)

## ---- 3. Notable data quirk --------------------------------------------
## albumin is constant across all Mexiletine patients (sd = 0) but varies
## normally in the other two arms -- it carries zero predictive variance
## specifically within the Mexiletine arm.
df |>
  group_by(treatment_arm) |>
  summarise(sd_albumin = sd(albumin), n_distinct_albumin = n_distinct(albumin), .groups = "drop") |>
  print()
