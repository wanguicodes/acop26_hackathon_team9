################################################################################
# eda.R  --  Exploratory Data Analysis for the ACoP 2026 DM1 Hackathon (Team 9)
#
# SCOPE
#   For EACH treatment arm, surface baseline covariates that may predict that
#   arm's CONTINUOUS efficacy outcome. (Some covariates are deliberately noise;
#   separating real signal from noise is the hackathon task.)
#
#   Arm              Continuous outcome        Direction of benefit
#   ---------------  ------------------------  -----------------------------
#   Mexiletine       myotonia_change_12wk      negative = improvement
#   Oligonucleotide  qmt_change_26wk           positive = improvement
#   Tideglusib       tideglusib_composite      positive = improvement (0-20)
#
#   Each outcome column is populated only for patients in its own arm.
#
# APPROACH
#   Custom EDA following MERGE / AstraZeneca conventions (no yspec file present):
#   theme_bw(), RColorBrewer "Dark2" palette, PNG figures, tidy summary tables
#   as CSV + PNG. Multipanel figures and stratified summary tables are
#   preferred to keep the number of output files small.
#
# NOTE ON THE SPLIT
#   This script explores the FULL dataset to understand structure and screen for
#   signal. Before modeling, re-run per-arm summaries on the TRAINING split only
#   (hold out the test set first) to avoid leakage.
#
# TRAIN / TEST COMPARISON
#   A handful of figures are duplicated with points colored/filled by whether
#   the row falls in the training or test split produced by
#   wangui/split_data.R (data/derived/split/{training,testing}_shared.csv).
#   This is a sanity check that the 80/20 split (stratified by responder,
#   within arm) did not introduce obvious covariate or outcome imbalance.
#   Density/proportion scales are used instead of raw counts because the
#   training set (n=398) is ~4x the size of the test set (n=102).
#
# OUTPUTS
#   wangui/eda_figures_fulldataset/  17 multipanel figures (PNG)
#   wangui/eda_tables_fulldataset/   2 summary tables (CSV + PNG)
################################################################################

### Packages -------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(scales)
  library(grid)
  library(gridExtra)
})

### Paths ----------------------------------------------------------------------
# Anchor at the hackathon repo root. here() finds the nearest project marker
# (.git / .Rproj); fall back to searching upward for the dataset if needed.
root <- tryCatch(here::here(), error = function(e) getwd())
data_path <- file.path(root, "data", "source", "dm1_clinical_trial_dataset.csv")
if (!file.exists(data_path)) {
  hits <- list.files(root, pattern = "dm1_clinical_trial_dataset\\.csv$",
                     recursive = TRUE, full.names = TRUE)
  if (length(hits) == 0)
    stop("Could not locate dm1_clinical_trial_dataset.csv under: ", root)
  data_path <- hits[[1]]
  root <- sub("/data/source/.*$", "", data_path)
}
fig_dir <- file.path(root, "wangui", "eda_figures_fulldataset")
tab_dir <- file.path(root, "wangui", "eda_tables_fulldataset")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)
message("Data:    ", data_path)
message("Figures: ", fig_dir)
message("Tables:  ", tab_dir)

### Theme / palette (MERGE conventions) ----------------------------------------
theme_set(theme_bw(base_size = 11) + theme(legend.position = "bottom"))
pal <- RColorBrewer::brewer.pal(n = 8, name = "Dark2")
options(ggplot2.discrete.colour = pal, ggplot2.discrete.fill = pal)

# Save a figure as PNG only.
save_fig <- function(plot, stem, width = 11, height = 8) {
  ggsave(file.path(fig_dir, paste0(stem, ".png")),
         plot = plot, width = width, height = height, dpi = 150)
  message("  saved figure: ", stem, " (png)")
}

save_table_png <- function(df, stem, title, subtitle = NULL,
                           width = 16, height = NULL, dpi = 220) {
  if (is.null(height)) {
    height <- max(8, 0.45 * nrow(df) + ifelse(is.null(subtitle), 1.2, 1.6))
  }
  grobs <- list(
    textGrob(title, gp = gpar(fontsize = 16, fontface = "bold"))
  )
  heights <- c(0.6)
  if (!is.null(subtitle)) {
    grobs <- c(grobs, list(textGrob(subtitle, gp = gpar(fontsize = 10))))
    heights <- c(heights, 0.35)
  }
  tbl_theme <- ttheme_minimal(core = list(fg_params = list(fontsize = 9)),
                              colhead = list(fg_params = list(fontsize = 10, fontface = "bold")))
  grobs <- c(grobs, list(tableGrob(df, rows = NULL, theme = tbl_theme)))
  heights <- c(heights, 1)
  png(file.path(tab_dir, paste0(stem, ".png")), width = width, height = height, units = "in", res = dpi)
  grid.arrange(grobs = grobs, ncol = 1, heights = heights)
  dev.off()
  message("  saved table: ", stem, " (csv + png)")
}

### Read data ------------------------------------------------------------------
dat <- fread(data_path, na.strings = c("", "NA", "."), data.table = FALSE)

### Train / test split tags -----------------------------------------------------
# Load the arm-stratified 80/20 split (see wangui/split_data.R) so a few EDA
# figures can compare training vs. test distributions. This only tags rows
# with their split membership; it does not touch outcome values.
split_dir <- file.path(root, "data", "derived", "split")
test_ids_path <- file.path(split_dir, "testing_shared.csv")
if (!file.exists(test_ids_path)) {
  stop("Could not find ", test_ids_path, ". Run wangui/split_data.R first.")
}
test_ids <- fread(test_ids_path, select = "patient_id", data.table = FALSE)$patient_id
dat$split <- factor(ifelse(dat$patient_id %in% test_ids, "Test", "Train"),
                     levels = c("Train", "Test"))

### Variable roles & human-readable labels -------------------------------------
cont_cov <- c("age", "age_at_onset", "disease_duration", "ctg_repeats",
              "somatic_mosaicism_index", "myotonia_severity", "muscle_strength",
              "cognitive_function", "ck_level", "pr_interval",
              "insulin_resistance", "albumin", "bmi")
cat_cov  <- c("sex", "phenotype", "family_history", "cataracts", "smoking_status")

arms <- c("Mexiletine", "Oligonucleotide", "Tideglusib")

# Continuous efficacy outcome for each arm
outcome_of <- c(Mexiletine      = "myotonia_change_12wk",
                Oligonucleotide = "qmt_change_26wk",
                Tideglusib      = "tideglusib_composite")

lab <- c(
  age                     = "Age (years)",
  age_at_onset            = "Age at onset (years)",
  disease_duration        = "Disease duration (years)",
  ctg_repeats             = "CTG repeats",
  somatic_mosaicism_index = "Somatic mosaicism index (0-1)",
  myotonia_severity       = "Myotonia severity (0-10)",
  muscle_strength         = "Muscle strength (20-100)",
  cognitive_function      = "Cognitive function (z-score)",
  ck_level                = "CK (U/L)",
  pr_interval             = "PR interval (ms)",
  insulin_resistance      = "Insulin resistance (HOMA-IR)",
  albumin                 = "Albumin (g/dL)",
  bmi                     = "BMI (kg/m2)",
  sex                     = "Sex",
  phenotype               = "Phenotype",
  family_history          = "Family history",
  cataracts               = "Cataracts",
  smoking_status          = "Smoking status"
)
# Outcome labels (used for facet titles)
olab <- c(
  Mexiletine      = "Mexiletine:\nMyotonia change, Wk12 (neg = better)",
  Oligonucleotide = "Oligonucleotide:\nQMT change, Wk26 (pos = better)",
  Tideglusib      = "Tideglusib:\nComposite score, Wk26 (pos = better)"
)

dat$treatment_arm <- factor(dat$treatment_arm, levels = arms)

################################################################################
# PART 1  --  DISTRIBUTIONS (understand structure of covariates & outcomes)
################################################################################

## F1-F3: Multipanel histograms of the 13 continuous covariates, by arm --------
make_cont_hist <- function(a) {
  sub <- dat[dat$treatment_arm == a, ]
  c_long <- sub %>%
    select(all_of(cont_cov)) %>%
    pivot_longer(everything(), names_to = "cov", values_to = "val") %>%
    mutate(label = factor(lab[cov], levels = lab[cont_cov]))

  ggplot(c_long, aes(val)) +
    geom_histogram(bins = 30, fill = pal[1], colour = "white", linewidth = 0.15) +
    facet_wrap(~label, scales = "free", labeller = label_wrap_gen(width = 22)) +
    labs(x = NULL, y = "Count",
         title = paste0(a, ": distributions of continuous baseline covariates (N = ", nrow(sub), ")"))
}

save_fig(make_cont_hist("Mexiletine"),
         "01_mexiletine_continuous_covariate_histograms", width = 12, height = 8)
save_fig(make_cont_hist("Oligonucleotide"),
         "02_oligonucleotide_continuous_covariate_histograms", width = 12, height = 8)
save_fig(make_cont_hist("Tideglusib"),
         "03_tideglusib_continuous_covariate_histograms", width = 12, height = 8)

## F15-F17: Same continuous covariate histograms, training vs. test ------------
make_cont_hist_split <- function(a) {
  sub <- dat[dat$treatment_arm == a, ]
  c_long <- sub %>%
    select(all_of(cont_cov), split) %>%
    pivot_longer(all_of(cont_cov), names_to = "cov", values_to = "val") %>%
    mutate(label = factor(lab[cov], levels = lab[cont_cov]))
  n_lab <- sub %>% count(split) %>% mutate(txt = paste0(split, " (n=", n, ")"))

  ggplot(c_long, aes(val, after_stat(density), fill = split)) +
    geom_histogram(bins = 30, colour = "white", linewidth = 0.15,
                   position = "identity", alpha = 0.55) +
    facet_wrap(~label, scales = "free", labeller = label_wrap_gen(width = 22)) +
    labs(x = NULL, y = "Density", fill = NULL,
         title = paste0(a, ": continuous covariates, training vs. test (",
                        paste(n_lab$txt, collapse = "; "), ")"))
}
save_fig(make_cont_hist_split("Mexiletine"),
         "15_mexiletine_continuous_covariate_histograms_by_split", width = 12, height = 8)
save_fig(make_cont_hist_split("Oligonucleotide"),
         "16_oligonucleotide_continuous_covariate_histograms_by_split", width = 12, height = 8)
save_fig(make_cont_hist_split("Tideglusib"),
         "17_tideglusib_continuous_covariate_histograms_by_split", width = 12, height = 8)

## F4-F6: Multipanel bar charts of the 5 categorical covariates, by arm ---------
make_cat_bar <- function(a) {
  sub <- dat[dat$treatment_arm == a, ]
  cat_long <- sub %>%
    select(all_of(cat_cov)) %>%
    mutate(across(everything(), as.character)) %>%
    pivot_longer(everything(), names_to = "cov", values_to = "level") %>%
    mutate(label = factor(lab[cov], levels = lab[cat_cov]))

  ggplot(cat_long, aes(level)) +
    geom_bar(fill = pal[2]) +
    geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3, size = 3) +
    facet_wrap(~label, scales = "free_x", nrow = 1,
               labeller = label_wrap_gen(width = 22)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = NULL, y = "Count",
         title = paste0(a, ": distributions of categorical baseline covariates (N = ", nrow(sub), ")"))
}

save_fig(make_cat_bar("Mexiletine"),
         "04_mexiletine_categorical_covariate_barplots", width = 12, height = 4)
save_fig(make_cat_bar("Oligonucleotide"),
         "05_oligonucleotide_categorical_covariate_barplots", width = 12, height = 4)
save_fig(make_cat_bar("Tideglusib"),
         "06_tideglusib_categorical_covariate_barplots", width = 12, height = 4)

## F18-F20: Same categorical covariates, training vs. test (as proportions) ----
# Proportions (not counts) are used within each split so unequal training vs.
# test sample sizes don't distort the comparison.
make_cat_bar_split <- function(a) {
  sub <- dat[dat$treatment_arm == a, ]
  cat_prop <- sub %>%
    select(all_of(cat_cov), split) %>%
    mutate(across(all_of(cat_cov), as.character)) %>%
    pivot_longer(all_of(cat_cov), names_to = "cov", values_to = "level") %>%
    count(cov, split, level) %>%
    group_by(cov, split) %>%
    mutate(prop = n / sum(n)) %>%
    ungroup() %>%
    mutate(label = factor(lab[cov], levels = lab[cat_cov]))
  n_lab <- sub %>% count(split) %>% mutate(txt = paste0(split, " (n=", n, ")"))

  ggplot(cat_prop, aes(level, prop, fill = split)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    facet_wrap(~label, scales = "free_x", nrow = 1,
               labeller = label_wrap_gen(width = 22)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
    scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.12))) +
    labs(x = NULL, y = "% of split", fill = NULL,
         title = paste0(a, ": categorical covariates, training vs. test (",
                        paste(n_lab$txt, collapse = "; "), ")"))
}
save_fig(make_cat_bar_split("Mexiletine"),
         "18_mexiletine_categorical_covariate_barplots_by_split", width = 12, height = 4)
save_fig(make_cat_bar_split("Oligonucleotide"),
         "19_oligonucleotide_categorical_covariate_barplots_by_split", width = 12, height = 4)
save_fig(make_cat_bar_split("Tideglusib"),
         "20_tideglusib_categorical_covariate_barplots_by_split", width = 12, height = 4)

## F3: Distribution of each arm's continuous efficacy outcome -------------------
out_df <- map_dfr(arms, function(a) {
  sub <- dat[dat$treatment_arm == a, ]
  tibble(arm = a, outcome_label = olab[a], val = sub[[outcome_of[a]]])
}) %>% filter(!is.na(val)) %>%
  mutate(outcome_label = factor(outcome_label, levels = olab[arms]))

f3 <- ggplot(out_df, aes(val, fill = arm)) +
  geom_histogram(bins = 30, colour = "white", linewidth = 0.15) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey30") +
  facet_wrap(~outcome_label, scales = "free", labeller = label_wrap_gen(width = 40)) +
  labs(x = "Change / score at endpoint", y = "Count", fill = "Arm",
       title = "Continuous efficacy outcome distribution, by treatment arm") +
  theme(legend.position = "none")
save_fig(f3, "07_outcome_distributions_by_arm", width = 12, height = 4.5)

## F21: Continuous efficacy outcome distribution, training vs. test, by arm ----
out_df_split <- map_dfr(arms, function(a) {
  sub <- dat[dat$treatment_arm == a, ]
  tibble(arm = a, outcome_label = olab[a], val = sub[[outcome_of[a]]], split = sub$split)
}) %>% filter(!is.na(val)) %>%
  mutate(outcome_label = factor(outcome_label, levels = olab[arms]))

f21 <- ggplot(out_df_split, aes(val, after_stat(density), fill = split)) +
  geom_histogram(bins = 30, colour = "white", linewidth = 0.15,
                 position = "identity", alpha = 0.55) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey30") +
  facet_wrap(~outcome_label, scales = "free", labeller = label_wrap_gen(width = 40)) +
  labs(x = "Change / score at endpoint", y = "Density", fill = NULL,
       title = "Continuous efficacy outcome distribution, training vs. test")
save_fig(f21, "21_outcome_distributions_by_split", width = 12, height = 4.5)

################################################################################
# PART 2  --  COVARIATE vs OUTCOME (screen for predictors, per arm)
################################################################################

## F4-F6: Continuous covariate vs outcome scatter grids (one per arm) -----------
make_scatter <- function(a) {
  y   <- unname(outcome_of[a])
  sub <- dat[dat$treatment_arm == a, ]
  long <- sub %>%
    select(all_of(c(cont_cov, y))) %>%
    pivot_longer(all_of(cont_cov), names_to = "cov", values_to = "val") %>%
    mutate(label = factor(lab[cov], levels = lab[cont_cov]))
  ggplot(long, aes(val, .data[[y]])) +
    geom_point(alpha = 0.4, colour = pal[1], size = 1) +
    geom_smooth(method = "lm", se = TRUE, colour = "black",
                linetype = "dashed", linewidth = 0.6, formula = y ~ x) +
    facet_wrap(~label, scales = "free_x", labeller = label_wrap_gen(width = 22)) +
    labs(x = NULL, y = unname(olab[a]),
         title = paste0(a, ": continuous covariates vs efficacy outcome"))
}
scatter_dims <- list(width = 12, height = 8.5)
save_fig(make_scatter("Mexiletine"),
         "08_mexiletine_continuous_cov_vs_outcome",
         width = scatter_dims$width, height = scatter_dims$height)
save_fig(make_scatter("Oligonucleotide"),
         "09_oligonucleotide_continuous_cov_vs_outcome",
         width = scatter_dims$width, height = scatter_dims$height)
save_fig(make_scatter("Tideglusib"),
         "10_tideglusib_continuous_cov_vs_outcome",
         width = scatter_dims$width, height = scatter_dims$height)

## F7-F9: Categorical covariate vs outcome boxplots (one per arm) ---------------
make_box <- function(a) {
  y   <- unname(outcome_of[a])
  sub <- dat[dat$treatment_arm == a, ]
  long <- sub %>%
    select(all_of(c(cat_cov, y))) %>%
    mutate(across(all_of(cat_cov), as.character)) %>%
    pivot_longer(all_of(cat_cov), names_to = "cov", values_to = "level") %>%
    mutate(label = factor(lab[cov], levels = lab[cat_cov]))
  ggplot(long, aes(level, .data[[y]])) +
    geom_boxplot(fill = pal[2], alpha = 0.5, outlier.shape = NA) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 0.8) +
    facet_wrap(~label, scales = "free_x", nrow = 1,
               labeller = label_wrap_gen(width = 22)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
    labs(x = NULL, y = unname(olab[a]),
         title = paste0(a, ": efficacy outcome by categorical covariate"))
}
box_dims <- list(width = 12, height = 4.5)
save_fig(make_box("Mexiletine"),
         "11_mexiletine_categorical_cov_vs_outcome",
         width = box_dims$width, height = box_dims$height)
save_fig(make_box("Oligonucleotide"),
         "12_oligonucleotide_categorical_cov_vs_outcome",
         width = box_dims$width, height = box_dims$height)
save_fig(make_box("Tideglusib"),
         "13_tideglusib_categorical_cov_vs_outcome",
         width = box_dims$width, height = box_dims$height)

## F10: Spearman correlation of each continuous covariate with each arm outcome -
cor_df <- expand.grid(cov = cont_cov, arm = arms,
                      stringsAsFactors = FALSE) %>%
  rowwise() %>%
  mutate(r = {
    sub <- dat[dat$treatment_arm == arm, ]
    y <- sub[[outcome_of[arm]]]
    x <- sub[[cov]]
    ok <- stats::complete.cases(x, y)
    suppressWarnings(stats::cor(x[ok], y[ok], method = "spearman"))
  }) %>%
  ungroup() %>%
  mutate(label = factor(lab[cov], levels = rev(lab[cont_cov])),
         arm   = factor(arm, levels = arms))

f10 <- ggplot(cor_df, aes(arm, label, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.2f", r)), size = 3) +
  scale_fill_gradient2(low = "#0571b0", mid = "white", high = "#ca0020",
                       midpoint = 0, limits = c(-1, 1), name = "Spearman r") +
  labs(x = NULL, y = NULL,
       title = "Correlation of continuous covariates\nwith each arm's efficacy outcome",
       caption = "Spearman rho, within-arm complete cases. Sign is relative to each outcome's raw scale.") +
  theme(legend.position = "right")
save_fig(f10, "14_covariate_outcome_correlation_heatmap", width = 7, height = 8)

################################################################################
# PART 3  --  SUMMARY TABLES
################################################################################

## T1: EDA summary by arm (classic "Table 1", covariates x arms) ----------------
fmt_cont <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return("-")
  sprintf("%.1f (%.1f); %.1f [%.1f, %.1f]",
          mean(x), sd(x), median(x),
          quantile(x, 0.25), quantile(x, 0.75))
}
grp_idx <- c(as.list(setNames(lapply(arms, function(a) dat$treatment_arm == a), arms)),
             list(Overall = rep(TRUE, nrow(dat))))
col_names <- c(arms, "Overall")

# N header row
n_row <- data.frame(
  Covariate = "N (patients)", Statistic = "", stringsAsFactors = FALSE)
for (cn in col_names) n_row[[cn]] <- as.character(sum(grp_idx[[cn]]))

# Continuous rows: "Mean (SD); Median [Q1, Q3]"
cont_rows <- map_dfr(cont_cov, function(cov) {
  row <- data.frame(Covariate = unname(lab[cov]),
                    Statistic = "Mean (SD); Median [Q1, Q3]",
                    stringsAsFactors = FALSE)
  for (cn in col_names) row[[cn]] <- fmt_cont(dat[[cov]][grp_idx[[cn]]])
  row
})

# Categorical rows: one row per level, "n (%)"
cat_rows <- map_dfr(cat_cov, function(cov) {
  levs <- sort(unique(as.character(dat[[cov]])))
  map_dfr(seq_along(levs), function(i) {
    l <- levs[i]
    row <- data.frame(
      Covariate = if (i == 1) unname(lab[cov]) else "",
      Statistic = paste0(l, ", n (%)"),
      stringsAsFactors = FALSE)
    for (cn in col_names) {
      v <- as.character(dat[[cov]][grp_idx[[cn]]])
      n <- sum(v == l)
      row[[cn]] <- sprintf("%d (%.1f%%)", n, 100 * n / length(v))
    }
    row
  })
})

tab1 <- bind_rows(n_row, cont_rows, cat_rows)
write_csv(tab1, file.path(tab_dir, "eda_summary_by_arm.csv"))
save_table_png(
  tab1,
  stem = "eda_summary_by_arm",
  title = "Baseline covariate summary by treatment arm",
  subtitle = "ACoP 2026 DM1 Hackathon (Team 9) - simulated data, N = 500",
  width = 16,
  height = max(10, 0.28 * nrow(tab1) + 2.2)
)

## T2: Covariate -> outcome association screen (per arm) ------------------------
fmt_p <- function(p) ifelse(is.na(p), "-",
                            ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))

screen_rows <- bind_rows(
  # Continuous: Spearman rho (p)
  map_dfr(cont_cov, function(cov) {
    row <- data.frame(Covariate = unname(lab[cov]), Type = "Continuous",
                      Test = "Spearman rho (p)", stringsAsFactors = FALSE)
    for (a in arms) {
      sub <- dat[dat$treatment_arm == a, ]
      y <- sub[[outcome_of[a]]]; x <- sub[[cov]]
      ok <- stats::complete.cases(x, y)
      if (stats::sd(x[ok]) == 0) {
        row[[a]] <- "constant (no variance)"
      } else {
        ct <- suppressWarnings(stats::cor.test(x[ok], y[ok], method = "spearman"))
        star <- if (!is.na(ct$p.value) && ct$p.value < 0.05) " *" else ""
        row[[a]] <- sprintf("%.2f (p=%s)%s", unname(ct$estimate), fmt_p(ct$p.value), star)
      }
    }
    row
  }),
  # Categorical: Kruskal-Wallis p across levels
  map_dfr(cat_cov, function(cov) {
    row <- data.frame(Covariate = unname(lab[cov]), Type = "Categorical",
                      Test = "Kruskal-Wallis (p)", stringsAsFactors = FALSE)
    for (a in arms) {
      sub <- dat[dat$treatment_arm == a, ]
      y <- sub[[outcome_of[a]]]; g <- factor(sub[[cov]])
      ok <- stats::complete.cases(y, g)
      kt <- suppressWarnings(stats::kruskal.test(y[ok], g[ok]))
      star <- if (!is.na(kt$p.value) && kt$p.value < 0.05) " *" else ""
      row[[a]] <- sprintf("p=%s%s", fmt_p(kt$p.value), star)
    }
    row
  })
)
write_csv(screen_rows, file.path(tab_dir, "covariate_outcome_screen.csv"))
save_table_png(
  screen_rows,
  stem = "covariate_outcome_screen",
  title = "Univariate covariate -> continuous outcome association, by arm",
  subtitle = "Screening only; not adjusted for multiplicity or confounders",
  width = 16,
  height = max(10, 0.34 * nrow(screen_rows) + 2.2)
)

message("\nEDA complete. Figures in wangui/eda_figures_fulldataset/, tables in wangui/eda_tables_fulldataset/.")
