"""Per-arm settings for the ML pipeline (Team 9, ACoP 2026 DM1 hackathon).

Edit `ARMS[<arm>]["covariates"]` once the EDA team finalizes each arm's list.
Everything else in the pipeline reads from here, so the three arm notebooks stay identical.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "derived" / "split"
TRAIN_PATH = SPLIT_DIR / "training_shared.csv"
TEST_PATH = SPLIT_DIR / "testing_shared.csv"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

SEED = 2026

ID_COL = "patient_id"
ARM_COL = "treatment_arm"

CAT_COLS = ["phenotype", "sex", "family_history", "cataracts", "smoking_status"]
NUM_COLS = [
    "ctg_repeats", "somatic_mosaicism_index", "age", "age_at_onset", "disease_duration",
    "myotonia_severity", "muscle_strength", "pr_interval", "cognitive_function",
    "ck_level", "insulin_resistance", "albumin", "bmi",
]

# Never allowed as model inputs: identifiers, the arm, every outcome and every responder label
# (including the coalesced `responder` column in the shared split files).
LEAKY_COLS = [
    ID_COL, ARM_COL, "responder",
    "myotonia_change_12wk", "qmt_change_26wk", "tideglusib_composite",
    "responder_phenotype_mex", "responder_phenotype_oligo", "responder_phenotype_tidge",
]

# Placeholder covariate lists: all 18 baseline covariates minus `age`
# (age == age_at_onset + disease_duration exactly). Albumin is constant in the Mexiletine arm.
# TODO: replace with the EDA team's final list per arm.
_DEFAULT_COVARIATES = [c for c in NUM_COLS if c != "age"] + CAT_COLS

ARMS = {
    "Mexiletine": {
        "slug": "mexiletine",
        "outcome": "myotonia_change_12wk",
        "outcome_label": "Change in myotonia severity, week 12",
        "better": "lower (negative = improvement)",
        "covariates": [c for c in _DEFAULT_COVARIATES if c != "albumin"],
    },
    "Oligonucleotide": {
        "slug": "oligonucleotide",
        "outcome": "qmt_change_26wk",
        "outcome_label": "QMT change, week 26",
        "better": "higher (positive = improvement)",
        "covariates": list(_DEFAULT_COVARIATES),
    },
    "Tideglusib": {
        "slug": "tideglusib",
        "outcome": "tideglusib_composite",
        "outcome_label": "Composite improvement score (0-20), week 26",
        "better": "higher",
        "covariates": list(_DEFAULT_COVARIATES),
    },
}

# Cross-validation design (shared by all arms)
CV_SETTINGS = {
    "outer_splits": 5,     # nested CV, outer loop: scores each modelling procedure
    "outer_repeats": 3,
    "inner_splits": 5,     # nested CV, inner loop: hyperparameter tuning
    "final_splits": 5,     # tuning on the full training set for the final models
    "final_repeats": 2,
    "n_bins": 4,           # folds are stratified on quartiles of the continuous outcome
}
