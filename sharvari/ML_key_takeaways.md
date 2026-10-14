# ML results: key takeaways (Team 9, ACoP 2026 DM1 hackathon)

## 1. Key takeaways

- **ML found no hidden predictors.** Five model types (random forest, XGBoost, elastic net,
  spline + ridge, and SVM regression (SVR)) all point to the same main covariate in each arm. When very
  different methods agree, that covariate is most likely a real driver of response, not noise.
- **For mexiletine and the oligonucleotide, a simple model is as good as ML.** A penalized linear
  model (elastic net) predicts as well as random forest or XGBoost. More complexity bought nothing.
- **For tideglusib, ML clearly helps.** Tree-based models beat every simpler model, because
  the response does not follow straight lines (see section 4).
- **What drives response matches how each drug works:** myotonia severity for mexiletine (a
  symptomatic sodium channel blocker), CTG repeat length for the oligonucleotide (which targets
  the toxic DMPK RNA), and cognition for tideglusib (a cognition + motor endpoint).
- **The patients with the most room to improve benefit the most:** worse myotonia, longer CTG
  repeats, worse cognition.

### What to emphasize in the presentation

- **One honest comparison.** Every model was trained on the same shared training set, compared
  on the identical cross-validation folds (5-fold × 3 repeats, with tuning done inside each fold),
  and scored on the same held-out test patients. The best model was chosen by cross-validation,
  not by its test score.
- **Agreement across methods.** This is our strongest evidence for separating signal from noise.
- **Use the right tool for each arm.** Keep the interpretable model where it ties with ML (2 arms).
  Use ML where the data are nonlinear (1 arm).
- **Be upfront about uncertainty.** There are only about 35 test patients per arm, so test-set
  confidence intervals are wide. Small score differences between models are not meaningful.
- **Oligonucleotide response is hard to predict** from baseline covariates (test R² ≈ 0.1–0.2).
  The main driver is clear, but most of the patient-to-patient variation is not explained.
  Say so plainly.
- **Signs:** the mexiletine outcome is negative = improvement. Always state effects as "more
  improvement" or "less improvement", never just "higher" or "lower".

### Best model to share per arm (full covariate set)

| Arm | Model to share | Why | Cross-validation RMSE (mean ± SD) | Test RMSE [95% CI] | Test R² [95% CI] |
|---|---|---|---|---|---|
| Mexiletine | **Elastic net** | Lowest cross-validation error and interpretable. Random forest is a tie (CV 0.789; test R² 0.72). | 0.771 ± 0.112 | 0.83 [0.51, 1.18] | 0.68 [0.56, 0.79] |
| Oligonucleotide | **Elastic net** | Lowest cross-validation error, but all models are within about 0.03 of each other. Random forest is a tie (CV 1.534; test R² 0.19). | 1.505 ± 0.135 | 1.73 [1.38, 2.05] | 0.12 [−0.15, 0.25] |
| Tideglusib | **XGBoost (tuned)** | Beats elastic net in 14 of 15 cross-validation folds, and spline + ridge and SVR in all 15. | 2.053 ± 0.334 | 2.33 [1.83, 2.79] | 0.53 [0.26, 0.68] |

Results on the reduced covariate sets are a sensitivity check only. The reduced sets were
chosen using the same training data the models were then scored on, so their cross-validation
scores are optimistic. Don't present them as the best models.

## 2. Top covariates per arm

| Arm | Main driver (all methods agree) | Supporting covariates (weaker, some methods only) | In improvement terms |
|---|---|---|---|
| Mexiletine | **myotonia_severity** | ctg_repeats ranks #2 in the tree models, but elastic net kept it in only 2 of 15 folds. It is most likely standing in for myotonia severity rather than having its own effect. | Worse baseline myotonia → larger improvement |
| Oligonucleotide | **ctg_repeats** | myotonia_severity, somatic_mosaicism_index, insulin_resistance | Longer CTG repeats → larger QMT gain |
| Tideglusib | **cognitive_function** | muscle_strength (curved effect; found only by models that allow curves), pr_interval, age | Worse baseline cognition → more improvement. Mid-range muscle strength → most improvement. Shorter PR interval and younger age → more improvement. |

## 3. Where ML helped

- **Tideglusib, in both prediction and insight.**
  - XGBoost had lower prediction error than elastic net by about 0.4 RMSE on average in
    cross-validation, and beat it in 14 of 15 folds. Test R² was 0.53 for XGBoost and 0.30 for
    elastic net.
  - ML picked up a curved effect of muscle_strength: patients with mid-range strength improved
    the most. A straight-line model misses this; elastic net almost never kept muscle_strength.
  - Smooth-curve models (spline + ridge, SVR) captured part of the gain but not all of it. This
    suggests covariates also interact with each other, which tree models handle naturally.
- **Mexiletine and oligonucleotide: ML served as a stress test.** It found nothing beyond what
  the simple model already captured. That answer is useful in itself: these arms behave close
  to linearly, so the interpretable model is enough.
