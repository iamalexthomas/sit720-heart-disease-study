# Study plan

Chosen paper: option 3, Bhagat, Sharma and Agarwal, DOI 10.1007/s11042-024-19293-7.
This plan was written after inspecting the paper and data, before fitting models.

## Part 1: reproduce the available description

- Keep all 1,025 rows and all 13 predictors; exclude target from inputs.
- Use an 80/20 shuffled row split, seed 42. This ratio is inferred from the paper's
  205-entry confusion matrices. The seed and stratification are not reported;
  do not stratify this first split. Do not search for a seed that matches results.
- Standardise inputs inside each model's training pipeline. Existing category
  codes are retained. The precise encoder/scaler is not stated in the paper.
- No imputation is needed: the supplied CSV has no blank/NaN values. Keep the
  unusual ca=4 and thal=0 codes; their meaning cannot safely be guessed.
- Fit LR, entropy DT, RF, XGBoost, Gaussian NB, KNN and a five-fold stack of all six.
  Use LR as the unspecified final classifier. Each base pipeline is fit inside
  the stacking folds. Record all parameter assumptions in the report.
- Compare Accuracy, Precision, Recall, F1 and probability-based ROC AUC with
  Table 11. Report specificity and MCC too, but explain errors in the paper's
  corresponding columns. Positive class is the CSV's unchanged target=1.

## Part 2: proposed solution and fair comparison

Question: can a simple pipeline give useful positive-class recall on genuinely
unseen records while avoiding repeated-record leakage and a six-model stack?

- Drop exact duplicate predictor records after checking for conflicting labels.
  This is a data integrity operation; no feature selection uses test labels.
- Use seeds 42 through 51 for ten predeclared stratified 80/20 splits of 302
  unique records. Use identical splits for all methods. Seed 42 is the example
  holdout; repeated-split results are the main empirical comparison.
- Rerun all seven Part 1 models under this duplicate-free protocol.
- Proposed model: standardise five continuous predictors, one-hot encode eight
  discrete predictors, fit L2 logistic regression (C=1), and select a threshold
  from 0.20 to 0.80 in steps of 0.05 using five-fold out-of-fold predictions
  from the outer training set only. Maximise F2, which weights recall more than
  precision. Break ties by proximity to 0.5, then lower threshold.
- This threshold is an educational objective, not a clinically validated rule.
  All recall statements refer to target=1, following the paper's convention.
- Ablations: encoded/scaled LR (already a reproduced baseline), one-hot LR at
  0.5, and the full proposed pipeline. Keep model settings fixed across seeds.
- Report all five required metrics, mean and sample SD across splits, paired
  metric differences and wins/ties/losses versus stacking. Reused records make
  the split results dependent; do not call these independent trials or use an
  ordinary t-test. Do not promise that the proposed model will improve results.
- Generate ROC curves and confusion matrices. Save predictions and split IDs
  so scores can be independently recomputed.

## Limits to claims

The paper lacks enough detail for an exact replication. This is a transparent
reproduction under stated assumptions. A lower score after removing overlap
does not alone isolate a causal effect: the training sample and split also change.
No patient identifiers or external validation data are available. Unique rows
are not proof of unique patients. This study classifies recorded labels, and
does not establish prediction of a future heart attack.
