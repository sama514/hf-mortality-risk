## Cross-validation (train set, 5x5 repeated stratified)

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| baseline (prior) | 0.500 ± 0.000 | 0.322 ± 0.009 | 0.218 |
| logistic regression | 0.790 ± 0.065 | 0.659 ± 0.085 | 0.187 |
| random forest | 0.783 ± 0.065 | 0.679 ± 0.089 | 0.176 |
| hist gradient boosting | 0.722 ± 0.064 | 0.595 ± 0.085 | 0.197 |

Selected by PR-AUC: **random forest** (event rate in train: 0.32, so PR-AUC baseline ≈ 0.32).

## Held-out test set (n=60, evaluated once)

- ROC-AUC: 0.792 (95% bootstrap CI 0.661-0.892)
- PR-AUC: 0.582
- Brier: 0.174
- At threshold 0.32: sensitivity 0.84, specificity 0.61, PPV 0.50

## Permutation importance (test set, drop in ROC-AUC)

- serum_creatinine: 0.087
- age: 0.078
- ejection_fraction: 0.066
- serum_sodium: 0.013
- creatinine_phosphokinase: 0.001
- sex: -0.001
- high_blood_pressure: -0.003
- anaemia: -0.005
- platelets: -0.005
- smoking: -0.005
- diabetes: -0.011
