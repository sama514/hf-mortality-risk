# Heart Failure Mortality Risk: a leakage-aware, honestly evaluated baseline

Estimating mortality risk for heart-failure patients from 11 routine clinical measurements, with emphasis on **correct evaluation** rather than headline accuracy.

## Data
UCI *Heart Failure Clinical Records* (Chicco & Jurman 2020; 299 patients, Faisalabad, Pakistan; CC BY 4.0; ~32% death events).
Download the CSV from the UCI repository (or its Kaggle mirror) and place it at `data/heart_failure_clinical_records_dataset.csv`.

## Design decisions
| Issue | What this project does |
|---|---|
| Target leakage | The `time` column (follow-up days) is **dropped**: it is only known after the outcome. Many public notebooks keep it and report inflated scores. A test guards this. |
| Tiny dataset (n=299) | 20% test set touched **once**; model selection via 5x5 repeated stratified CV on the rest, reported as mean ± std. |
| Class imbalance | Selection by **PR-AUC** and ROC-AUC, not accuracy. Compared against a no-skill baseline. |
| Is it calibrated? | Brier score + calibration plot (`reports/calibration.png`). |
| Clinical usefulness | Threshold fixed on train out-of-fold predictions for ~80% sensitivity; test sensitivity, specificity and PPV reported. |
| Uncertainty | Bootstrap 95% CI on test ROC-AUC. With ~60 test patients, expect it to be wide. |
| Interpretability | Permutation importance on held-out data. |

## Run
```bash
pip install -r requirements.txt
python -m src.train            # writes reports/metrics.md, calibration.png, model.joblib
python -m pytest
streamlit run app.py           # optional demo
python -m src.train --synthetic   # smoke test only; numbers are meaningless
```

## Results
Generated into [`reports/metrics.md`](reports/metrics.md) after you run training. Paste the real table here; nothing is reported that was not produced by the code.
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

## Limitations
- ~300 patients, one hospital, one region: no claim of generalization to other populations (including Egypt) without external validation.
- Observational data; feature importance is **not** causal.
- Not a medical device. The demo app is educational only.

## Future work
External validation on another heart-failure cohort; survival modelling (Cox / random survival forest) using time-to-event properly; fairness check across sex and age groups.
