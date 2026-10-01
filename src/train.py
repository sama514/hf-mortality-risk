"""Heart-failure mortality risk: leakage-aware, honestly evaluated baseline."""
import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# `time` (follow-up days) is deliberately excluded: it is only known AFTER the
# outcome (patients who die have shorter follow-up) -> target leakage.
FEATURES = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
            "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
            "sex", "smoking"]
TARGET = "DEATH_EVENT"
SCORING = ["roc_auc", "average_precision", "neg_brier_score"]


def synthetic(n=300, seed=0):
    """Same schema, fake values. For smoke tests ONLY, never for reported results."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": r.normal(60, 12, n).clip(40, 95), "anaemia": r.integers(0, 2, n),
        "creatinine_phosphokinase": r.lognormal(6, 1, n), "diabetes": r.integers(0, 2, n),
        "ejection_fraction": r.normal(38, 12, n).clip(14, 80),
        "high_blood_pressure": r.integers(0, 2, n),
        "platelets": r.normal(260000, 95000, n).clip(25000),
        "serum_creatinine": r.lognormal(0.1, 0.4, n), "serum_sodium": r.normal(137, 4, n),
        "sex": r.integers(0, 2, n), "smoking": r.integers(0, 2, n)})
    z = -1.0 + 0.04 * (df.age - 60) - 0.05 * (df.ejection_fraction - 38) + 0.9 * (df.serum_creatinine - 1.1)
    df[TARGET] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def boot_auc_ci(y, p, seed, n=1000):
    r, y, v = np.random.default_rng(seed), np.asarray(y), []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart_failure_clinical_records_dataset.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.synthetic:
        print("WARNING: synthetic data - smoke test only, results are meaningless.")
        df = synthetic()
    elif Path(a.data).exists():
        df = pd.read_csv(a.data)
    else:
        raise SystemExit(f"Dataset not found at {a.data}. See README > Data.")

    X, y = df[FEATURES], df[TARGET]
    # Test set is touched exactly once, at the very end.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)

    models = {
        "baseline (prior)": DummyClassifier(strategy="prior"),
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=a.seed),
        "hist gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=a.seed),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=a.seed)
    cv_rows, best, best_ap = [], None, -1
    for name, m in models.items():
        s = cross_validate(m, Xtr, ytr, cv=cv, scoring=SCORING)
        auc, ap_, br = (s[f"test_{k}"] for k in SCORING)
        cv_rows.append(f"| {name} | {auc.mean():.3f} ± {auc.std():.3f} | {ap_.mean():.3f} ± {ap_.std():.3f} | {-br.mean():.3f} |")
        if not name.startswith("baseline") and ap_.mean() > best_ap:
            best, best_ap = name, ap_.mean()

    # Operating threshold chosen on train out-of-fold predictions (target sensitivity 80%).
    oof = cross_val_predict(clone(models[best]), Xtr, ytr, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=a.seed))[:, 1]
    _, tpr, th = roc_curve(ytr, oof)
    thr = float(th[np.argmax(tpr >= 0.8)])

    final = clone(models[best]).fit(Xtr, ytr)
    p = final.predict_proba(Xte)[:, 1]
    pred = p >= thr
    lo, hi = boot_auc_ci(yte, p, a.seed)
    imp = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=a.seed)
    imp_s = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

    Path("reports").mkdir(exist_ok=True)
    frac, mean_p = calibration_curve(yte, p, n_bins=5, strategy="quantile")
    plt.figure(figsize=(4, 4)); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.plot(mean_p, frac, "o-"); plt.xlabel("Predicted risk"); plt.ylabel("Observed rate")
    plt.title("Calibration (test set)"); plt.tight_layout(); plt.savefig("reports/calibration.png", dpi=120)

    note = "**SYNTHETIC DATA - meaningless**\n\n" if a.synthetic else ""
    Path("reports/metrics.md").write_text(
        f"{note}## Cross-validation (train set, 5x5 repeated stratified)\n\n"
        "| Model | ROC-AUC | PR-AUC | Brier |\n|---|---|---|---|\n" + "\n".join(cv_rows) +
        f"\n\nSelected by PR-AUC: **{best}** (event rate in train: {ytr.mean():.2f}, so PR-AUC baseline ≈ {ytr.mean():.2f}).\n\n"
        f"## Held-out test set (n={len(yte)}, evaluated once)\n\n"
        f"- ROC-AUC: {roc_auc_score(yte, p):.3f} (95% bootstrap CI {lo:.3f}-{hi:.3f})\n"
        f"- PR-AUC: {average_precision_score(yte, p):.3f}\n- Brier: {brier_score_loss(yte, p):.3f}\n"
        f"- At threshold {thr:.2f}: sensitivity {recall_score(yte, pred):.2f}, "
        f"specificity {recall_score(yte, pred, pos_label=0):.2f}, PPV {precision_score(yte, pred, zero_division=0):.2f}\n\n"
        "## Permutation importance (test set, drop in ROC-AUC)\n\n" +
        "\n".join(f"- {k}: {v:.3f}" for k, v in imp_s.items()) + "\n")
    joblib.dump({"model": final, "features": FEATURES, "threshold": thr}, "reports/model.joblib")
    print(Path("reports/metrics.md").read_text())


if __name__ == "__main__":
    main()
