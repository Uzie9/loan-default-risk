"""
Evaluates all trained models beyond plain accuracy, and picks an operating
threshold based on business cost rather than the default 0.5 cutoff.

Cost framing: a missed default (false negative) costs far more than an
unnecessary denial (false positive). We assume a 10:1 cost ratio here --
tune COST_FN / COST_FP to match a real lender's numbers.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)

COST_FN = 10  # cost of an undetected default
COST_FP = 1   # cost of an unnecessary denial


def get_probas(model, X, scaler=None):
    X_input = scaler.transform(X) if scaler is not None else X
    return model.predict_proba(X_input)[:, 1]


def find_cost_optimal_threshold(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_cost = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn * COST_FN + fp * COST_FP
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold, best_cost


def evaluate_model(name, model, X_test, y_test, scaler=None):
    y_proba = get_probas(model, X_test, scaler)
    auc = roc_auc_score(y_test, y_proba)

    best_threshold, best_cost = find_cost_optimal_threshold(y_test, y_proba)
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\n-- At default 0.5 threshold --")
    print(classification_report(y_test, y_pred_default, digits=3))
    print(f"-- At cost-optimal threshold ({best_threshold:.2f}, "
          f"assuming FN costs {COST_FN}x FP) --")
    print(classification_report(y_test, y_pred_optimal, digits=3))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
    cost_at_default = fn * COST_FN + fp * COST_FP
    savings_pct = (1 - best_cost / cost_at_default) * 100 if cost_at_default else 0
    print(f"Cost at default 0.5 threshold: {cost_at_default:.0f}")
    print(f"Cost at optimal {best_threshold:.2f} threshold: {best_cost:.0f} "
          f"({savings_pct:+.1f}% vs. default)")

    return {"model": name, "auc": auc, "best_threshold": best_threshold, "cost": best_cost}


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = joblib.load("models/data_split.joblib")
    scaler = joblib.load("models/scaler.joblib")

    results = []
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        model = joblib.load(f"models/{name}.joblib")
        use_scaler = scaler if name == "logistic_regression" else None
        results.append(evaluate_model(name, model, X_test, y_test, use_scaler))

    print(f"\n{'=' * 60}\nMODEL COMPARISON SUMMARY\n{'=' * 60}")
    summary = pd.DataFrame(results).sort_values("auc", ascending=False)
    print(summary.to_string(index=False))
    summary.to_csv("reports/model_comparison.csv", index=False)
    print("\nSaved comparison table to reports/model_comparison.csv")
