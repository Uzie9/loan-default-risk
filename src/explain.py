"""
SHAP explainability for the best-performing model (XGBoost by default).
This is the piece that turns "I built a model" into "I can explain a model's
decisions to a non-technical stakeholder" -- the skill that actually
overlaps with BI/dashboard work.
"""
import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe for scripts/CI
import matplotlib.pyplot as plt
import shap


def explain_model(model_name="xgboost", n_samples=500):
    X_train, X_test, y_train, y_test = joblib.load("models/data_split.joblib")
    model = joblib.load(f"models/{model_name}.joblib")

    sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)

    # Global summary: which features matter most overall
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150)
    plt.close()
    print("Saved reports/shap_summary.png")

    # Local explanation: one individual prediction, the kind you'd walk
    # a stakeholder through ("here's why THIS applicant was flagged")
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_individual_example.png", dpi=150)
    plt.close()
    print("Saved reports/shap_individual_example.png")


if __name__ == "__main__":
    explain_model()
