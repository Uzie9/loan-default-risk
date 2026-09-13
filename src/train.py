"""
Trains and compares three models of increasing complexity:
  1. Logistic Regression -- interpretable baseline
  2. Random Forest       -- non-linear, handles interactions
  3. XGBoost             -- gradient boosting, usually the strongest performer

Class imbalance is handled via class weighting (simpler and more defensible
in an interview than SMOTE, and avoids synthetic-sample leakage across
train/test folds).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_raw, TARGET
from feature_engineering import build_features

RANDOM_STATE = 42


def prepare_data(test_size: float = 0.2):
    df = load_raw()
    df = build_features(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}

    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    logreg.fit(X_train_scaled, y_train)
    models["logistic_regression"] = logreg

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    models["xgboost"] = xgb

    return models, scaler


def save_artifacts(models, scaler, feature_names, out_dir="models"):
    for name, model in models.items():
        joblib.dump(model, f"{out_dir}/{name}.joblib")
    joblib.dump(scaler, f"{out_dir}/scaler.joblib")
    joblib.dump(feature_names, f"{out_dir}/feature_names.joblib")
    print(f"Saved {len(models)} models + scaler + feature list to {out_dir}/")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    models, scaler = train_models(X_train, y_train)
    save_artifacts(models, scaler, list(X_train.columns))

    # Stash the split for evaluate.py / explain.py to reuse without retraining
    joblib.dump((X_train, X_test, y_train, y_test), "models/data_split.joblib")
    print("Training complete.")
