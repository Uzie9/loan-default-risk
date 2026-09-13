# Loan Default Risk Prediction

A classification pipeline that predicts the probability a borrower will
experience serious delinquency in the next two years, with a business-cost
framing rather than a plain-accuracy one: **which threshold minimizes total
cost, given a missed default is far more expensive than an unnecessary
denial?**

Built on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
dataset structure — 150K+ borrower records with financial, credit-history,
and delinquency features.

## Why this project

Credit risk scoring is one of the clearest real-world classification
problems: imbalanced classes, features that need domain-aware cleaning
(not just `dropna()`), and a decision that has to be explainable to a
non-technical stakeholder before it can be used. This project is built to
demonstrate that full loop — not just a model that scores well on paper.

## Pipeline

```
data/raw/cs-training.csv
        │
        ▼
  src/data_loader.py        -- load + sanity checks (missingness, placeholder codes)
        │
        ▼
  src/feature_engineering.py -- cap placeholder codes, median-impute with
                                 missingness flags, engineer ratios/buckets
        │
        ▼
  src/train.py               -- Logistic Regression / Random Forest / XGBoost,
                                 class-weighted for imbalance
        │
        ▼
  src/evaluate.py            -- ROC-AUC, precision/recall, cost-optimal
                                 threshold (not the default 0.5)
        │
        ▼
  src/explain.py             -- SHAP: global feature importance + one
                                 individual prediction walked through
```

## Getting started

```bash
pip install -r requirements.txt

# Option A: use the included synthetic data generator to test the full
# pipeline immediately (mirrors the real dataset's schema and quirks)
python main.py

# Option B: use the real Kaggle data
#   1. Download cs-training.csv from the Kaggle competition above
#   2. Place it at data/raw/cs-training.csv
#   3. python main.py   (it will detect the real file and skip synthetic generation)
```

Outputs land in:
- `models/` — trained model artifacts (`.joblib`)
- `reports/model_comparison.csv` — AUC + cost-optimal threshold per model
- `reports/shap_summary.png` — global feature importance
- `reports/shap_individual_example.png` — one prediction explained

## Key design decisions (interview talking points)

- **Imbalance handling**: class weighting rather than SMOTE. Simpler to
  reason about, and avoids synthetic minority samples leaking signal
  across train/test splits.
- **Imputation**: median (not mean) for skewed financial fields, paired
  with a `_was_missing` flag so the model can use missingness itself as a
  signal if it's informative.
- **Placeholder code handling**: the real dataset has known data-quality
  bugs (late-payment counts of 96/98 as error codes) — capped rather than
  dropped, since dropping would bias the sample toward "clean" records.
- **Threshold selection**: cost-based, not the default 0.5. A 10:1
  cost ratio (missed default vs. unnecessary denial) is assumed — this is
  the number you'd actually negotiate with a real lending team.
- **Explainability**: SHAP over feature importances alone, because it
  supports both a global view ("what drives risk in general") and a local
  one ("why was this specific applicant flagged") — the second is what a
  compliance or ops stakeholder actually needs.

## Extending this project

- Swap the cost ratio in `src/evaluate.py` to model different lending
  policies and show how the optimal threshold shifts.
- Add a simple Flask/FastAPI endpoint to serve predictions (turns this
  into a deployable artifact, not just a notebook).
- Feed `reports/model_comparison.csv` and top SHAP features into a Power
  BI or Tableau dashboard for a stakeholder-facing view — pairs well with
  a companion BI-focused project.
