"""
Generates a synthetic dataset that mirrors the schema of the "Give Me Some
Credit" Kaggle dataset (https://www.kaggle.com/c/GiveMeSomeCredit), so the
pipeline can be run end-to-end before you swap in the real data.

Real dataset columns (target: SeriousDlqin2yrs):
  SeriousDlqin2yrs, RevolvingUtilizationOfUnsecuredLines, age,
  NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, MonthlyIncome,
  NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
  NumberRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse,
  NumberOfDependents

To use the REAL data instead:
  1. Download cs-training.csv from Kaggle's "Give Me Some Credit" competition.
  2. Place it at data/raw/cs-training.csv (same column names as above).
  3. Skip this script entirely -- the pipeline reads whatever is in
     data/raw/cs-training.csv.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 20000


def generate(n=N):
    age = RNG.integers(21, 80, n)
    monthly_income = np.round(RNG.gamma(shape=3.0, scale=2000, size=n) + 500, 2)
    debt_ratio = np.round(RNG.beta(2, 5, n) * 2, 4)
    revolving_util = np.round(RNG.beta(1.5, 3, n) * 1.3, 4)
    open_credit_lines = RNG.poisson(7, n)
    real_estate_loans = RNG.poisson(1, n)
    dependents = RNG.poisson(0.8, n)

    # Late-payment counts: mostly zero, occasional bursts (mimics real-world sparsity)
    late_30_59 = RNG.poisson(0.3, n)
    late_60_89 = RNG.poisson(0.1, n)
    late_90 = RNG.poisson(0.08, n)

    # Introduce missingness the way real financial data actually has it
    income_missing_mask = RNG.random(n) < 0.20
    monthly_income = monthly_income.astype(float)
    monthly_income[income_missing_mask] = np.nan

    dependents_missing_mask = RNG.random(n) < 0.03
    dependents = dependents.astype(float)
    dependents[dependents_missing_mask] = np.nan

    # Build a latent default-risk score from the features, then binarize
    # (this is what makes the synthetic target *learnable*, not random noise)
    income_filled = np.nan_to_num(monthly_income, nan=np.nanmedian(monthly_income))
    risk_score = (
        0.9 * late_90
        + 0.5 * late_60_89
        + 0.3 * late_30_59
        + 1.2 * revolving_util
        + 0.8 * debt_ratio
        - 0.00006 * income_filled
        - 0.01 * (age - 40)
        + RNG.normal(0, 0.6, n)  # noise so it's not trivially separable
    )
    threshold = np.quantile(risk_score, 0.93)  # ~7% default rate, similar to real data
    target = (risk_score > threshold).astype(int)

    df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": target,
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/raw/cs-training.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Default rate: {df['SeriousDlqin2yrs'].mean():.2%}")
