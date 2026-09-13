"""
Feature engineering with explicit, defensible choices -- the kind of thing
you should be ready to justify in an interview, not just code silently.
"""
import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"


def clean_placeholder_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    The real Kaggle dataset has a known data-quality bug: late-payment
    columns sometimes contain 96 or 98 as placeholder/error codes instead
    of genuine counts. Cap them rather than silently dropping rows.
    """
    df = df.copy()
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in late_cols:
        if col in df.columns:
            df[col] = df[col].clip(upper=20)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    MonthlyIncome and NumberOfDependents are the two columns with real
    missingness. Median imputation is a defensible baseline here (not
    mean, since income is right-skewed) -- flagged with an explicit
    "was_missing" indicator so the model can learn from missingness itself
    if it's informative, rather than pretending it wasn't there.
    """
    df = df.copy()
    for col in ["MonthlyIncome", "NumberOfDependents"]:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total delinquency count across all severity buckets -- often more
    # predictive as a combined signal than any single bucket alone.
    late_cols = [
        c
        for c in [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
        if c in df.columns
    ]
    if late_cols:
        df["TotalTimesLate"] = df[late_cols].sum(axis=1)

    # Income-adjusted debt burden -- DebtRatio alone doesn't capture scale.
    if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
        df["DebtPerIncome"] = df["DebtRatio"] * df["MonthlyIncome"].replace(0, np.nan)
        df["DebtPerIncome"] = df["DebtPerIncome"].fillna(df["DebtPerIncome"].median())

    # Credit lines per age -- a young person with many open lines reads
    # differently than an older person with the same count.
    if "NumberOfOpenCreditLinesAndLoans" in df.columns and "age" in df.columns:
        df["CreditLinesPerAge"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"].clip(lower=18)

    # Age bucket -- lets tree models split on life-stage without relying
    # purely on a raw numeric split.
    if "age" in df.columns:
        df["AgeBucket"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 120],
            labels=["under_30", "30_45", "45_60", "60_plus"],
        )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline, in the order that matters."""
    df = clean_placeholder_codes(df)
    df = impute_missing(df)
    df = add_engineered_features(df)
    if "AgeBucket" in df.columns:
        df = pd.get_dummies(df, columns=["AgeBucket"], drop_first=True)
    return df


if __name__ == "__main__":
    from data_loader import load_raw

    df = load_raw()
    df_feat = build_features(df)
    print(f"Original columns: {df.shape[1]} -> Engineered columns: {df_feat.shape[1]}")
    print(df_feat.columns.tolist())
