"""Load the raw credit risk data and do basic sanity checks."""
import pandas as pd

TARGET = "SeriousDlqin2yrs"
RAW_PATH = "data/raw/cs-training.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:  # Kaggle's real file has a stray index column
        df = df.drop(columns=["Unnamed: 0"])
    return df


def sanity_report(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Target rate ({TARGET}): {df[TARGET].mean():.2%}")
    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False))
    print("\nSuspicious values (domain knowledge checks):")
    if "age" in df.columns:
        print(f"  age <= 0: {(df['age'] <= 0).sum()}")
    late_cols = [c for c in df.columns if "Past" in c or "Late" in c]
    for c in late_cols:
        extreme = (df[c] > 90).sum()  # Kaggle's real data has a known "96/98" placeholder bug
        if extreme:
            print(f"  {c} > 90 (likely placeholder codes): {extreme}")


if __name__ == "__main__":
    df = load_raw()
    sanity_report(df)
