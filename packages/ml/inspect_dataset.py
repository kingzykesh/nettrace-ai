import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("datasets/raw/NF-UNSW-NB15-V2.parquet")
FEATURES_PATH = Path("datasets/raw/NetFlow_v2_Features.xlsx")
OUTPUT_DIR = Path("datasets/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def inspect_dataset():
    print("Loading dataset...")
    df = pd.read_parquet(RAW_DATA_PATH)

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    possible_label_cols = ["label", "Label", "attack_cat", "Attack", "class", "Class"]

    print("\n===== POSSIBLE LABEL COLUMNS =====")
    for col in possible_label_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts())

    summary_path = OUTPUT_DIR / "dataset_summary.csv"
    df.describe(include="all").to_csv(summary_path)

    print(f"\nDataset summary saved to: {summary_path}")

def inspect_features_file():
    if FEATURES_PATH.exists():
        print("\nLoading NetFlow features file...")
        features_df = pd.read_excel(FEATURES_PATH)
        print(features_df.head())
        print(features_df.columns.tolist())
    else:
        print("\nFeatures file not found.")

if __name__ == "__main__":
    inspect_dataset()
    inspect_features_file()