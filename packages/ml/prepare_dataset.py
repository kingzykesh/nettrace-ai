import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("datasets/raw/NF-UNSW-NB15-V2.parquet")
OUTPUT_DIR = Path("datasets/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicate rows if any appear later
    df = df.drop_duplicates()

    # Replace infinite values with NaN, then fill them
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill numeric NaN values with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill object/string NaN values with Unknown
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")

    return df

def prepare_binary_dataset(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col not in ["Label", "Attack"]]

    X = df[feature_cols]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train.to_parquet(OUTPUT_DIR / "X_train_binary.parquet", index=False)
    X_test.to_parquet(OUTPUT_DIR / "X_test_binary.parquet", index=False)
    y_train.to_frame("Label").to_parquet(OUTPUT_DIR / "y_train_binary.parquet", index=False)
    y_test.to_frame("Label").to_parquet(OUTPUT_DIR / "y_test_binary.parquet", index=False)

    print("\nBinary dataset prepared successfully.")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

def prepare_multiclass_dataset(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col not in ["Label", "Attack"]]

    X = df[feature_cols]
    y = df["Attack"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train.to_parquet(OUTPUT_DIR / "X_train_multiclass.parquet", index=False)
    X_test.to_parquet(OUTPUT_DIR / "X_test_multiclass.parquet", index=False)
    y_train.to_frame("Attack").to_parquet(OUTPUT_DIR / "y_train_multiclass.parquet", index=False)
    y_test.to_frame("Attack").to_parquet(OUTPUT_DIR / "y_test_multiclass.parquet", index=False)

    print("\nMulticlass dataset prepared successfully.")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

def save_metadata(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col not in ["Label", "Attack"]]

    metadata = {
        "total_rows": [len(df)],
        "total_columns": [len(df.columns)],
        "feature_count": [len(feature_cols)],
        "target_columns": ["Label, Attack"],
        "dataset_type": ["NF-UNSW-NB15-V2 NetFlow IDS Dataset"]
    }

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(OUTPUT_DIR / "dataset_metadata.csv", index=False)

    pd.Series(feature_cols).to_csv(
        OUTPUT_DIR / "feature_columns.csv",
        index=False,
        header=["feature_name"]
    )

    df["Label"].value_counts().to_csv(OUTPUT_DIR / "binary_class_distribution.csv")
    df["Attack"].value_counts().to_csv(OUTPUT_DIR / "attack_class_distribution.csv")

    print("\nMetadata files saved successfully.")

def main():
    print("Loading raw dataset...")
    df = pd.read_parquet(RAW_DATA_PATH)

    print("Cleaning dataset...")
    df = clean_dataset(df)

    print("Saving cleaned full dataset...")
    df.to_parquet(OUTPUT_DIR / "cleaned_nf_unsw_nb15_v2.parquet", index=False)

    save_metadata(df)
    prepare_binary_dataset(df)
    prepare_multiclass_dataset(df)

    print("\nPhase 1 dataset preparation completed.")

if __name__ == "__main__":
    main()