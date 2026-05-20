"""
Train the NSL-KDD intrusion detection model and save all artifacts.

Run from the project root:
    python train.py

Produces these files in artifacts/:
    - preprocessor.pkl      (fitted NSLKDDPreprocessor)
    - model.pkl             (trained XGBoost classifier)
    - label_encoder.pkl     (fitted LabelEncoder for attack_category)
    - metadata.json         (training metadata for reproducibility)

The final model uses the tuned hyperparameters and balanced sample weights
identified in the underlying classification analysis.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.preprocessing import NSLKDDPreprocessor, COLUMN_NAMES


# ----- Configuration -----

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
RANDOM_STATE = 42

# Tuned hyperparameters from RandomizedSearchCV (see project brief).
# These match the final XGB + Weighted (Tuned) model from the analysis.
XGB_PARAMS = {
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 7,
    "n_estimators": 200,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "subsample": 1.0,
    "objective": "multi:softprob",
    "num_class": 5,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Maps raw NSL-KDD attack_type values to the 5 attack categories.
ATTACK_MAPPING = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load a raw NSL-KDD .txt file into a DataFrame with named columns."""
    return pd.read_csv(path, names=COLUMN_NAMES, header=None)


def add_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 5-class 'attack_category' column derived from 'attack_type'."""
    df = df.copy()
    df["attack_category"] = df["attack_type"].map(ATTACK_MAPPING)
    if df["attack_category"].isna().any():
        unmapped = df.loc[df["attack_category"].isna(), "attack_type"].unique()
        raise ValueError(f"Unmapped attack types: {unmapped}")
    return df


def main():
    print("=" * 70)
    print("NSL-KDD Intrusion Detection — Model Training")
    print("=" * 70)

    # ----- Load data -----
    print("\n[1/6] Loading data...")
    train_df = load_data(DATA_DIR / "KDDTrain+.txt")
    test_df = load_data(DATA_DIR / "KDDTest+.txt")
    print(f"  Training rows: {len(train_df):,}")
    print(f"  Test rows:     {len(test_df):,}")

    # ----- Add the 5-class target -----
    print("\n[2/6] Mapping attack types to 5 categories...")
    train_df = add_attack_category(train_df)
    test_df = add_attack_category(test_df)
    print(f"  Class distribution (train):")
    for cat, count in train_df["attack_category"].value_counts().items():
        print(f"    {cat:8s} {count:>7,}")

    # ----- Fit preprocessor and transform features -----
    print("\n[3/6] Fitting preprocessor and transforming features...")
    preprocessor = NSLKDDPreprocessor()
    preprocessor.fit(train_df)
    X_train = preprocessor.transform(train_df)
    X_test = preprocessor.transform(test_df)
    print(f"  Feature columns: {len(preprocessor.feature_columns)}")
    print(f"  X_train shape:   {X_train.shape}")
    print(f"  X_test shape:    {X_test.shape}")

    # ----- Encode labels -----
    print("\n[4/6] Encoding target labels...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["attack_category"])
    y_test = label_encoder.transform(test_df["attack_category"])
    print(f"  Class mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # ----- Train XGBoost with balanced sample weights -----
    print("\n[5/6] Training XGBoost (tuned, balanced sample weights)...")
    sample_weights = compute_sample_weight("balanced", y_train)
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # ----- Evaluate -----
    print("\n[6/6] Evaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    macro_recall = recall_score(y_test, y_pred, average="macro")

    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  Macro F1:      {macro_f1:.4f}")
    print(f"  Macro Recall:  {macro_recall:.4f}")
    print()
    target_names = list(label_encoder.classes_)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    # ----- Save artifacts -----
    print("Saving artifacts...")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
    joblib.dump(model, ARTIFACTS_DIR / "model.pkl")
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")

    metadata = {
        "feature_columns": preprocessor.feature_columns,
        "classes": label_encoder.classes_.tolist(),
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if not callable(v)},
        "test_metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "macro_recall": float(macro_recall),
        },
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/:")
    for path in sorted(ARTIFACTS_DIR.iterdir()):
        print(f"  {path.name}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()