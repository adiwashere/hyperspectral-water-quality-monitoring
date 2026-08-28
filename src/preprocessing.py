"""
preprocessing.py
================
Preprocessing Pipeline for Hyperspectral Water Quality Data.

This module handles:
1. Feature matrix (X) and Target vector (y) extraction.
2. Train/Test Split (80% Train, 20% Test).
3. Standard Feature Scaling (StandardScaler) fitted STRICTLY on X_train to prevent Data Leakage.
4. Serializing the fitted scaler to models/scaler.joblib for production inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset, TARGET_COL


def prepare_data(
    raw_data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_save_path: str = os.path.join("models", "scaler.joblib")
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str], StandardScaler]:
    """
    Prepares raw dataset for machine learning model training.
    
    CRITICAL: Fits StandardScaler ONLY on X_train to prevent Data Leakage.
    """
    print("\n[preprocessing] Loading raw dataset...")
    df, band_cols, target_col = load_dataset(raw_data_path)
    
    # 1. Separate Features (X) and Target (y)
    X = df[band_cols]
    y = df[target_col]
    
    print(f"[preprocessing] Extracting Features X ({X.shape[1]} bands) and Target y ({target_col})")
    print(f"[preprocessing] Data Source Tag: {df['data_source'].iloc[0] if 'data_source' in df.columns else 'Unknown'}")
    
    # 2. Perform Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[preprocessing] Train Set Size: {len(X_train)} samples | Test Set Size: {len(X_test)} samples")
    
    # 3. Fit StandardScaler strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Transform X_test using parameters derived from X_train
    X_test_scaled = scaler.transform(X_test)
    print("[preprocessing] Standard Scaling applied (mean=0, std=1) without data leakage.")
    
    # 5. Save fitted scaler for downstream prediction pipeline
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"[preprocessing] Fitted scaler saved to: {scaler_save_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, band_cols, scaler


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    X_tr, X_te, y_tr, y_te, bands, sc = prepare_data(raw_path)
    print(f"[OK] Preprocessing self-test completed! X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
