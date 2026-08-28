"""
features.py
===========
Feature Extraction & Principal Component Analysis (PCA) Module.

This module handles:
1. Cumulative Explained Variance analysis across 90%, 95%, 99%, and 99.9% thresholds.
2. Fitting PCA STRICTLY on X_train_scaled to prevent Data Leakage.
3. Transforming X_train_scaled and X_test_scaled into PCA component space.
4. Serializing the fitted PCA transformer to models/pca_transformer.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def analyze_pca_variance(X_train_scaled: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Fits full PCA on X_train_scaled and evaluates component counts required for variance thresholds.
    """
    pca_full = PCA(n_components=X_train_scaled.shape[1])
    pca_full.fit(X_train_scaled)
    
    var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    thresholds = [0.90, 0.95, 0.99, 0.999]
    threshold_counts = {}
    
    for thresh in thresholds:
        # Find index where cumulative variance reaches or exceeds threshold
        n_comp = np.argmax(cum_var >= thresh) + 1
        threshold_counts[f"{thresh*100:.1f}% Variance"] = int(n_comp)
        
    print("\n[features] PCA EXPLAINED VARIANCE ANALYSIS (Fitted strictly on X_train):")
    for thresh_str, count in threshold_counts.items():
        actual_var = cum_var[count - 1] * 100
        print(f"   - {thresh_str:<15}: {count:>2} components (Actual Variance: {actual_var:.2f}%)")
        
    return threshold_counts, var_ratio, cum_var


def apply_pca(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    n_components: int = None,
    variance_threshold: float = 0.99,
    pca_save_path: str = os.path.join("models", "pca_transformer.joblib")
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Applies PCA dimensionality reduction.
    
    CRITICAL: Fits PCA ONLY on X_train_scaled, then transforms both X_train and X_test.
    """
    if n_components is None:
        # Determine components based on threshold
        pca_temp = PCA()
        pca_temp.fit(X_train_scaled)
        cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
        
    print(f"\n[features] Fitting PCA with {n_components} components on X_train_scaled...")
    pca = PCA(n_components=n_components)
    
    # 1. Fit ONLY on training data and transform
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # 2. Transform test data using fitted parameters
    X_test_pca = pca.transform(X_test_scaled)
    
    total_var = sum(pca.explained_variance_ratio_) * 100
    print(f"[features] Transformed feature space: {X_train_scaled.shape[1]} bands -> {n_components} components")
    print(f"[features] Total Retained Variance: {total_var:.2f}%")
    
    # Save PCA transformer
    os.makedirs(os.path.dirname(pca_save_path), exist_ok=True)
    joblib.dump(pca, pca_save_path)
    print(f"[features] Saved PCA transformer to: {pca_save_path}")
    
    return X_train_pca, X_test_pca, pca


if __name__ == "__main__":
    from preprocessing import prepare_data
    raw_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    X_tr, X_te, y_tr, y_te, bands, sc = prepare_data(raw_path)
    counts, var_r, cum_v = analyze_pca_variance(X_tr)
    X_tr_pca, X_te_pca, pca_obj = apply_pca(X_tr, X_te, variance_threshold=0.99)
    print(f"[OK] Features PCA module test passed! Output X_train_pca shape: {X_tr_pca.shape}")
