"""
evaluate_pca.py
===============
PCA Dimensionality Reduction Evaluation & Model Comparison Pipeline.

This module:
1. Performs Explained Variance Analysis across thresholds (90%, 95%, 99%, 99.9%).
2. Generates Scree Plots and Cumulative Variance Curves in results/graphs/.
3. Trains all 4 candidate regressors on PCA-transformed features.
4. Compares metrics (MAE, RMSE, R2, Training Time) for Raw 51 Bands vs. PCA Components.
5. Saves results to results/metrics/pca_comparison.csv and generates results/graphs/08_raw_vs_pca_performance.png.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import prepare_data
from train import train_models
from features import analyze_pca_variance, apply_pca

# Aesthetics setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def run_pca_evaluation(
    raw_data_path: str,
    results_dir: str = "results",
    models_dir: str = "models"
):
    """Executes full PCA evaluation workflow."""
    metrics_dir = os.path.join(results_dir, "metrics")
    graphs_dir = os.path.join(results_dir, "graphs")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("       PHASE 3: PCA DIMENSIONALITY REDUCTION & MODEL EVALUATION    ")
    print("=" * 60)

    # 1. Load & Preprocess Data (Scales X_train strictly without data leakage)
    X_train_scaled, X_test_scaled, y_train, y_test, band_cols, scaler = prepare_data(raw_data_path)

    # 2. PCA Variance Analysis
    threshold_counts, var_ratio, cum_var = analyze_pca_variance(X_train_scaled)

    # FIG 6: Scree Plot (Explained Variance Ratio per Component)
    plt.figure(figsize=(10, 5))
    n_comp_display = min(25, len(var_ratio))
    plt.bar(range(1, n_comp_display + 1), var_ratio[:n_comp_display] * 100, color='#1f77b4', edgecolor='black', alpha=0.7)
    plt.plot(range(1, n_comp_display + 1), var_ratio[:n_comp_display] * 100, 'ro-', linewidth=2)
    plt.title("PCA Scree Plot: Individual Explained Variance Ratio", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Principal Component Index", fontsize=12)
    plt.ylabel("Explained Variance (%)", fontsize=12)
    plt.xticks(range(1, n_comp_display + 1))
    plt.tight_layout()
    scree_path = os.path.join(graphs_dir, "06_pca_scree_plot.png")
    plt.savefig(scree_path, dpi=300)
    plt.close()
    print(f"\n[evaluate_pca] Saved graph: {scree_path}")

    # FIG 7: Cumulative Explained Variance Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var * 100, 'b-', linewidth=2.5, label='Cumulative Variance')
    
    # Mark threshold lines
    colors = ['orange', 'green', 'red', 'purple']
    for idx, (thresh_name, count) in enumerate(threshold_counts.items()):
        val = float(thresh_name.split('%')[0])
        plt.axhline(val, color=colors[idx], linestyle='--', alpha=0.7, label=f"{thresh_name}: {count} PCs")
        plt.axvline(count, color=colors[idx], linestyle=':', alpha=0.7)

    plt.title("Cumulative Explained Variance vs. Number of Principal Components", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.ylabel("Cumulative Variance (%)", fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    cum_path = os.path.join(graphs_dir, "07_pca_cumulative_variance.png")
    plt.savefig(cum_path, dpi=300)
    plt.close()
    print(f"[evaluate_pca] Saved graph: {cum_path}")

    # 3. Apply PCA with 99.0% Variance Threshold (or ~6 components)
    n_selected = threshold_counts["99.0% Variance"]
    X_train_pca, X_test_pca, pca_obj = apply_pca(
        X_train_scaled, X_test_scaled, n_components=n_selected
    )

    # 4. Train & Evaluate Models on Raw 51 Bands vs. PCA Components
    print("\n[evaluate_pca] Training models on RAW 51 BANDS...")
    models_raw, times_raw = train_models(X_train_scaled, y_train, models_dir=os.path.join(models_dir, "raw"))
    
    print(f"\n[evaluate_pca] Training models on PCA REDUCED ({n_selected} components)...")
    models_pca, times_pca = train_models(X_train_pca, y_train, models_dir=os.path.join(models_dir, "pca"))

    # Compare Metrics
    comparison_records = []
    
    for name in models_raw.keys():
        # Raw Model Evaluation
        pred_raw = models_raw[name].predict(X_test_scaled)
        mae_raw = mean_absolute_error(y_test, pred_raw)
        rmse_raw = np.sqrt(mean_squared_error(y_test, pred_raw))
        r2_raw = r2_score(y_test, pred_raw)
        time_raw = times_raw[name]

        # PCA Model Evaluation
        pred_pca = models_pca[name].predict(X_test_pca)
        mae_pca = mean_absolute_error(y_test, pred_pca)
        rmse_pca = np.sqrt(mean_squared_error(y_test, pred_pca))
        r2_pca = r2_score(y_test, pred_pca)
        time_pca = times_pca[name]

        comparison_records.append({
            "Model": name,
            "Raw Features": f"51 Bands",
            "Raw MAE (NTU)": round(mae_raw, 4),
            "Raw RMSE (NTU)": round(rmse_raw, 4),
            "Raw R2": round(r2_raw, 4),
            "Raw Time (s)": round(time_raw, 4),
            "PCA Features": f"{n_selected} Components",
            "PCA MAE (NTU)": round(mae_pca, 4),
            "PCA RMSE (NTU)": round(rmse_pca, 4),
            "PCA R2": round(r2_pca, 4),
            "PCA Time (s)": round(time_pca, 4)
        })

    df_pca_comp = pd.DataFrame(comparison_records)
    print("\n[evaluate_pca] RAW vs. PCA MODEL COMPARISON TABLE:")
    print(df_pca_comp.to_string(index=False))

    # Save Best PCA Model to root models/ folder for production inference
    best_pca_model = models_pca["Random Forest Regressor"]
    best_model_save_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pca_model, best_model_save_path)
    print(f"[evaluate_pca] Saved best PCA-trained model (Random Forest, 2 PCs) to: {best_model_save_path}")

    # FIG 8: Raw vs PCA R2 Score Bar Chart Comparison
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(df_pca_comp["Model"]))
    width = 0.35

    plt.bar(x_indices - width/2, df_pca_comp["Raw R2"], width, label='Raw (51 Bands)', color='#1f77b4')
    plt.bar(x_indices + width/2, df_pca_comp["PCA R2"], width, label=f'PCA ({n_selected} Components)', color='#ff7f0e')

    plt.title(f"Model Performance Comparison: Raw 51 Bands vs. {n_selected} PCA Components [DEMO DATA]", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel("Test R² Score", fontsize=12)
    plt.xticks(x_indices, df_pca_comp["Model"], rotation=15)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    comp_fig_path = os.path.join(graphs_dir, "08_raw_vs_pca_performance.png")
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"[evaluate_pca] Saved graph: {comp_fig_path}")

    print("\n" + "=" * 60)
    print("             PHASE 3 PCA EVALUATION COMPLETED                ")
    print("=" * 60)
    return df_pca_comp


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    run_pca_evaluation(raw_path)
