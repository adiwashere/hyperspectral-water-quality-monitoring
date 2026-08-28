"""
evaluate.py
===========
Model Evaluation and Visual Comparison Pipeline.

This module:
1. Evaluates all candidate models on unseen test data (X_test, y_test).
2. Calculates empirical MAE, RMSE, R2, and training duration.
3. Generates comparative evaluation charts in results/graphs/:
   - 03_actual_vs_predicted.png
   - 04_residual_error_distribution.png
   - 05_feature_importance.png
4. Identifies and saves the best performing model to models/best_model.joblib.
5. Saves results matrix to results/metrics/model_comparison.csv.
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

# Aesthetics setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def evaluate_models(
    trained_models: dict,
    training_times: dict,
    X_test: np.ndarray,
    y_test: pd.Series,
    band_cols: list[str],
    results_dir: str = "results",
    models_dir: str = "models"
) -> tuple[pd.DataFrame, str]:
    """
    Evaluates trained models on unseen X_test/y_test and generates comparative graphs.
    """
    metrics_dir = os.path.join(results_dir, "metrics")
    graphs_dir = os.path.join(results_dir, "graphs")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    results_list = []
    predictions_dict = {}

    print("\n" + "=" * 60)
    print("               EVALUATING MODELS ON UNSEEN TEST DATA            ")
    print("=" * 60)

    for name, model in trained_models.items():
        # Predict on unseen test set
        y_pred = model.predict(X_test)
        predictions_dict[name] = y_pred

        # Compute empirical metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        train_time = training_times.get(name, 0.0)

        results_list.append({
            "Model": name,
            "MAE (NTU)": round(mae, 4),
            "RMSE (NTU)": round(rmse, 4),
            "R2 Score": round(r2, 4),
            "Training Time (s)": round(train_time, 4)
        })

    # Create Comparison DataFrame
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)

    print("\n[evaluate] MODEL COMPARISON MATRIX:")
    print(df_results.to_string(index=False))

    # Save Comparison CSV
    metrics_path = os.path.join(metrics_dir, "model_comparison.csv")
    df_results.to_csv(metrics_path, index=False)
    print(f"\n[evaluate] Saved comparison metrics table to: {metrics_path}")

    # Select Best Model based on highest R2 Score
    best_model_row = df_results.iloc[0]
    best_model_name = best_model_row["Model"]
    best_model_obj = trained_models[best_model_name]
    
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    print(f"\n[evaluate] BEST MODEL SELECTED: {best_model_name}")
    print(f"           - Test R2 Score: {best_model_row['R2 Score']}")
    print(f"           - Test MAE:      {best_model_row['MAE (NTU)']} NTU")
    print(f"           - Test RMSE:     {best_model_row['RMSE (NTU)']} NTU")
    print(f"           Saved best model to: {best_model_path}")

    # FIG 3: Actual vs Predicted Parity Plots (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = predictions_dict[name]
        ax = axes[i]

        ax.scatter(y_test, y_pred, alpha=0.7, color=colors[i], edgecolors='k', s=40, label='Test Predictions')
        
        # Perfect prediction parity line (y = x)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Parity (y=x)')

        r2_val = df_results.loc[df_results["Model"] == name, "R2 Score"].values[0]
        mae_val = df_results.loc[df_results["Model"] == name, "MAE (NTU)"].values[0]

        ax.set_title(f"{name}\n(R² = {r2_val:.4f} | MAE = {mae_val:.2f} NTU)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual Turbidity (NTU)", fontsize=11)
        ax.set_ylabel("Predicted Turbidity (NTU)", fontsize=11)
        ax.legend(loc='upper left', frameon=True)

    plt.suptitle("Actual vs. Predicted Turbidity across Candidates [DEMO DATA - BIO-OPTICAL SIMULATION]", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    parity_path = os.path.join(graphs_dir, "03_actual_vs_predicted.png")
    plt.savefig(parity_path, dpi=300)
    plt.close()
    print(f"[evaluate] Saved graph: {parity_path}")

    # FIG 4: Residual Error Distribution Plots (Histograms)
    plt.figure(figsize=(12, 6))
    for i, (name, model) in enumerate(trained_models.items()):
        residuals = y_test - predictions_dict[name]
        sns.kdeplot(residuals, label=f"{name}", linewidth=2)

    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.title("Residual Error Distribution (Actual - Predicted Turbidity)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Residual Error (NTU)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    residual_path = os.path.join(graphs_dir, "04_residual_error_distribution.png")
    plt.savefig(residual_path, dpi=300)
    plt.close()
    print(f"[evaluate] Saved graph: {residual_path}")

    # FIG 5: Feature Importance Plot for Tree-based models
    if "Random Forest Regressor" in trained_models:
        rf_model = trained_models["Random Forest Regressor"]
        importances = rf_model.feature_importances_
        
        # Wavelengths
        w_vals = [int(col.replace("band_", "").replace("nm", "")) for col in band_cols]
        
        plt.figure(figsize=(12, 5))
        plt.bar(w_vals, importances, width=8, color='#2ca02c', edgecolor='black', alpha=0.8)
        plt.title("Hyperspectral Band Feature Importance (Random Forest)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Wavelength (nm)", fontsize=12)
        plt.ylabel("Gini Feature Importance", fontsize=12)
        plt.axvspan(650, 750, color='red', alpha=0.1, label='Red/NIR Turbidity Peak Sensitivity')
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        
        importance_path = os.path.join(graphs_dir, "05_feature_importance.png")
        plt.savefig(importance_path, dpi=300)
        plt.close()
        print(f"[evaluate] Saved graph: {importance_path}")

    print("\n" + "=" * 60)
    print("               EVALUATION COMPLETE SUCCESSFULLY                 ")
    print("=" * 60)
    return df_results, best_model_name


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    X_tr, X_te, y_tr, y_te, bands, sc = prepare_data(raw_path)
    models, times = train_models(X_tr, y_tr)
    results_df, best_name = evaluate_models(models, times, X_te, y_te, bands)
