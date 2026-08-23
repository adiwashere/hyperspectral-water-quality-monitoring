"""
explore_data.py
===============
Exploratory Data Analysis (EDA) for Hyperspectral Water Quality Dataset.

This script performs:
1. Dataset shape and data type inspection.
2. Missing value and duplicate row verification.
3. Summary statistics calculation.
4. Spectral Signature Plotting (Reflectance vs. Wavelength across Turbidity levels).
5. Target Parameter (Turbidity NTU) distribution plotting.
6. Saving visualizations to results/graphs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_dataset, BAND_COLS, TARGET_COL, WAVELENGTHS

# Configure plot aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def run_exploration(raw_data_path: str, output_dir: str):
    """Performs comprehensive exploratory data analysis."""
    print("=" * 60)
    print("      HYPERSPECTRAL WATER QUALITY - EXPLORATORY DATA ANALYSIS      ")
    print("=" * 60)
    
    # 1. Load Data
    df, band_cols, target_col = load_dataset(raw_data_path)
    
    # 2. Dimensions & Basic Info
    print(f"\n1. DATASET DIMENSIONS:")
    print(f"   - Total Rows (Samples): {df.shape[0]}")
    print(f"   - Total Columns: {df.shape[1]}")
    print(f"   - Spectral Bands (Features): {len(band_cols)} ({WAVELENGTHS[0]}nm to {WAVELENGTHS[-1]}nm)")
    print(f"   - Target Variable: {target_col}")

    # 3. Check Data Source & Label
    if "data_source" in df.columns:
        print(f"   - Data Source Tag: {df['data_source'].iloc[0]}")

    # 4. Check Missing Values
    missing_count = df.isnull().sum().sum()
    print(f"\n2. MISSING VALUES CHECK:")
    print(f"   - Total Missing Cells: {missing_count}")
    if missing_count > 0:
        print("   WARNING: Missing values detected! Inspecting column counts:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("   [OK] Clean dataset: Zero missing values.")

    # 5. Check Duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\n3. DUPLICATE ROWS CHECK:")
    print(f"   - Duplicate Rows Count: {duplicate_count}")
    if duplicate_count > 0:
        print("   WARNING: Duplicate rows detected!")
    else:
        print("   [OK] Clean dataset: Zero duplicate rows.")

    # 6. Summary Statistics for Target Parameter
    print(f"\n4. TARGET PARAMETER SUMMARY STATISTICS ({target_col}):")
    target_stats = df[target_col].describe()
    print(f"   - Mean Turbidity:   {target_stats['mean']:.2f} NTU")
    print(f"   - Std Deviation:    {target_stats['std']:.2f} NTU")
    print(f"   - Min Turbidity:    {target_stats['min']:.2f} NTU")
    print(f"   - 25% Quartile:     {target_stats['25%']:.2f} NTU")
    print(f"   - Median (50%):     {target_stats['50%']:.2f} NTU")
    print(f"   - 75% Quartile:     {target_stats['75%']:.2f} NTU")
    print(f"   - Max Turbidity:    {target_stats['max']:.2f} NTU")

    # 7. Visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # FIG 1: Spectral Signatures Curve Plot
    plt.figure(figsize=(12, 6))
    
    # Categorize samples into Turbidity levels for visualization clarity
    df['turbidity_category'] = pd.qcut(df[target_col], q=4, labels=['Low Turbidity', 'Moderate Turbidity', 'High Turbidity', 'Very High Turbidity'])
    
    palette = sns.color_palette("viridis", 4)
    
    for idx, (cat_name, group) in enumerate(df.groupby('turbidity_category', observed=False)):
        # Calculate mean reflectance across bands for this turbidity group
        mean_spectrum = group[band_cols].mean(axis=0).values
        std_spectrum = group[band_cols].std(axis=0).values
        
        plt.plot(WAVELENGTHS, mean_spectrum, label=f"{cat_name} (Mean Turbidity: {group[target_col].mean():.1f} NTU)", linewidth=2.5, color=palette[idx])
        plt.fill_between(WAVELENGTHS, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color=palette[idx], alpha=0.15)

    plt.title("Hyperspectral Water Reflectance Signatures by Turbidity Level", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Remote Sensing Reflectance R_rs (sr^-1)", fontsize=12)
    plt.axvspan(400, 500, color='blue', alpha=0.05, label='Blue Region (400-500nm)')
    plt.axvspan(500, 600, color='green', alpha=0.05, label='Green Region (500-600nm)')
    plt.axvspan(600, 700, color='red', alpha=0.05, label='Red Region (600-700nm)')
    plt.axvspan(700, 900, color='purple', alpha=0.05, label='NIR Region (700-900nm)')
    
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    spectral_fig_path = os.path.join(output_dir, "01_spectral_signatures.png")
    plt.savefig(spectral_fig_path, dpi=300)
    plt.close()
    print(f"\n5. SAVED GRAPH: {spectral_fig_path}")

    # FIG 2: Turbidity Target Distribution Histogram & KDE
    plt.figure(figsize=(10, 5))
    sns.histplot(df[target_col], kde=True, color='#1f77b4', bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(target_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {target_stats['mean']:.2f} NTU")
    plt.axvline(target_stats['50%'], color='green', linestyle='-', linewidth=2, label=f"Median: {target_stats['50%']:.2f} NTU")
    
    plt.title("Target Parameter Distribution: Turbidity (NTU)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Turbidity (NTU)", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()

    dist_fig_path = os.path.join(output_dir, "02_turbidity_distribution.png")
    plt.savefig(dist_fig_path, dpi=300)
    plt.close()
    print(f"   SAVED GRAPH: {dist_fig_path}")
    print("\n[OK] Exploration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    graphs_dir = os.path.join("results", "graphs")
    run_exploration(raw_path, graphs_dir)
