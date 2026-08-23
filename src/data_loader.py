"""
data_loader.py
==============
Module for loading and generating Hyperspectral Water Quality datasets.

This script handles:
1. Generating a bio-optically realistic hyperspectral water dataset if no dataset is present.
2. Loading raw dataset files into pandas DataFrames.
3. Separating hyperspectral band features (X) from the target parameter (y: Turbidity in NTU).
"""

import os
import numpy as np
import pandas as pd

# Define standard spectral wavelengths: 400nm to 900nm at 10nm increments (51 spectral bands)
WAVELENGTHS = np.arange(400, 901, 10)
BAND_COLS = [f"band_{w}nm" for w in WAVELENGTHS]
TARGET_COL = "turbidity_ntu"


def generate_demo_dataset(file_path: str, n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a physics-based bio-optical hyperspectral dataset for water quality.
    
    Bio-Optical Physics basis:
    - Pure water absorbs light heavily in NIR (700-900nm) and weakly in Blue/Green (400-550nm).
    - Suspended sediments (Turbidity) increase backscattering across all bands, especially Red/NIR.
    - Chlorophyll-a adds an absorption trough around 675nm and a fluorescence/scattering peak around 700nm.
    
    Labels:
    - Clear label: DEMO DATA - BIO-OPTICAL SIMULATION
    """
    np.random.seed(random_seed)
    
    # 1. Generate realistic Turbidity values (0.5 NTU to 45.0 NTU)
    # Log-normal distribution represents natural water body distributions well
    turbidity = np.random.lognormal(mean=1.8, sigma=0.8, size=n_samples)
    turbidity = np.clip(turbidity, 0.5, 50.0)  # Restrict within standard freshwater range (NTU)

    # 2. Generate Chlorophyll-a (1.0 to 80.0 ug/L) for spectral variation
    chl_a = 0.8 * turbidity + np.random.normal(5, 2, size=n_samples)
    chl_a = np.clip(chl_a, 0.5, 100.0)

    # 3. Base absorption spectrum for pure water (a_w) and backscattering (b_bw)
    # Wavelength array
    w = WAVELENGTHS
    
    # Simulating Remote Sensing Reflectance R_rs(lambda)
    # R_rs approx proportional to b_b / (a + b_b)
    
    # Absorption of pure water increases exponentially with wavelength
    a_water = 0.01 * np.exp((w - 400) / 100.0) + 0.02
    
    # Particle absorption (sediment + phytoplankton)
    a_particles = 0.05 * np.exp(-(w - 400) / 150.0)
    
    # Phytoplankton specific absorption peak at 675nm
    a_phytoplankton = 0.02 * np.exp(-((w - 675) ** 2) / (2 * 15**2))
    
    reflectance_data = []

    for i in range(n_samples):
        turb = turbidity[i]
        chla = chl_a[i]
        
        # Particle backscattering b_bp scales linearly with turbidity/suspended sediments
        b_bp = (0.012 * turb) * ((400.0 / w) ** 0.5)
        
        # Phytoplankton reflectance peak near 705nm
        b_phyto = (0.001 * chla) * np.exp(-((w - 705) ** 2) / (2 * 20**2))
        
        # Total absorption and total backscattering
        a_total = a_water + a_particles + (chla * a_phytoplankton * 0.01)
        b_b_total = 0.002 * (400.0 / w)**4 + b_bp + b_phyto  # Water + particulate scattering
        
        # Remote Sensing Reflectance R_rs (sr^-1)
        r_rs = 0.05 * (b_b_total / (a_total + b_b_total))
        
        # Add realistic sensor noise (SNR ~ 100:1)
        sensor_noise = np.random.normal(0, 0.0003, size=len(w))
        r_rs_noisy = np.clip(r_rs + sensor_noise, 0.0001, 0.15)
        
        reflectance_data.append(r_rs_noisy)
        
    # Create DataFrame
    df_bands = pd.DataFrame(reflectance_data, columns=BAND_COLS)
    df_target = pd.DataFrame({
        "sample_id": [f"WATER_SMP_{i+1:04d}" for i in range(n_samples)],
        TARGET_COL: np.round(turbidity, 2),
        "chlorophyll_a_ugL": np.round(chl_a, 2),
        "data_source": "DEMO DATA - BIO-OPTICAL SIMULATION"
    })

    df_full = pd.concat([df_target, df_bands], axis=1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_full.to_csv(file_path, index=False)
    print(f"[data_loader] Demo hyperspectral dataset saved successfully to: {file_path}")
    return df_full


def load_dataset(file_path: str) -> tuple[pd.DataFrame, list[str], str]:
    """
    Loads raw CSV dataset from disk. If missing, generates the bio-optical demo dataset.
    
    Returns:
        df (pd.DataFrame): Complete DataFrame
        band_cols (list[str]): Names of feature columns (X)
        target_col (str): Name of target column (y)
    """
    if not os.path.exists(file_path):
        print(f"[data_loader] File not found at {file_path}. Initializing bio-optical demo dataset...")
        df = generate_demo_dataset(file_path)
    else:
        df = pd.read_csv(file_path)
        print(f"[data_loader] Loaded dataset from {file_path} with shape: {df.shape}")

    # Extract spectral band columns (all columns ending with 'nm' or starting with 'band_')
    band_cols = [c for c in df.columns if c.startswith("band_") or c.endswith("nm")]
    target_col = TARGET_COL if TARGET_COL in df.columns else "turbidity_ntu"

    return df, band_cols, target_col


if __name__ == "__main__":
    # Self-test when run directly
    raw_data_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    df, bands, target = load_dataset(raw_data_path)
    print(f"Dataset Loaded Successfully!")
    print(f"Number of Samples: {len(df)}")
    print(f"Number of Spectral Bands (Features X): {len(bands)}")
    print(f"Target Parameter (y): {target}")
    print("\nSample Preview:")
    print(df[["sample_id", target, bands[0], bands[10], bands[25], bands[50]]].head())
