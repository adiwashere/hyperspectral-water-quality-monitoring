"""
predict.py
==========
Production Prediction Pipeline for Hyperspectral Water Quality Monitoring.

Inference Transformation Sequence:
New 51-Band Spectral Vector 
            ↓
  [ scaler.transform() ]   (Loaded from models/scaler.joblib - 51 features)
            ↓
   [ pca.transform() ]     (Loaded from models/pca_transformer.joblib - 51 -> 2 components)
            ↓
  [ model.predict() ]     (Loaded from models/best_model.joblib - 2 components)
            ↓
   Predicted Turbidity (NTU)

CRITICAL: NO training or refitting occurs during inference (.fit() is NEVER called).
Data Source Tag: DEMO DATA - BIO-OPTICAL SIMULATION
"""

import os
import joblib
import numpy as np
import pandas as pd

from data_loader import BAND_COLS


class HyperspectralPredictionPipeline:
    """Standalone production prediction pipeline."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.scaler_path = os.path.join(models_dir, "scaler.joblib")
        self.pca_path = os.path.join(models_dir, "pca_transformer.joblib")
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        
        self.scaler = None
        self.pca = None
        self.model = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Loads serialized scaler, PCA transformer, and ML model from disk."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler asset not found at {self.scaler_path}. Run preprocessing first.")
        if not os.path.exists(self.pca_path):
            raise FileNotFoundError(f"PCA asset not found at {self.pca_path}. Run features/evaluate_pca first.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model asset not found at {self.model_path}. Run training first.")

        self.scaler = joblib.load(self.scaler_path)
        self.pca = joblib.load(self.pca_path)
        self.model = joblib.load(self.model_path)

        print("[predict_pipeline] INFERENCE ASSETS LOADED SUCCESSFULLY:")
        print(f"   1. Scaler:          Expects {self.scaler.n_features_in_} features")
        print(f"   2. PCA Transformer: Transforms {self.pca.n_features_in_} features -> {self.pca.n_components_} components")
        print(f"   3. Best Model:       Expects {self.model.n_features_in_} features ({type(self.model).__name__})")

        # Dimensionality Consistency Validation
        assert self.scaler.n_features_in_ == 51, f"Expected 51 features in scaler, got {self.scaler.n_features_in_}"
        assert self.pca.n_components_ == self.model.n_features_in_, (
            f"Mismatch! PCA produces {self.pca.n_components_} components, but Model expects {self.model.n_features_in_}"
        )

    def predict(self, raw_spectral_input: dict | list | np.ndarray) -> dict:
        """
        Executes prediction on a new 51-band hyperspectral observation.
        
        Args:
            raw_spectral_input: Dictionary with band names, List of 51 numbers, or 2D numpy array.
            
        Returns:
            dict: Structured prediction response.
        """
        # 1. Format input into 2D numpy array (shape: 1, 51)
        if isinstance(raw_spectral_input, dict):
            # Extract band values in exact wavelength order
            spectral_vector = [raw_spectral_input[col] for col in BAND_COLS if col in raw_spectral_input]
            if len(spectral_vector) != 51:
                raise ValueError(f"Dictionary input must contain all 51 spectral bands ({len(BAND_COLS)} expected).")
            X_raw = pd.DataFrame([spectral_vector], columns=BAND_COLS)
        elif isinstance(raw_spectral_input, list):
            if len(raw_spectral_input) != 51:
                raise ValueError(f"Input list must contain exactly 51 spectral values (got {len(raw_spectral_input)}).")
            X_raw = np.array(raw_spectral_input, dtype=np.float64).reshape(1, -1)
        elif isinstance(raw_spectral_input, np.ndarray):
            if raw_spectral_input.ndim == 1:
                X_raw = raw_spectral_input.reshape(1, -1)
            else:
                X_raw = raw_spectral_input
            if X_raw.shape[1] != 51:
                raise ValueError(f"Numpy array must have 51 feature columns (got {X_raw.shape[1]}).")
        else:
            raise TypeError("Unsupported input type. Provide dict, list, or numpy array.")

        # Input Dimension Verification
        assert X_raw.shape[1] == 51, f"Expected 51 raw bands, got {X_raw.shape[1]}"

        # 2. Apply StandardScaler (ONLY transform, NO fit)
        X_scaled = self.scaler.transform(X_raw)

        # 3. Apply PCA Transformer (ONLY transform, NO fit)
        X_pca = self.pca.transform(X_scaled)
        assert X_pca.shape[1] == self.pca.n_components_, f"Expected {self.pca.n_components_} PCs, got {X_pca.shape[1]}"

        # 4. Perform Model Inference (ONLY predict, NO fit)
        predicted_turbidity = self.model.predict(X_pca)
        turb_value = float(round(predicted_turbidity[0], 2))

        # Determine qualitative water turbidity category
        if turb_value < 5.0:
            quality_status = "Clear Water (Low Turbidity)"
        elif turb_value < 15.0:
            quality_status = "Moderate Turbidity"
        elif turb_value < 30.0:
            quality_status = "High Turbidity"
        else:
            quality_status = "Very High Turbidity (Cloudy / Sedimentary)"

        # Construct response
        response = {
            "predicted_turbidity": turb_value,
            "unit": "NTU",
            "quality_status": quality_status,
            "input_dimensions": {
                "raw_bands": X_raw.shape[1],
                "pca_components": X_pca.shape[1]
            },
            "model_architecture": f"{type(self.model).__name__} (PCA {self.pca.n_components_} Components)",
            "data_source": "DEMO DATA - BIO-OPTICAL SIMULATION"
        }
        return response


def run_test_prediction():
    """Executes verification test on a sample observation."""
    print("=" * 60)
    print("        VERIFYING STANDALONE INFERENCE PREDICTION PIPELINE       ")
    print("=" * 60)
    
    # 1. Instantiate Pipeline
    pipeline = HyperspectralPredictionPipeline()

    # 2. Load a sample from raw dataset for testing
    raw_csv_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    df = pd.read_csv(raw_csv_path)
    
    sample_row = df.iloc[10]
    sample_id = sample_row["sample_id"]
    true_turbidity = sample_row["turbidity_ntu"]
    
    # Extract 51 spectral bands as a dictionary input
    sample_dict = {col: sample_row[col] for col in BAND_COLS}

    print(f"\n[test] Testing Sample: {sample_id}")
    print(f"       Ground Truth Turbidity: {true_turbidity} NTU")

    # 3. Execute Inference
    result = pipeline.predict(sample_dict)

    print("\n[test] INFERENCE RESPONSE OUTPUT:")
    for k, v in result.items():
        print(f"   - {k:<22}: {v}")

    # 4. Verify Dimension Integrity
    assert result["input_dimensions"]["raw_bands"] == 51, "Raw bands dimension check failed!"
    assert result["input_dimensions"]["pca_components"] == 2, "PCA components dimension check failed!"
    
    error = abs(result["predicted_turbidity"] - true_turbidity)
    print(f"\n[test] Absolute Prediction Difference: {error:.4f} NTU")
    print("[OK] INFERENCE PIPELINE VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_test_prediction()
