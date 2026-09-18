"""
camera_simulator.py
===================
Edge Hyperspectral Camera Stream Simulator.

Simulates continuous 51-band hyperspectral frame acquisitions.
Reuses the EXISTING inference pipeline (scaler.transform -> pca.transform -> best_model.predict).

Data Source Tag: SIMULATION - HARDWARE CAMERA STREAM EMULATOR
CRITICAL: NO model refitting or training occurs during streaming.
"""

import os
import time
import math
import numpy as np
from datetime import datetime

from data_loader import BAND_COLS, WAVELENGTHS
from predict import HyperspectralPredictionPipeline


class HyperspectralCameraSimulator:
    """Simulates continuous hardware camera stream acquisitions."""

    def __init__(self, models_dir: str = "models"):
        # Reuse existing prediction pipeline
        self.pipeline = HyperspectralPredictionPipeline(models_dir=models_dir)
        self.frame_index = 0
        self.start_time = time.time()

    def generate_simulated_frame(self) -> tuple[dict, str]:
        """Generates realistic 51-band hyperspectral reflectance signature with time-varying sediment flow."""
        self.frame_index += 1
        elapsed = time.time() - self.start_time
        
        # Simulate natural water body turbidity fluctuations over time (sinusoidal water flow drift)
        # Turbidity fluctuates between ~ 3.5 NTU (clear) and ~ 22.0 NTU (turbid)
        base_turbidity = 12.0 + 8.5 * math.sin(elapsed / 15.0) + 3.0 * math.cos(elapsed / 7.0)
        base_turbidity = max(1.0, min(45.0, base_turbidity))
        
        # Calculate bio-optical reflectance across 51 bands
        w = WAVELENGTHS
        b_bp = (0.012 * base_turbidity) * ((400.0 / w) ** 0.5)
        a_total = 0.01 * np.exp((w - 400) / 100.0) + 0.02 + 0.05 * np.exp(-(w - 400) / 150.0)
        b_b_total = 0.002 * (400.0 / w)**4 + b_bp
        
        r_rs = 0.05 * (b_b_total / (a_total + b_b_total))
        noise = np.random.normal(0, 0.0002, size=len(w))
        r_rs_noisy = np.clip(r_rs + noise, 0.0001, 0.15)
        
        # Format 51 bands dictionary
        bands_dict = {col: float(np.round(r_rs_noisy[idx], 6)) for idx, col in enumerate(BAND_COLS)}
        frame_id = f"FRAME_{self.frame_index:05d}"
        
        return bands_dict, frame_id

    def get_next_frame(self) -> dict:
        """
        Retrieves next camera frame and passes it through the existing inference pipeline.
        
        NO fitting or retraining occurs.
        """
        bands_dict, frame_id = self.generate_simulated_frame()
        
        # Pass raw 51-band frame into existing prediction pipeline
        # Pipeline sequence: scaler.transform() -> pca.transform() -> best_model.predict()
        prediction = self.pipeline.predict(bands_dict)
        
        # Attach stream metadata
        stream_payload = {
            "frame_id": frame_id,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "iso_timestamp": datetime.now().isoformat(),
            "bands": bands_dict,
            "prediction": prediction,
            "data_source": "SIMULATION - HARDWARE CAMERA STREAM EMULATOR",
            "is_simulation": True
        }
        return stream_payload


def test_simulator():
    print("=" * 60)
    print("      VERIFYING HYPERSPECTRAL CAMERA STREAM SIMULATOR        ")
    print("=" * 60)
    
    simulator = HyperspectralCameraSimulator()
    print("\n[simulator] Generating 3 continuous frames (2s interval simulation)...")
    
    for i in range(3):
        frame = simulator.get_next_frame()
        print(f"\n--- Frame #{i+1} [{frame['frame_id']} at {frame['timestamp']}] ---")
        print(f"   - Predicted Turbidity: {frame['prediction']['predicted_turbidity']} NTU")
        print(f"   - Quality Status:      {frame['prediction']['quality_status']}")
        print(f"   - Raw Bands Count:     {frame['prediction']['input_dimensions']['raw_bands']}")
        print(f"   - Data Source Tag:     {frame['data_source']}")
        time.sleep(0.5)

    print("\n[OK] CAMERA SIMULATOR VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_simulator()
