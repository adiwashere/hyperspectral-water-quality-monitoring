"""
backend/main.py
===============
FastAPI Web Server for Hyperspectral Water-Quality Monitoring.

Endpoints:
- GET  /             : API Root welcome & documentation link
- GET  /health       : System health status check
- GET  /model-info   : Active ML model metadata & feature dimensions
- GET  /sample-input : Ready-to-use sample 51-band JSON payload
- POST /predict      : Validate 51 hyperspectral bands & return Turbidity prediction

Data Source Tag: DEMO DATA - BIO-OPTICAL SIMULATION
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Add src directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from predict import HyperspectralPredictionPipeline
from camera_simulator import HyperspectralCameraSimulator
from data_loader import BAND_COLS


# 1. Initialize FastAPI Application
app = FastAPI(
    title="Hyperspectral Water-Quality Monitoring API",
    description="Machine Learning REST API predicting Turbidity (NTU) from continuous hyperspectral reflectance signatures.",
    version="1.0.0"
)

# 2. Configure CORS Middleware for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from React/Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Prediction Pipeline & Camera Simulator Assets at Server Startup
try:
    models_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    pipeline = HyperspectralPredictionPipeline(models_dir=models_dir_path)
    simulator = HyperspectralCameraSimulator(models_dir=models_dir_path)
except Exception as e:
    print(f"[ERROR] Failed to load ML pipeline assets: {e}")
    pipeline = None
    simulator = None


# 4. Pydantic Request Validation Model
class PredictRequest(BaseModel):
    bands: Dict[str, float] | List[float]

    @field_validator("bands")
    @classmethod
    def validate_spectral_bands(cls, v: Any) -> Any:
        if isinstance(v, list):
            if len(v) != 51:
                raise ValueError(f"Input list must contain exactly 51 spectral values (received {len(v)}).")
            for idx, val in enumerate(v):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValueError(f"Non-numeric value detected at index {idx}: '{val}'. All bands must be numeric floats.")
        elif isinstance(v, dict):
            if len(v) != 51:
                raise ValueError(f"Input dictionary must contain exactly 51 spectral bands (received {len(v)}).")
            missing = [col for col in BAND_COLS if col not in v]
            if missing:
                raise ValueError(f"Missing required band keys: {missing[:3]}... Total missing: {len(missing)}")
            for k, val in v.items():
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValueError(f"Non-numeric value detected for band '{k}': '{val}'.")
        else:
            raise ValueError("Input 'bands' must be either a Dictionary of 51 named bands or a List of 51 numbers.")
        return v


# 5. API Route Handlers

@app.get("/", tags=["General"])
def root():
    """API Root endpoint."""
    return {
        "message": "Welcome to the Hyperspectral Water-Quality Monitoring API!",
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "data_source": "DEMO DATA - BIO-OPTICAL SIMULATION"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Returns server health and pipeline loaded status."""
    is_ready = pipeline is not None
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "pipeline_loaded": is_ready,
        "service": "Hyperspectral Water Quality Backend"
    }


@app.get("/model-info", tags=["Metadata"])
def get_model_info():
    """Returns metadata about the active ML model architecture and feature requirements."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction pipeline is not loaded.")
    
    return {
        "target_parameter": "Turbidity",
        "unit": "NTU",
        "raw_input_bands": pipeline.scaler.n_features_in_,
        "wavelength_range": f"{BAND_COLS[0]} to {BAND_COLS[-1]} (10nm intervals)",
        "pca_components": pipeline.pca.n_components_,
        "retained_variance": f"{sum(pipeline.pca.explained_variance_ratio_) * 100:.2f}%",
        "model_architecture": type(pipeline.model).__name__,
        "data_source": "DEMO DATA - BIO-OPTICAL SIMULATION",
        "is_demo_data": True
    }


@app.get("/sample-input", tags=["Utility"])
def get_sample_input():
    """Returns a valid 51-band sample JSON payload for testing POST /predict."""
    raw_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "water_quality_hyperspectral_data.csv"))
    if os.path.exists(raw_csv_path):
        df = pd.read_csv(raw_csv_path)
        sample_row = df.iloc[0]
        sample_dict = {col: float(sample_row[col]) for col in BAND_COLS}
        sample_id = sample_row["sample_id"]
        ground_truth = float(sample_row["turbidity_ntu"])
    else:
        # Fallback dictionary if file is absent
        sample_dict = {f"band_{w}nm": 0.005 for w in range(400, 901, 10)}
        sample_id = "SAMPLE_FALLBACK"
        ground_truth = 4.2

    return {
        "sample_id": sample_id,
        "ground_truth_turbidity_ntu": ground_truth,
        "payload": {
            "bands": sample_dict
        }
    }


@app.post("/predict", tags=["Inference"])
def predict_water_quality(request: PredictRequest):
    """
    Accepts 51 hyperspectral reflectance bands and returns predicted Turbidity (NTU).
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction pipeline is not loaded on server."
        )

    try:
        # Execute inference pipeline (scaler -> PCA -> model)
        prediction_result = pipeline.predict(request.bands)
        return prediction_result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/stream/next-frame", tags=["Stream Simulation"])
def get_next_stream_frame():
    """
    Retrieves the next simulated 51-band hyperspectral camera frame and prediction.
    
    Data Source Tag: SIMULATION - HARDWARE CAMERA STREAM EMULATOR
    """
    if simulator is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Camera simulator is not loaded on server."
        )

    try:
        frame_payload = simulator.get_next_frame()
        return frame_payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream generation error: {str(e)}"
        )

