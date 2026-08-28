"""
train.py
========
Machine Learning Model Training Pipeline for Turbidity Prediction.

This module trains 4 regression models:
1. Linear Regression (Baseline)
2. Support Vector Regression (SVR)
3. Random Forest Regressor
4. XGBoost Regressor

Records actual measured training time for each model and saves serialized models to models/.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from preprocessing import prepare_data


def get_models() -> dict:
    """Returns a dictionary of candidate regression models to evaluate."""
    return {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    }


def train_models(X_train: np.ndarray, y_train: pd.Series, models_dir: str = "models") -> tuple[dict, dict]:
    """
    Trains candidate ML models on scaled feature matrix X_train and target y_train.
    
    Returns:
        trained_models (dict): Dictionary mapping model names to fitted model instances.
        training_times (dict): Dictionary mapping model names to measured training time in seconds.
    """
    os.makedirs(models_dir, exist_ok=True)
    models = get_models()
    trained_models = {}
    training_times = {}
    
    print("\n" + "=" * 60)
    print("            STARTING MACHINE LEARNING MODEL TRAINING            ")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n[train] Training {name}...")
        start_time = time.perf_counter()
        
        # Fit model on training set
        model.fit(X_train, y_train)
        
        elapsed_time = time.perf_counter() - start_time
        trained_models[name] = model
        training_times[name] = round(elapsed_time, 4)
        
        # Save individual trained model object
        file_name = name.lower().replace(" ", "_") + ".joblib"
        model_save_path = os.path.join(models_dir, file_name)
        joblib.dump(model, model_save_path)
        
        print(f"   [OK] Completed in {elapsed_time:.4f} seconds.")
        print(f"   Saved model to: {model_save_path}")
        
    print("\n" + "=" * 60)
    print("             ALL MODELS TRAINED SUCCESSFULLY!             ")
    print("=" * 60)
    return trained_models, training_times


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "water_quality_hyperspectral_data.csv")
    X_tr, X_te, y_tr, y_te, bands, sc = prepare_data(raw_path)
    trained, times = train_models(X_tr, y_tr)
