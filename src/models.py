#!/usr/bin/env python3
"""
Machine Learning model operations for California Housing Price Prediction API
Model loading, inference utilities, and prediction calculations
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd

# Setup logger
logger = logging.getLogger(__name__)

# Global model variable
model = None
model_name = "DecisionTree"  # Using the best performing model


def calculate_confidence_score(model, feature_array):
    """Calculate confidence score for prediction"""
    try:
        if hasattr(model, "predict_proba"):
            # For probabilistic models
            probas = model.predict_proba(feature_array)
            return float(max(probas[0]))
        elif hasattr(model, "decision_function"):
            # For models with decision function
            decision = abs(model.decision_function(feature_array)[0])
            return float(min(decision / 10.0, 1.0))  # Normalize to 0-1
        else:
            # For tree-based models, use path similarity as proxy
            if hasattr(model, "decision_path"):
                path = model.decision_path(feature_array)
                depth = path.toarray().sum()
                max_depth = model.tree_.max_depth
                return float(depth / max_depth) if max_depth > 0 else 0.8
            else:
                return 0.75  # Default confidence for simple models
    except Exception:
        return 0.6  # Fallback confidence


def calculate_prediction_range(prediction, confidence):
    """Calculate prediction range based on confidence"""
    # Use confidence to determine range width
    range_factor = (1.0 - confidence) * 0.3 + 0.1  # 10-40% range
    range_amount = prediction * range_factor

    return {
        "min": float(max(0, prediction - range_amount)),
        "max": float(prediction + range_amount),
        "range_percent": f"{range_factor * 100:.1f}%",
    }


def load_model():
    """Load the trained model"""
    global model
    # Get the directory of this script and build absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", f"{model_name}.pkl")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info(f"Model {model_name} loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e


def get_model():
    """Get the loaded model instance"""
    return model


def get_model_name():
    """Get the current model name"""
    return model_name


def is_model_loaded():
    """Check if model is loaded"""
    return model is not None


def make_prediction(feature_array):
    """Make a prediction with the loaded model"""
    if model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    return model.predict(feature_array)


def prepare_feature_array(features):
    """Prepare feature array from HousingFeatures object with proper column names"""
    # Create DataFrame with proper feature names to match training data
    if isinstance(features, list):
        # Handle batch of features
        feature_data = {
            "MedInc": [f.MedInc for f in features],
            "HouseAge": [f.HouseAge for f in features],
            "AveRooms": [f.AveRooms for f in features],
            "AveBedrms": [f.AveBedrms for f in features],
            "Population": [f.Population for f in features],
            "AveOccup": [f.AveOccup for f in features],
            "Latitude": [f.Latitude for f in features],
            "Longitude": [f.Longitude for f in features],
        }
    else:
        # Handle single feature
        feature_data = {
            "MedInc": [features.MedInc],
            "HouseAge": [features.HouseAge],
            "AveRooms": [features.AveRooms],
            "AveBedrms": [features.AveBedrms],
            "Population": [features.Population],
            "AveOccup": [features.AveOccup],
            "Latitude": [features.Latitude],
            "Longitude": [features.Longitude],
        }
    return pd.DataFrame(feature_data)


def validate_business_logic(features):
    """Validate business logic rules for housing features"""
    errors = []

    if features.AveBedrms > features.AveRooms:
        errors.append(
            {
                "error": (
                    "Invalid input: Average bedrooms cannot exceed " "average rooms"
                ),
                "suggestion": "Please ensure AveBedrms <= AveRooms",
            }
        )

    return errors
