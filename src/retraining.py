#!/usr/bin/env python3
"""
Model re-training functionality for California Housing API
Handles new data submission and automated retraining triggers
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sqlite3
import json
import threading
import time

# Setup logger
logger = logging.getLogger(__name__)


class ModelRetrainingManager:
    """
    Manages model retraining triggers and processes
    """

    def __init__(self, db_path: str, models_dir: str, data_dir: str):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Retraining configuration
        self.min_new_samples = 50  # Minimum new samples before retraining
        self.max_retrain_attempts = 2  # Maximum retrain attempts per day

        # Initialize new data storage
        self.new_data_file = self.data_dir / "new_training_data.csv"
        self.retrain_log_file = self.data_dir / "retrain_log.json"

        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage files for new data"""
        if not self.new_data_file.exists():
            # Create empty CSV with proper columns
            empty_df = pd.DataFrame(
                columns=[
                    "MedInc",
                    "HouseAge",
                    "AveRooms",
                    "AveBedrms",
                    "Population",
                    "AveOccup",
                    "Latitude",
                    "Longitude",
                    "target",
                    "timestamp",
                ]
            )
            empty_df.to_csv(self.new_data_file, index=False)

        if not self.retrain_log_file.exists():
            with open(self.retrain_log_file, "w") as f:
                json.dump([], f)

    def add_training_data(
        self, features: Dict[str, float], target: float
    ) -> Dict[str, Any]:
        """
        Add new training data point
        """
        try:
            # Create new data row
            new_row = {
                "MedInc": features["MedInc"],
                "HouseAge": features["HouseAge"],
                "AveRooms": features["AveRooms"],
                "AveBedrms": features["AveBedrms"],
                "Population": features["Population"],
                "AveOccup": features["AveOccup"],
                "Latitude": features["Latitude"],
                "Longitude": features["Longitude"],
                "target": target,
                "timestamp": datetime.now().isoformat(),
            }

            # Append to CSV
            new_df = pd.DataFrame([new_row])
            if self.new_data_file.exists() and self.new_data_file.stat().st_size > 0:
                new_df.to_csv(self.new_data_file, mode="a", header=False, index=False)
            else:
                new_df.to_csv(self.new_data_file, index=False)

            # Get current count of new samples
            current_data = pd.read_csv(self.new_data_file)
            new_sample_count = len(current_data)

            logger.info(
                f"Added new training sample. Total new samples: {new_sample_count}"
            )

            # Check if retraining should be triggered
            should_retrain, reason = self._should_trigger_retraining(new_sample_count)

            return {
                "status": "success",
                "message": f"Training data added successfully",
                "new_samples_count": new_sample_count,
                "should_retrain": should_retrain,
                "retrain_reason": reason if should_retrain else None,
            }

        except Exception as e:
            logger.error(f"Error adding training data: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to add training data: {str(e)}",
            }

    def _should_trigger_retraining(self, new_sample_count: int) -> Tuple[bool, str]:
        """
        Determine if retraining should be triggered
        """
        # Check minimum samples threshold
        if new_sample_count >= self.min_new_samples:
            return (
                True,
                f"Minimum samples threshold reached ({new_sample_count} >= {self.min_new_samples})",
            )

        return False, "Not enough new samples for retraining"

    def _get_daily_retrain_attempts(self) -> int:
        """Get number of retraining attempts today"""
        try:
            with open(self.retrain_log_file, "r") as f:
                retrain_log = json.load(f)

            today = datetime.now().date().isoformat()
            today_attempts = [
                entry
                for entry in retrain_log
                if entry.get("date", "").startswith(today)
            ]
            return len(today_attempts)
        except:
            return 0

    def trigger_retraining(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Trigger model retraining process
        """
        retrain_start = time.time()

        try:
            logger.info(f"Starting model retraining: {reason}")

            # Check daily attempts
            if self._get_daily_retrain_attempts() >= self.max_retrain_attempts:
                return {
                    "status": "error",
                    "message": f"Maximum daily retrain attempts ({self.max_retrain_attempts}) reached",
                }

            # Load new training data
            if not self.new_data_file.exists():
                return {"status": "error", "message": "No new training data available"}

            new_data = pd.read_csv(self.new_data_file)
            if len(new_data) == 0:
                return {
                    "status": "error",
                    "message": "No new training data samples found",
                }

            # Load existing training data (from original dataset)
            original_data = self._load_original_training_data()

            # Combine datasets
            combined_data = pd.concat(
                [original_data, new_data[original_data.columns]], ignore_index=True
            )

            logger.info(
                f"Combined dataset size: {len(combined_data)} samples (original: {len(original_data)}, new: {len(new_data)})"
            )

            # Prepare features and targets
            feature_cols = [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ]
            X = combined_data[feature_cols]
            y = combined_data["target"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train only DecisionTree model (best performing from original training)
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            best_model_name = "DecisionTree"
            best_model_info = {"model": model, "rmse": rmse, "r2": r2}

            # Save new model
            model_save_path = self.models_dir / f"{best_model_name}_retrained.pkl"
            joblib.dump(best_model_info["model"], model_save_path)

            # Backup old model and replace with new one
            old_model_path = self.models_dir / f"{best_model_name}.pkl"
            if old_model_path.exists():
                backup_path = (
                    self.models_dir
                    / f"{best_model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                )
                old_model_path.rename(backup_path)

            model_save_path.rename(old_model_path)

            # Archive new training data
            self._archive_training_data()

            # Log retraining event
            retrain_duration = time.time() - retrain_start
            self._log_retrain_event(
                reason, best_model_name, best_model_info, retrain_duration
            )

            logger.info(
                f"Retraining completed successfully. Best model: {best_model_name} (RMSE: {best_model_info['rmse']:.2f})"
            )

            return {
                "status": "success",
                "message": "Model retraining completed successfully",
                "best_model": best_model_name,
                "model_performance": {
                    "rmse": round(best_model_info["rmse"], 2),
                    "r2_score": round(best_model_info["r2"], 3),
                },
                "training_data_size": len(combined_data),
                "new_samples_used": len(new_data),
                "duration_seconds": round(retrain_duration, 2),
            }

        except Exception as e:
            retrain_duration = time.time() - retrain_start
            error_msg = f"Retraining failed: {str(e)}"
            logger.error(error_msg)

            # Log failed attempt
            self._log_retrain_event(
                reason, "failed", {"error": str(e)}, retrain_duration, success=False
            )

            return {
                "status": "error",
                "message": error_msg,
                "duration_seconds": round(retrain_duration, 2),
            }

    def _load_original_training_data(self) -> pd.DataFrame:
        """Load original training data"""
        from sklearn.datasets import fetch_california_housing

        # Load California housing dataset
        housing = fetch_california_housing()

        # Create DataFrame
        feature_names = housing.feature_names
        data = pd.DataFrame(housing.data, columns=feature_names)
        data["target"] = housing.target

        # Convert target to hundreds of thousands (as used in training)
        data["target"] = data["target"] * 100  # Original is in hundreds of thousands

        return data

    def _archive_training_data(self):
        """Archive new training data after successful retraining"""
        try:
            archive_path = (
                self.data_dir
                / f"archived_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.new_data_file.rename(archive_path)

            # Create fresh empty file
            empty_df = pd.DataFrame(
                columns=[
                    "MedInc",
                    "HouseAge",
                    "AveRooms",
                    "AveBedrms",
                    "Population",
                    "AveOccup",
                    "Latitude",
                    "Longitude",
                    "target",
                    "timestamp",
                ]
            )
            empty_df.to_csv(self.new_data_file, index=False)

        except Exception as e:
            logger.warning(f"Failed to archive training data: {str(e)}")

    def _log_retrain_event(
        self,
        reason: str,
        model_name: str,
        model_info: Dict,
        duration: float,
        success: bool = True,
    ):
        """Log retraining event"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().date().isoformat(),
                "reason": reason,
                "model_name": model_name,
                "success": success,
                "duration_seconds": duration,
            }

            if success and isinstance(model_info, dict):
                log_entry.update(
                    {
                        "rmse": model_info.get("rmse", 0),
                        "r2_score": model_info.get("r2", 0),
                    }
                )
            elif not success:
                log_entry["error"] = str(model_info.get("error", "Unknown error"))

            # Read existing log
            with open(self.retrain_log_file, "r") as f:
                retrain_log = json.load(f)

            # Add new entry
            retrain_log.append(log_entry)

            # Keep only last 100 entries
            if len(retrain_log) > 100:
                retrain_log = retrain_log[-100:]

            # Write back
            with open(self.retrain_log_file, "w") as f:
                json.dump(retrain_log, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to log retrain event: {str(e)}")

    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status and statistics"""
        try:
            # Get new data count
            new_data_count = 0
            if self.new_data_file.exists():
                new_data = pd.read_csv(self.new_data_file)
                new_data_count = len(new_data)

            # Get model age
            model_age_days = None
            latest_model_path = self.models_dir / "DecisionTree.pkl"
            if latest_model_path.exists():
                model_age_days = (
                    datetime.now()
                    - datetime.fromtimestamp(latest_model_path.stat().st_mtime)
                ).days

            # Get recent retraining history
            recent_retrains = []
            if self.retrain_log_file.exists():
                with open(self.retrain_log_file, "r") as f:
                    retrain_log = json.load(f)
                recent_retrains = retrain_log[-5:]  # Last 5 entries

            # Check if retraining should be triggered
            should_retrain, reason = self._should_trigger_retraining(new_data_count)

            return {
                "new_data_samples": new_data_count,
                "model_age_days": model_age_days,
                "should_retrain": should_retrain,
                "retrain_reason": reason if should_retrain else None,
                "daily_retrain_attempts": self._get_daily_retrain_attempts(),
                "max_daily_attempts": self.max_retrain_attempts,
                "thresholds": {
                    "min_new_samples": self.min_new_samples,
                },
                "recent_retrains": recent_retrains,
            }

        except Exception as e:
            logger.error(f"Error getting retraining status: {str(e)}")
            return {"error": f"Failed to get retraining status: {str(e)}"}


# Global retraining manager instance
_retraining_manager = None


def get_retraining_manager(
    db_path: str, models_dir: str, data_dir: str
) -> ModelRetrainingManager:
    """Get or create the global retraining manager"""
    global _retraining_manager
    if _retraining_manager is None:
        _retraining_manager = ModelRetrainingManager(db_path, models_dir, data_dir)
    return _retraining_manager
