#!/usr/bin/env python3
"""
Configuration and setup for California Housing Price Prediction API
Handles logging configuration, paths, and application settings
"""

import sys
import logging
from pathlib import Path


class Config:
    """Configuration class for API settings"""

    # API Configuration
    API_TITLE = "California Housing Price Prediction API"
    API_DESCRIPTION = (
        "API for predicting California housing prices using trained ML models"
    )
    API_VERSION = "1.0.0"

    # Server Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    LOG_LEVEL = "info"
    RELOAD = True

    # Model Configuration
    MODEL_NAME = "DecisionTree"

    # Paths
    @classmethod
    def get_project_root(cls) -> Path:
        """Get the project root directory"""
        return Path(__file__).parent.parent

    @classmethod
    def get_models_dir(cls) -> Path:
        """Get the models directory"""
        return cls.get_project_root() / "models"

    @classmethod
    def get_logs_dir(cls) -> Path:
        """Get the logs directory"""
        return cls.get_project_root() / "logs"

    @classmethod
    def get_log_file_path(cls) -> Path:
        """Get the API log file path"""
        return cls.get_logs_dir() / "api_requests.log"

    @classmethod
    def get_db_path(cls) -> Path:
        """Get the SQLite database path"""
        return cls.get_logs_dir() / "predictions.db"


def setup_logging():
    """Setup logging configuration"""
    # Ensure logs directory exists
    Config.get_logs_dir().mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.get_log_file_path()),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist"""
    Config.get_models_dir().mkdir(exist_ok=True, parents=True)


# Initialize configuration on import
setup_directories()
logger = setup_logging()
