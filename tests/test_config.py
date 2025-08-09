#!/usr/bin/env python3
"""
Unit tests for configuration module
"""

from pathlib import Path
import sys
from config import Config

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfig:
    """Test cases for Config class"""

    def test_api_constants(self):
        """Test API configuration constants"""
        assert Config.API_TITLE == "California Housing Price Prediction API"
        assert Config.API_VERSION == "1.0.0"
        assert isinstance(Config.API_DESCRIPTION, str)
        assert len(Config.API_DESCRIPTION) > 0

    def test_server_configuration(self):
        """Test server configuration settings"""
        assert Config.HOST == "0.0.0.0"
        assert Config.PORT == 8000
        assert Config.LOG_LEVEL == "info"
        assert Config.RELOAD is True

    def test_model_configuration(self):
        """Test model configuration"""
        assert Config.MODEL_NAME == "DecisionTree"

    def test_get_project_root(self):
        """Test project root path calculation"""
        project_root = Config.get_project_root()
        assert isinstance(project_root, Path)
        assert project_root.exists()
        # Should be the parent of src directory
        assert (project_root / "src").exists()

    def test_get_models_dir(self):
        """Test models directory path"""
        models_dir = Config.get_models_dir()
        assert isinstance(models_dir, Path)
        # Should be models folder in project root
        assert models_dir.name == "models"

    def test_get_logs_dir(self):
        """Test logs directory path"""
        logs_dir = Config.get_logs_dir()
        assert isinstance(logs_dir, Path)
        assert logs_dir.name == "logs"

    def test_get_log_file_path(self):
        """Test log file path"""
        log_file = Config.get_log_file_path()
        assert isinstance(log_file, Path)
        assert log_file.name == "api_requests.log"
        assert log_file.parent.name == "logs"

    def test_path_relationships(self):
        """Test relationships between different paths"""
        project_root = Config.get_project_root()
        models_dir = Config.get_models_dir()
        logs_dir = Config.get_logs_dir()
        
        # All should be under project root
        assert models_dir.parent == project_root
        assert logs_dir.parent == project_root
