#!/usr/bin/env python3
"""
Unit tests for models module
"""

from pathlib import Path
import sys
import numpy as np
import models
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModelUtilities:
    """Test cases for model utility functions"""

    def test_calculate_confidence_score_with_predict_proba(self):
        """Test confidence calculation for probabilistic models"""
        # Mock model with predict_proba
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.7, 0.3]]

        feature_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        confidence = models.calculate_confidence_score(mock_model,
                                                       feature_array)

        assert confidence == 0.7
        mock_model.predict_proba.assert_called_once()

    def test_calculate_confidence_score_with_decision_function(self):
        """Test confidence calculation for models with decision function"""
        # Mock model with decision_function
        mock_model = Mock()
        mock_model.predict_proba = None
        mock_model.decision_function.return_value = [5.0]

        feature_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        confidence = models.calculate_confidence_score(mock_model,
                                                       feature_array)

        assert confidence == 0.5  # min(5.0/10.0, 1.0)
        mock_model.decision_function.assert_called_once()

    def test_calculate_confidence_score_with_decision_path(self):
        """Test confidence calculation for tree-based models"""
        # Mock tree model
        mock_model = Mock()
        mock_model.predict_proba = None
        mock_model.decision_function = None

        # Mock decision path
        mock_path = Mock()
        mock_path.toarray.return_value = np.array([[1, 1, 1, 0, 0]])  # depth 3
        mock_model.decision_path.return_value = mock_path

        # Mock tree structure
        mock_tree = Mock()
        mock_tree.max_depth = 5
        mock_model.tree_ = mock_tree

        feature_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        confidence = models.calculate_confidence_score(mock_model,
                                                       feature_array)

        assert confidence == 0.6  # 3/5

    def test_calculate_confidence_score_fallback(self):
        """Test confidence calculation fallback for simple models"""
        # Mock simple model without special methods
        mock_model = Mock()
        mock_model.predict_proba = None
        mock_model.decision_function = None
        mock_model.decision_path = None

        feature_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        confidence = models.calculate_confidence_score(mock_model,
                                                       feature_array)

        assert confidence == 0.75  # Default for simple models

    def test_calculate_confidence_score_exception_handling(self):
        """Test confidence calculation exception handling"""
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.predict_proba.side_effect = Exception("Test error")
        feature_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        confidence = models.calculate_confidence_score(mock_model,
                                                       feature_array)

        assert confidence == 0.6  # Fallback confidence

    def test_calculate_prediction_range(self):
        """Test prediction range calculation"""
        prediction = 400000.0
        confidence = 0.8

        result = models.calculate_prediction_range(prediction, confidence)

        assert "min" in result
        assert "max" in result
        assert "range_percent" in result
        assert result["min"] < prediction
        assert result["max"] > prediction
        assert result["min"] >= 0  # Should not go below 0

    def test_calculate_prediction_range_low_confidence(self):
        """Test prediction range with low confidence"""
        prediction = 500000.0
        confidence = 0.3  # Low confidence

        result = models.calculate_prediction_range(prediction, confidence)

        # Lower confidence should result in wider range
        range_width = result["max"] - result["min"]
        assert range_width > prediction * 0.3  # Should be substantial range

    def test_get_model_name(self):
        """Test getting model name"""
        name = models.get_model_name()
        assert name == "DecisionTree"

    def test_is_model_loaded_initially_false(self):
        """Test model loaded status initially false"""
        # Reset model to None
        models.model = None
        assert models.is_model_loaded() is False

    def test_get_model_when_none(self):
        """Test getting model when none loaded"""
        models.model = None
        assert models.get_model() is None

    @patch('models.joblib.load')
    @patch('models.os.path.exists')
    def test_load_model_success(self, mock_exists, mock_joblib_load):
        """Test successful model loading"""
        # Setup mocks
        mock_exists.return_value = True
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model

        # Load model
        models.load_model()

        # Verify
        assert models.is_model_loaded() is True
        assert models.get_model() == mock_model
        mock_joblib_load.assert_called_once()

    @patch('models.os.path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading when file doesn't exist"""
        mock_exists.return_value = False

        try:
            models.load_model()
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "Model file not found" in str(e)

    @patch('models.joblib.load')
    @patch('models.os.path.exists')
    def test_load_model_loading_error(self, mock_exists, mock_joblib_load):
        """Test model loading with joblib error"""
        mock_exists.return_value = True
        mock_joblib_load.side_effect = Exception("Loading failed")

        try:
            models.load_model()
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Loading failed" in str(e)

    def test_make_prediction_no_model_loaded(self):
        """Test making prediction when no model loaded"""
        models.model = None

        try:
            models.make_prediction(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]))
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Model not loaded" in str(e)

    def test_make_prediction_with_model(self):
        """Test making prediction with loaded model"""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = [450000.0]
        models.model = mock_model

        feature_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        prediction = models.make_prediction(feature_array)

        assert prediction == [450000.0]
        mock_model.predict.assert_called_once_with(feature_array)
