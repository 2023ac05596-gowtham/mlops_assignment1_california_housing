#!/usr/bin/env python3
"""
Unit tests for schemas module
"""

from pathlib import Path
import sys
import pytest
from pydantic import ValidationError

from schemas import HousingFeatures, PredictionResponse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestHousingFeatures:
    """Test cases for HousingFeatures schema"""

    def test_valid_housing_features(self):
        """Test valid housing features creation"""
        features = HousingFeatures(
            MedInc=8.33,
            HouseAge=41.0,
            AveRooms=6.98,
            AveBedrms=1.02,
            Population=322.0,
            AveOccup=2.56,
            Latitude=37.88,
            Longitude=-122.23,
        )

        assert features.MedInc == 8.33
        assert features.HouseAge == 41.0
        assert features.AveRooms == 6.98
        assert features.AveBedrms == 1.02
        assert features.Population == 322.0
        assert features.AveOccup == 2.56
        assert features.Latitude == 37.88
        assert features.Longitude == -122.23

    def test_median_income_validation(self):
        """Test median income field validation"""
        # Valid range
        features = HousingFeatures(
            MedInc=5.0, HouseAge=25.0, AveRooms=5.0, AveBedrms=1.0,
            Population=1000.0, AveOccup=3.0, Latitude=35.0,
            Longitude=-120.0
        )
        assert features.MedInc == 5.0

        # Below minimum
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=0.4, HouseAge=25.0, AveRooms=5.0, AveBedrms=1.0,
                Population=1000.0, AveOccup=3.0, Latitude=35.0,
                Longitude=-120.0
            )

        # Above maximum
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=16.0, HouseAge=25.0, AveRooms=5.0, AveBedrms=1.0,
                Population=1000.0, AveOccup=3.0, Latitude=35.0,
                Longitude=-120.0
            )

    def test_house_age_validation(self):
        """Test house age field validation"""
        # Below minimum
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=5.0, HouseAge=0.5, AveRooms=5.0, AveBedrms=1.0,
                Population=1000.0, AveOccup=3.0, Latitude=35.0,
                Longitude=-120.0
            )

        # Above maximum
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=5.0, HouseAge=53.0, AveRooms=5.0, AveBedrms=1.0,
                Population=1000.0, AveOccup=3.0, Latitude=35.0,
                Longitude=-120.0
            )

    def test_latitude_longitude_validation(self):
        """Test latitude and longitude California bounds"""
        # Invalid latitude (too high)
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=5.0, HouseAge=25.0, AveRooms=5.0, AveBedrms=1.0,
                Population=1000.0, AveOccup=3.0, Latitude=45.0,
                Longitude=-120.0
            )

        # Invalid longitude (too east)
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=5.0, HouseAge=25.0, AveRooms=5.0, AveBedrms=1.0,
                Population=1000.0, AveOccup=3.0, Latitude=35.0,
                Longitude=-100.0
            )

    def test_population_validation(self):
        """Test population field validation"""
        # Below minimum
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=5.0, HouseAge=25.0, AveRooms=5.0, AveBedrms=1.0,
                Population=2.0, AveOccup=3.0, Latitude=35.0, Longitude=-120.0
            )

        # Above maximum
        with pytest.raises(ValidationError):
            HousingFeatures(
                MedInc=5.0, HouseAge=25.0, AveRooms=5.0, AveBedrms=1.0,
                Population=40001.0, AveOccup=3.0, Latitude=35.0,
                Longitude=-120.0
            )

    def test_from_dict(self):
        """Test creating HousingFeatures from dictionary"""
        data = {
            "MedInc": 8.33,
            "HouseAge": 41.0,
            "AveRooms": 6.98,
            "AveBedrms": 1.02,
            "Population": 322.0,
            "AveOccup": 2.56,
            "Latitude": 37.88,
            "Longitude": -122.23,
        }
        features = HousingFeatures(**data)
        assert features.MedInc == data["MedInc"]
        assert features.Latitude == data["Latitude"]


class TestPredictionResponse:
    """Test cases for PredictionResponse schema"""

    def test_valid_prediction_response(self):
        """Test valid prediction response creation"""
        response = PredictionResponse(
            predicted_price=450000.0,
            confidence_score=0.85,
            model_name="DecisionTree",
            prediction_range={
                "min": 400000.0,
                "max": 500000.0,
                "range_percent": "11.1%"
            }
        )

        assert response.predicted_price == 450000.0
        assert response.confidence_score == 0.85
        assert response.model_name == "DecisionTree"
        assert response.prediction_range["min"] == 400000.0

    def test_prediction_response_serialization(self):
        """Test prediction response JSON serialization"""
        response = PredictionResponse(
            predicted_price=450000.0,
            confidence_score=0.85,
            model_name="DecisionTree",
            prediction_range={
                "min": 400000.0,
                "max": 500000.0,
                "range_percent": "11.1%"
            }
        )

        json_data = response.model_dump()
        assert "predicted_price" in json_data
        assert "confidence_score" in json_data
        assert "model_name" in json_data
        assert "prediction_range" in json_data
