#!/usr/bin/env python3
"""
Unit tests for data preprocessing module
"""

from pathlib import Path
import sys
import pandas as pd
import data_preprocessing
import tempfile
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataPreprocessing:
    """Test cases for data preprocessing functions"""

    def test_preprocess_function_exists(self):
        """Test that preprocess function exists and is callable"""
        assert hasattr(data_preprocessing, 'preprocess')
        assert callable(data_preprocessing.preprocess)

    @patch('data_preprocessing.pd.read_csv')
    @patch('data_preprocessing.os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_preprocess_basic_functionality(
        self, mock_to_csv, mock_makedirs, mock_read_csv
    ):
        """Test basic preprocessing functionality"""
        # Create mock dataframe
        mock_df = MagicMock()
        mock_df.dropna.return_value = mock_df
        mock_df.__setitem__ = MagicMock()  # For df["MedHouseVal"] = ...
        mock_read_csv.return_value = mock_df

        # Call preprocess
        data_preprocessing.preprocess()

        # Verify calls
        mock_read_csv.assert_called_once_with(
            "data/raw/california_housing.csv"
        )
        mock_df.dropna.assert_called_once()
        mock_makedirs.assert_called_once_with(
            "data/processed", exist_ok=True
        )
        mock_to_csv.assert_called_once_with(
            "data/processed/california_housing.csv", index=False
        )

    @patch('data_preprocessing.pd.read_csv')
    def test_preprocess_handles_missing_file(self, mock_read_csv):
        """Test preprocessing handles missing input file"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        try:
            data_preprocessing.preprocess()
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected behavior

    def test_preprocess_with_real_data(self):
        """Test preprocessing with actual sample data"""
        # Create temporary directories and files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths
            raw_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)

            # Create sample data
            sample_data = pd.DataFrame({
                'MedInc': [8.3252, 8.3014, 7.2574],
                'HouseAge': [41.0, 21.0, 52.0],
                'AveRooms': [6.98, 6.24, 8.29],
                'AveBedrms': [1.02, 0.97, 1.07],
                'Population': [322.0, 2401.0, 496.0],
                'AveOccup': [2.56, 2.11, 2.03],
                'Latitude': [37.88, 37.86, 37.85],
                'Longitude': [-122.23, -122.22, -122.24],
                'MedHouseVal': [4.526, 3.585, 3.521]  # Original values
            })

            # Add some NaN values to test dropna
            sample_data.loc[1, 'MedInc'] = None

            raw_file = os.path.join(raw_dir, "california_housing.csv")
            sample_data.to_csv(raw_file, index=False)

            # Patch the file paths in the function
            with patch('data_preprocessing.pd.read_csv') as mock_read_csv, \
                 patch('data_preprocessing.os.makedirs') as mock_makedirs:

                mock_read_csv.return_value = sample_data
                mock_makedirs.return_value = None

                # Mock the to_csv method to save in our temp directory
                with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                    data_preprocessing.preprocess()

                    # Verify the data transformation

                    # Check that the function was called
                    mock_read_csv.assert_called_once()
                    mock_makedirs.assert_called_once()
                    mock_to_csv.assert_called_once()

    def test_median_house_value_transformation(self):
        """Test that MedHouseVal is properly transformed"""
        # Create sample data
        sample_data = pd.DataFrame({
            'MedInc': [8.3252],
            'HouseAge': [41.0],
            'MedHouseVal': [4.526]  # Original value
        })

        with patch('data_preprocessing.pd.read_csv') as mock_read_csv, \
             patch('data_preprocessing.os.makedirs'), \
             patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:

            mock_read_csv.return_value = sample_data

            # Mock to_csv to capture the dataframe being saved
            def capture_df(path, **kwargs):
                # The dataframe should have transformed MedHouseVal
                assert sample_data['MedHouseVal'].iloc[0] == 452600.0
                # 4.526 * 100000

            mock_to_csv.side_effect = capture_df

            data_preprocessing.preprocess()

    def test_directory_creation(self):
        """Test that processed directory is created"""
        with patch('data_preprocessing.pd.read_csv') as mock_read_csv, \
             patch('data_preprocessing.os.makedirs') as mock_makedirs, \
             patch.object(pd.DataFrame, 'to_csv'):

            # Setup mock dataframe
            mock_df = MagicMock()
            mock_df.dropna.return_value = mock_df
            mock_read_csv.return_value = mock_df

            data_preprocessing.preprocess()

            # Verify directory creation
            mock_makedirs.assert_called_once_with(
                "data/processed", exist_ok=True
            )
