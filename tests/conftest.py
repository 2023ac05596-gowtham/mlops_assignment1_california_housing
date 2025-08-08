#!/usr/bin/env python3
"""
Test configuration for API tests
"""

import sys
import tempfile
import os
import stat
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Make test runner script executable
test_runner_script = current_dir.parent / "run_tests.sh"
if test_runner_script.exists():
    # Add execute permission for the owner
    current_permissions = test_runner_script.stat().st_mode
    test_runner_script.chmod(current_permissions | stat.S_IEXEC)

# Test configuration
TEST_DATA_DIR = current_dir / "test_data"
TEMP_DIR = tempfile.mkdtemp()

# Sample housing features for testing
SAMPLE_HOUSING_FEATURES = {
    "MedInc": 8.33,
    "HouseAge": 41.0,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322.0,
    "AveOccup": 2.56,
    "Latitude": 37.88,
    "Longitude": -122.23,
}

# Invalid housing features for testing validation
INVALID_HOUSING_FEATURES = {
    "MedInc": -1.0,  # Below minimum
    "HouseAge": 100.0,  # Above maximum
    "AveRooms": 1.0,  # Below minimum
    "AveBedrms": 10.0,  # Above maximum
    "Population": 1.0,  # Below minimum
    "AveOccup": 100.0,  # Above maximum
    "Latitude": 50.0,  # Outside California bounds
    "Longitude": -100.0,  # Outside California bounds
}
