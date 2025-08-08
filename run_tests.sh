#!/bin/bash
# Test runner script for California Housing Price Prediction API

echo "🧪 Running tests for California Housing Price Prediction API"
echo "============================================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider activating one."
fi

# Install test dependencies if not already installed
echo "📦 Installing test dependencies..."
pip install -r test_requirements.txt

# Run different test suites
echo ""
echo "🔬 Running unit tests..."
pytest tests/test_config.py tests/test_schemas.py tests/test_models.py tests/test_metrics.py tests/test_data_preprocessing.py tests/test_database.py -v

echo ""
echo "🔗 Running integration tests..."
pytest tests/test_api_integration.py -v

echo ""
echo "🚀 Running end-to-end tests..."
pytest tests/test_end_to_end.py -v

echo ""
echo "📊 Running all tests with coverage..."
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

echo ""
echo "✨ Test run complete! Check htmlcov/index.html for detailed coverage report."
