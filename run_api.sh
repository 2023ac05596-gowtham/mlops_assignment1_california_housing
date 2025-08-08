#!/bin/bash

# Script to run the California Housing Price Prediction API

echo "🏠 Starting California Housing Price Prediction API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run 'python -m venv venv' first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check and install dependencies only if needed
echo "⚙️ Checking dependencies..."
pip install -r requirements.txt --quiet

# Create logs directory
mkdir -p logs

# Check if models exist
if [ ! -f "models/DecisionTree.pkl" ]; then
    echo "❌ Model files not found. Please train models first using 'dvc repro' or 'python src/train_model.py'"
    exit 1
fi

echo "🚀 Starting FastAPI server..."
echo "📍 API: http://localhost:8000 | 📖 Docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the API
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload