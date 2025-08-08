#!/bin/bash
# CI/CD optimized test runner script

set -e  # Exit on any error

echo "🧪 Running tests for California Housing Price Prediction API"
echo "============================================================"

# Check if we're in CI environment
if [[ "$CI" == "true" ]] || [[ "$GITHUB_ACTIONS" == "true" ]] || [[ "$GITLAB_CI" == "true" ]]; then
    echo "🤖 CI/CD environment detected"
    CI_MODE=true
else
    echo "💻 Local development environment"
    CI_MODE=false
fi

# Install dependencies
echo "📦 Installing dependencies..."
if [[ "$CI_MODE" == "true" ]]; then
    # In CI, be more verbose and explicit
    python -m pip install --upgrade pip
    pip install -r test_requirements.txt --no-cache-dir
else
    # Local development
    pip install -r test_requirements.txt
fi

# Check if all required packages are available
echo "🔍 Checking required packages..."
python -c "
import sys
required_packages = ['fastapi', 'pytest', 'httpx', 'pydantic', 'numpy', 'pandas', 'scikit-learn', 'joblib']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg} - MISSING')

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required packages are available!')
"

# Set test options based on environment
if [[ "$CI_MODE" == "true" ]]; then
    PYTEST_OPTIONS="--tb=short --strict-markers --disable-warnings"
    COVERAGE_OPTIONS="--cov=src --cov-report=xml --cov-report=html --cov-report=term-missing"
else
    PYTEST_OPTIONS="-v"
    COVERAGE_OPTIONS="--cov=src --cov-report=html --cov-report=term-missing"
fi

# Run tests
echo ""
echo "🧪 Running all tests..."
pytest tests/ $PYTEST_OPTIONS $COVERAGE_OPTIONS

# Check exit code
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ All tests passed!"
    if [[ "$CI_MODE" == "false" ]]; then
        echo "📊 Coverage report: htmlcov/index.html"
    fi
else
    echo ""
    echo "❌ Some tests failed!"
    exit 1
fi
