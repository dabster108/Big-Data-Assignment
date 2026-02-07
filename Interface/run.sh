#!/bin/bash

# Bus Travel Time Prediction - Quick Start Script

echo "=========================================="
echo "Bus Travel Time Prediction System"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the Interface folder"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check if model files exist
MODEL_DIR="../Model"
if [ ! -d "$MODEL_DIR/rf_model_latest" ]; then
    echo "⚠️  Warning: Model files not found in $MODEL_DIR"
    echo "Please ensure models are trained and saved before running"
fi

# Launch Streamlit
echo ""
echo "=========================================="
echo "🚀 Launching Streamlit application..."
echo "=========================================="
echo ""
streamlit run app.py
