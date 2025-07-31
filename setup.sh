#!/bin/bash

# Llama3 Fine-tuning Project Setup Script

echo "Setting up Llama3 MAWPS Fine-tuning Project..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python3 is required but not installed. Please install Python3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run training: python scripts/train.py"
echo "3. Or explore the notebook: jupyter lab notebooks/llama3_mawps_finetuning.ipynb"
echo ""
echo "For help:"
echo "  python scripts/train.py --help"
echo "  python scripts/inference.py --help"
echo "  python scripts/evaluate.py --help"
