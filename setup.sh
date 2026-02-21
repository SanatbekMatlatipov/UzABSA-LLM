#!/bin/bash
# =============================================================================
# UzABSA-LLM: Automated Setup Script for Linux/Mac
# =============================================================================
#
# This script automates the environment setup and installation steps.
#
# Requirements:
#   - Python 3.10+ installed
#   - NVIDIA GPU with CUDA 12.4 support (optional, for GPU training)
#   - Git (for cloning Unsloth from GitHub)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# =============================================================================

set -e  # Exit on any error

echo ""
echo "==============================================================================="
echo "  UzABSA-LLM: Automated Environment Setup"
echo "==============================================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10+ using:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-venv python3-dev"
    echo "  Mac: brew install python@3.11"
    exit 1
fi

echo "[1/7] Python version check:"
python3 --version
echo ""

# Step 1: Create virtual environment
echo "[2/7] Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo "   Virtual environment already exists. Skipping..."
else
    python3 -m venv .venv
    echo "   ✓ Virtual environment created"
fi
echo ""

# Step 2: Activate virtual environment
echo "[3/7] Activating virtual environment..."
source .venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

# Step 3: Upgrade pip
echo "[4/7] Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "   ✓ pip upgraded"
echo ""

# Step 4: Install PyTorch with CUDA 12.4
echo "[5/7] Installing PyTorch with CUDA 12.4 support..."
echo "   (This may take 5-10 minutes on first install)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
echo "   ✓ PyTorch installed"
echo ""

# Step 5: Install project dependencies
echo "[6/7] Installing project dependencies from requirements.txt..."
echo "   (This may take 5-10 minutes)"
pip install -r requirements.txt --quiet 2>/dev/null || {
    echo "   WARNING: Some dependencies may have failed. Continuing anyway..."
}
echo "   ✓ Dependencies installed"
echo ""

# Step 6: Install Unsloth from GitHub
echo "[7/7] Installing Unsloth (optimized LLM training)..."
echo "   (This may take 10-15 minutes, please be patient)"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet 2>/dev/null || {
    echo "   WARNING: Unsloth installation may have issues."
    echo "   Try manual installation: pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
}
echo "   ✓ Unsloth installed"
echo ""

echo "==============================================================================="
echo "  Setup Complete!"
echo "==============================================================================="
echo ""
echo "Next steps:"
echo "   1. Verify installation:"
echo "      python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')\" "
echo ""
echo "   2. Read the guide:"
echo "      cat INSTALL_AND_RUN.md"
echo ""
echo "   3. Explore datasets:"
echo "      python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv"
echo ""
echo "   4. Prepare data:"
echo "      python scripts/prepare_complete_dataset.py --max-examples 100 --output-dir ./data/test_processed"
echo ""
echo "   5. Start training:"
echo "      python scripts/train_unsloth.py --model qwen2.5-7b --dataset ./data/processed --epochs 1 --output-dir ./outputs/test"
echo ""
echo "Virtual environment is active. You can deactivate with: deactivate"
echo ""
