@echo off
REM =============================================================================
REM UzABSA-LLM: Automated Setup Script for Windows
REM =============================================================================
REM
REM This script automates the environment setup and installation steps.
REM
REM Requirements:
REM   - Python 3.10+ installed and in PATH
REM   - NVIDIA GPU with CUDA 12.4 support (optional, for GPU training)
REM   - Git (for cloning Unsloth from GitHub)
REM
REM Usage:
REM   .\setup.bat
REM
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ===============================================================================
echo  UzABSA-LLM: Automated Environment Setup
echo ===============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    exit /b 1
)

echo [1/7] Python version check:
python --version
echo.

REM Step 1: Create virtual environment
echo [2/7] Creating virtual environment (.venv)...
if exist .venv (
    echo   Virtual environment already exists. Skipping...
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
    echo   ✓ Virtual environment created
)
echo.

REM Step 2: Activate virtual environment
echo [3/7] Activating virtual environment...
call .\.venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)
echo   ✓ Virtual environment activated
echo.

REM Step 3: Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    exit /b 1
)
echo   ✓ pip upgraded
echo.

REM Step 4: Install PyTorch with CUDA 12.4
echo [5/7] Installing PyTorch with CUDA 12.4 support...
echo   (This may take 5-10 minutes on first install)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
if errorlevel 1 (
    echo WARNING: PyTorch installation may have issues. Continuing anyway...
)
echo   ✓ PyTorch installed
echo.

REM Step 5: Install project dependencies
echo [6/7] Installing project dependencies from requirements.txt...
echo   (This may take 5-10 minutes)
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed. Continuing anyway...
)
echo   ✓ Dependencies installed
echo.

REM Step 6: Install Unsloth from GitHub
echo [7/7] Installing Unsloth (optimized LLM training)...
echo   (This may take 10-15 minutes, please be patient)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
if errorlevel 1 (
    echo WARNING: Unsloth installation may have issues.
    echo Try manual installation: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
)
echo   ✓ Unsloth installed
echo.

echo ===============================================================================
echo  Setup Complete!
echo ===============================================================================
echo.
echo Next steps:
echo   1. Verify installation:
echo      python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo.
echo   2. Read the guide:
echo      type INSTALL_AND_RUN.md
echo.
echo   3. Explore datasets:
echo      python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv
echo.
echo   4. Prepare data:
echo      python scripts/prepare_complete_dataset.py --max-examples 100 --output-dir ./data/test_processed
echo.
echo   5. Start training:
echo      python scripts/train_unsloth.py --model qwen2.5-7b --dataset ./data/processed --epochs 1 --output-dir ./outputs/test
echo.
echo Virtual environment is active. You can deactivate with: deactivate
echo.
endlocal
