# UzABSA-LLM: Complete Installation & End-to-End Walkthrough

This guide walks you through **environment setup** and **running the complete pipeline** to fine-tune an LLM on Uzbek Aspect-Based Sentiment Analysis.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Step-by-Step Environment Setup](#step-by-step-environment-setup)
3. [End-to-End Training Walkthrough](#end-to-end-training-walkthrough)
4. [Inference & Evaluation](#inference--evaluation)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Requirement | Recommended | Minimum |
|------------|-------------|---------|
| **Python** | 3.11+ | 3.10 |
| **CUDA** | 12.4 | 11.8 |
| **GPU Memory** | 45+ GB (RTX A6000) | 24+ GB (RTX 4090) |
| **Disk Space** | 200+ GB | 100+ GB |
| **OS** | Windows 10/11, Linux | Any |

### Check Your System

```powershell
# Windows - Show NVIDIA GPU info
nvidia-smi

# Check CUDA version (if installed)
nvcc --version

# Check Python version
python --version
```

---

## Step-by-Step Environment Setup

### Step 1: Clone Repository & Navigate

```powershell
cd C:\Users\{YourUsername}\code
git clone https://github.com/yourusername/UzABSA-LLM.git
cd UzABSA-LLM
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify activation (you should see `(.venv)` in your terminal prompt):
```powershell
# Should show path to .venv
python -c "import sys; print(sys.prefix)"
```

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install PyTorch with CUDA 12.4 (CRITICAL)

This step is **ESSENTIAL** for GPU support. Standard `pip install torch` installs CPU-only version.

**For NVIDIA GPUs (RTX A6000, RTX 4090, etc.):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For CPU only (testing/development):**
```powershell
pip install torch torchvision torchaudio
```

**Verify PyTorch installation:**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

Expected output (with GPU):
```
PyTorch: 2.6.0+cu124
CUDA Available: True
GPU Count: 4
```

### Step 5: Install Project Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- HuggingFace Transformers, Datasets, TRL
- PEFT, bitsandbytes (for 4-bit quantization)
- WandB, TensorBoard (for experiment tracking)
- Matplotlib (for training curves)
- Development tools (black, pytest, etc.)

### Step 6: Install Unsloth (Optimized LLM Training)

```powershell
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

This provides 2x faster training compared to standard HuggingFace.

### Step 7: Verify Complete Setup

```powershell
python -c "
import torch
import transformers
from datasets import load_dataset
from trl import SFTTrainer
from src.gpu_config import get_gpu_info
from src.training_metrics import TrainingMetricsCallback

print('✓ PyTorch', torch.__version__)
print('✓ Transformers', transformers.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU count:', torch.cuda.device_count())
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f'✓ GPU info loaded: {len(gpu_info)} GPUs detected')
print('✓ All imports successful!')
"
```

---

## End-to-End Training Walkthrough

### Full Pipeline Overview

```
1. Data Exploration          → Examine raw & annotated datasets
2. Data Preparation         → Format for instruction tuning, train/val split
3. Model Training           → QLoRA fine-tuning with metrics tracking
4. Evaluation               → Test on validation set
5. Inference                → Run on new Uzbek texts
```

### 0. (Optional) Download Raw Dataset

The raw reviews from sharh.commeta.uz should be in `data/raw/reviews.csv`.

If you need to add more, place CSV file at:
```
data/raw/reviews.csv
```

### 1. Explore Datasets

Examine both the raw reviews and the annotated HuggingFace dataset:

```powershell
# Show statistics on both raw and annotated data
python scripts/explore_datasets.py `
    --raw-file ./data/raw/reviews.csv `
    --analyze
```

Expected output:
```
[1/2] Loading Raw Reviews from sharh.commeta.uz
  Loaded 5058 reviews
  Avg words per text: 13.55
  Avg chars per text: 96.26

[2/2] Loading Annotated ABSA Dataset
  Loaded 6175 examples from Sanatbek/aspect-based-sentiment-analysis-uzbek
```

### 2. Prepare Complete Dataset

Convert raw and annotated data into instruction-tuning format with train/val split:

```powershell
# Prepare full dataset (6,000+ examples, will take 5-10 minutes)
python scripts/prepare_complete_dataset.py `
    --max-examples -1 `
    --output-dir ./data/processed `
    --train-split 0.9
```

Options:
- `--max-examples -1` = use all examples (omit for testing with first 100)
- `--output-dir ./data/processed` = where to save processed data
- `--train-split 0.9` = 90% train, 10% validation

Expected output:
```
Processed dataset saved:
  ./data/processed/train
  ./data/processed/validation
Summary: 5,557 train + 617 validation = 6,174 total
```

### 3. Check GPU Configuration

Before training, verify your GPUs and get batch size recommendations:

```powershell
# Show GPU info and auto-tuning recommendations
python -m src.gpu_config --check

# Get batch size recommendations
python -m src.gpu_config --recommend
```

Expected output (with RTX A6000):
```
GPU Configuration:
  GPU 0: NVIDIA RTX A6000 (45 GB)
  GPU 1: NVIDIA RTX A6000 (45 GB)
  GPU 2: NVIDIA RTX A6000 (45 GB)
  GPU 3: NVIDIA RTX A6000 (45 GB)
  Total: 180 GB

Recommended Config:
  batch_size: 8
  grad_accumulation_steps: 2
  max_steps: 500
```

### 4. Train the Model

Fine-tune Qwen 2.5 7B on Uzbek ABSA dataset:

```powershell
# Basic training (3 epochs, up to max step limit)
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/processed `
    --batch-size 4 `
    --grad-accum 2 `
    --learning-rate 2e-4 `
    --epochs 3 `
    --output-dir ./outputs/qwen_baseline `
    --run-name qwen25-7b-v1
```

**Common Model Options:**
- `qwen2.5-7b` (fastest, lowest VRAM) ← Default
- `qwen2.5-14b` (balanced)
- `qwen2.5-32b` (largest, highest quality)
- `llama3.1-8b`
- `deepseek-7b`
- `mistral-7b`
- `gemma2-9b`

**Training Arguments:**
- `--batch-size 4` = per-device batch size (2-8 recommended)
- `--grad-accum 2` = gradient accumulation steps
- `--epochs 3` = number of training epochs
- `--learning-rate 2e-4` = learning rate (default 2e-4, try 1e-4 to 5e-4)
- `--gpu-id 0` = use specific GPU (default: auto-detect all)
- `--multi-gpu` = enable distributed training across all GPUs

**Training Output:**

During training, you'll see:
```
[Step 10/100] Loss: 2.456, LR: 1.95e-04
[Step 20/100] Loss: 2.123, LR: 1.90e-04
...
[Step 100/100] Loss: 1.234, LR: 1.50e-04
Training complete! Loss: 1.234

Fine-tuning complete!
Model saved to: ./outputs/qwen_baseline/qwen25-7b-v1/
  lora_adapters/          — LoRA weights only (600 MB)
  merged_model/           — Full model (16 GB)
  training_history.json   — Per-step metrics
  training_curves.png     — Loss curve plot (paper-ready)
  experiment_summary.json — Full config + results
```

**Inspect Training Results:**

```powershell
# View saved outputs
ls ./outputs/qwen_baseline/qwen25-7b-v1/

# View training curves (opens in default image viewer)
ii ./outputs/qwen_baseline/qwen25-7b-v1/training_curves.png

# View experiment summary
cat ./outputs/qwen_baseline/qwen25-7b-v1/experiment_summary.json | python -m json.tool

# View training metrics CSV
cat ./outputs/qwen_baseline/qwen25-7b-v1/training_history.csv
```

### 5. Evaluate on Validation Set

Test the fine-tuned model on validation examples:

```powershell
python scripts/evaluate.py `
    --model ./outputs/qwen_baseline/qwen25-7b-v1/merged_model `
    --test-data ./data/processed/validation `
    --output ./outputs/qwen_baseline/qwen25-7b-v1/eval_results.json `
    --uzbek
```

Expected output:
```
Evaluating on 617 examples...

Aspect Term Extraction (Exact Match):
  Precision: 0.7234
  Recall:    0.6891
  F1:        0.7058

Aspect-Polarity Pairs:
  Precision: 0.6543
  Recall:    0.6234
  F1:        0.6385

Sentiment Classification (on matched aspects):
  Accuracy:  0.8234
  Macro F1:  0.7956

Results saved to: ./outputs/qwen_baseline/qwen25-7b-v1/eval_results.json
```

---

## Inference & Evaluation

### Run Inference on Custom Text

Extract aspects and sentiments from Uzbek texts:

```powershell
python scripts/inference.py `
    --model ./outputs/qwen_baseline/qwen25-7b-v1/merged_model `
    --text "Telefon juda barakali, batareya uzoq davom etadi. Narx esa qachon pasaydi?" `
    --uzbek
```

Expected output:
```json
{
  "text": "Telefon juda barakali, batareya uzoq davom etadi. Narx esa qachon pasaydi?",
  "aspects": [
    {
      "term": "telefon",
      "category": "product",
      "polarity": "positive"
    },
    {
      "term": "batareya",
      "category": "battery",
      "polarity": "positive"
    },
    {
      "term": "narx",
      "category": "price",
      "polarity": "negative_query"
    }
  ]
}
```

### Batch Inference from File

Process multiple texts from a JSON file:

```powershell
python scripts/inference.py `
    --model ./outputs/qwen_baseline/qwen25-7b-v1/merged_model `
    --input-file texts.json `
    --output-file predictions.json `
    --batch-size 8
```

Format of `texts.json`:
```json
[
  "Telefon juda barakali",
  "Xizmat yomon edi",
  "Ovqat taomma, lekin qimmat"
]
```

---

## Advanced: Multi-GPU Training

If you have multiple GPUs, enable distributed training:

```powershell
# Use all GPUs automatically
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/processed `
    --multi-gpu `
    --batch-size 4 `
    --learning-rate 2e-4 `
    --output-dir ./outputs/qwen_multigpu
```

Or use specific GPUs:

```powershell
# Use only GPU 0 and 1
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/processed `
    --gpu-id 0 `
    --batch-size 8 `
    --output-dir ./outputs/qwen_gpu0
```

---

## Comparison: Training Multiple Models

Often academic papers compare multiple models. Do this:

**Train Model 1:**
```powershell
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/processed `
    --output-dir ./outputs/comparison `
    --run-name qwen-7b-v1
```

**Train Model 2:**
```powershell
python scripts/train_unsloth.py `
    --model llama3.1-8b `
    --dataset ./data/processed `
    --output-dir ./outputs/comparison `
    --run-name llama-8b-v1
```

**Generate Comparison Plot:**

```powershell
python -c "
from src.training_metrics import plot_model_comparison

dirs = [
    './outputs/comparison/qwen-7b-v1',
    './outputs/comparison/llama-8b-v1',
]
plot_model_comparison(dirs, './outputs/comparison/model_comparison.png')
print('✓ Comparison plot saved!')
"
```

This generates a figure comparing loss curves across models (paper-ready, 300 DPI).

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
1. Reduce batch size: `--batch-size 2` (instead of 4)
2. Increase gradient accumulation: `--grad-accum 8` (instead of 2)
3. Use smaller model: `--model qwen2.5-7b` (instead of 14b or 32b)
4. Enable gradient checkpointing (already enabled by default)

### Issue: "No module named 'torch'" after pip install

**Solution:** Make sure you're using the `.venv` Python:

```powershell
# Verify you're in the right environment
which python  # Linux/Mac
where python  # Windows

# Should show path like: C:\...\UzABSA-LLM\.venv\Scripts\python.exe

# Re-activate if needed
.\.venv\Scripts\Activate.ps1
```

### Issue: "torch was not compiled with CUDA support"

**Solution:** Reinstall PyTorch with CUDA index URL:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Issue: GPUs not detected (CUDA Available: False)

**Solution:**
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA Toolkit 12.4 is installed
3. Uninstall CPU torch: `pip uninstall torch -y`
4. Reinstall with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

### Issue: Training is very slow

**Possible causes:**
1. Using CPU-only PyTorch → Reinstall with CUDA 12.4
2. Batch size too small → Increase `--batch-size`
3. Gradient accumulation too high → Reduce `--grad-accum`
4. Logging too frequent → Default is good (logging_steps=10)

### Issue: "ModuleNotFoundError: No module named 'unsloth'"

**Solution:**

```powershell
# Install Unsloth from GitHub
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify
python -c "from unsloth import FastLanguageModel; print('✓ Unsloth installed')"
```

### Issue: Out of disk space during training

**Solution:** The merged model is ~16 GB. Ensure you have:
- At least 100 GB free for: model + intermediate checkpoints + outputs
- Consider using LoRA-only (no merged) to save space: Remove merged_model save

---

## Summary

**Quick Command to Run Everything (5-10 minutes on RTX A6000):**

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Explore data
python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv

# 3. Prepare dataset
python scripts/prepare_complete_dataset.py `
    --max-examples -1 `
    --output-dir ./data/processed

# 4. Train model
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/processed `
    --epochs 3 `
    --output-dir ./outputs/my_first_run

# 5. Evaluate
python scripts/evaluate.py `
    --model ./outputs/my_first_run/*/merged_model `
    --test-data ./data/processed/validation

# 6. Check results
ls ./outputs/my_first_run/*/
```

All outputs (models + training curves + metrics) are saved to `./outputs/` for academic paper use.

---

**Questions or issues?** See [TROUBLESHOOTING](#troubleshooting) or check the detailed documentation in `GPU_SETUP.md` and `RUN_END_TO_END.md`.
