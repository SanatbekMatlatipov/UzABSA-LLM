# UzABSA-LLM: Quick Start Guide

Get started with fine-tuning LLMs for Uzbek ABSA in 5 minutes.

## 1. Setup (First Time Only)

### Windows:
```powershell
# Run automated setup
.\setup.bat

# Verify installation
python -c "import torch; print('✓ Ready!' if torch.cuda.is_available() else 'CPU mode')"
```

### Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh

# Verify installation
python -c "import torch; print('✓ Ready!' if torch.cuda.is_available() else 'CPU mode')"
```

---

## 2. Run End-to-End Training

```powershell
# Activate environment (if not already active)
.\.venv\Scripts\Activate.ps1

# Step 1: Explore data (2 min)
python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv

# Step 2: Prepare dataset (5 min for full, <1 min for test)
# For testing:
python scripts/prepare_complete_dataset.py --max-examples 100 --output-dir ./data/test_processed

# For full run:
python scripts/prepare_complete_dataset.py --max-examples -1 --output-dir ./data/processed

# Step 3: Train model (10-60 min depending on size + GPU)
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/processed `
    --epochs 3 `
    --batch-size 4 `
    --output-dir ./outputs/my_run

# Step 4: View results
ls ./outputs/my_run/*/
```

---

## 3. Expected Outputs

After training, your outputs directory contains:

| File | Purpose | Usage |
|------|---------|-------|
| `merged_model/` | Full fine-tuned model | Inference/deployment |
| `lora_adapters/` | LoRA weights only | Resume training, smaller storage |
| `training_curves.png` | Loss curves (300 DPI) | Paper figures |
| `lr_schedule.png` | Learning rate plot | Paper figures |
| `training_history.json` | All per-step metrics | Plotting, analysis |
| `training_history.csv` | Metrics as CSV | Excel, pandas, R |
| `experiment_summary.json` | Full config + results | Paper appendix, reproducibility |

---

## 4. Run Inference

```powershell
# Single text
python scripts/inference.py `
    --model ./outputs/my_run/*/merged_model `
    --text "Telefon juda barakali lekin qimmat"

# Batch from file
python scripts/inference.py `
    --model ./outputs/my_run/*/merged_model `
    --input-file texts.json `
    --output-file predictions.json
```

---

## 5. Evaluate on Test Set

```powershell
python scripts/evaluate.py `
    --model ./outputs/my_run/*/merged_model `
    --test-data ./data/processed/validation `
    --output ./outputs/my_run/*/eval_results.json
```

---

## 6. Command Reference

### Model Training
```powershell
python scripts/train_unsloth.py `
    --model qwen2.5-7b `           # Model name (see below)
    --dataset ./data/processed `   # Dataset directory
    --batch-size 4 `               # Per-device batch size
    --grad-accum 2 `               # Gradient accumulation
    --epochs 3 `                   # Training epochs
    --learning-rate 2e-4 `         # Learning rate
    --output-dir ./outputs/my_run `# Output directory
    --run-name experiment_v1 `     # Experiment name
    --no-wandb                     # Skip WandB logging
```

### Available Models
- `qwen2.5-7b` ← **Default** (fastest)
- `qwen2.5-14b` (balanced)
- `qwen2.5-32b` (largest)
- `llama3.1-8b`
- `deepseek-7b`
- `mistral-7b`
- `gemma2-9b`

### GPU Configuration
```powershell
# Check available GPUs
python -m src.gpu_config --check

# Get batch size recommendations
python -m src.gpu_config --recommend

# Use specific GPU
python scripts/train_unsloth.py --gpu-id 0 --model qwen2.5-7b --dataset ./data/processed

# Use all GPUs
python scripts/train_unsloth.py --multi-gpu --model qwen2.5-7b --dataset ./data/processed
```

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| **"CUDA not found"** | `pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| **"Out of memory"** | Reduce `--batch-size` (2 or 1) or `--max-seq-length` |
| **Very slow training** | Check if CUDA available: `python -c "import torch; print(torch.cuda.is_available())"` |
| **Module not found** | Make sure you're in `.venv`: `python -c "import sys; print(sys.prefix)"` |
| **Unsloth install fails** | Try: `pip install --upgrade pip` then retry |
| **"AttrsDescriptor" error** | Triton version mismatch: `pip install triton-windows==3.2.0.post19` |
| **Empty training log files** | Likely an import crash — check with: `python -c "from unsloth import FastLanguageModel"` |

See [INSTALL_AND_RUN.md](INSTALL_AND_RUN.md) for detailed troubleshooting.

---

## 8. Paper-Ready Results

After training, you have everything needed for a research paper:

```powershell
# View training curves
ii ./outputs/my_run/*/training_curves.png

# View experiment config + results
cat ./outputs/my_run/*/experiment_summary.json | python -m json.tool

# Compare multiple models
python -c "
from src.training_metrics import plot_model_comparison
plot_model_comparison([
    './outputs/my_run/qwen-v1',
    './outputs/my_run/llama-v1',
], './outputs/comparison.png')
"
```

All outputs are 300 DPI, publication-ready PNG files.

---

## 9. Full Documentation

For complete details, see:
- [INSTALL_AND_RUN.md](INSTALL_AND_RUN.md) — Full installation guide
- [GPU_SETUP.md](GPU_SETUP.md) — GPU configuration details  
- [RESEARCH_LOG.md](RESEARCH_LOG.md) — Research methodology
- [README.md](README.md) — Project overview

---

**Questions?** Check the [INSTALL_AND_RUN.md](INSTALL_AND_RUN.md) troubleshooting section or the code documentation.
