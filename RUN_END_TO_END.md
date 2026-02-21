# UzABSA-LLM: Complete End-to-End Guide
# ======================================================================
# How to Run the Project from Start to Finish
# Date: February 2026
# ======================================================================


## STEP 0: Environment Setup (One-Time)

### 0.1 Activate Virtual Environment

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

**Expected output:** `.venv` appears in terminal prompt like `(.venv) C:\Users\User\code\UzABSA-LLM>`

### 0.2 Verify Python Installation

```bash
python --version
# Expected: Python 3.10+

pip --version
# Expected: pip from .venv location
```

### 0.3 Install/Update Requirements (if needed)

```bash
# Install all dependencies
pip install -r requirements.txt

# Install Unsloth from GitHub (for GPU optimization)
pip install git+https://github.com/unslothai/unsloth.git
```

**Time estimate:** 5-10 minutes (first time only)


## STEP 1: Check GPU Configuration

### 1.1 Display Available GPUs

```bash
python -m src.gpu_config --check
```

**Expected output:**
```
GPU 0: NVIDIA RTX A6000 (46.0 GB)
GPU 1: NVIDIA RTX A6000 (46.0 GB)
...
```

**If no GPUs found:** Check CUDA installation, nvidia-smi, torch.cuda.is_available()

### 1.2 Get Training Recommendations

```bash
python -m src.gpu_config --recommend
```

**Expected output:**
```
Recommended Training Configuration
====================================
num_gpus        : 2
total_memory_gb : 92.0
batch_size      : 8
grad_accum      : 2
device_map      : auto
note            : Multi-GPU training (2 GPUs)
```

### 1.3 Estimate Memory for 7B Model

```bash
python -m src.gpu_config --estimate 7
```

**Expected output:**
```
Memory Estimation for 7.0B parameter model:
  model_weights_gb: 3.26
  optimizer_state_gb: 1.63
  activations_gb: 0.03
  total_gb: 4.92
  note: 4-bit quantization, batch_size=1
```

---

## STEP 2: Explore Datasets

### 2.1 Examine Raw Reviews

```bash
python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv --analyze
```

**What it does:**
- Loads 5,058 raw reviews from sharh.commeta.uz
- Computes statistics (word count, character count, duplicates)
- Shows sample reviews

**Expected output:**
```
Loaded 5,058 raw reviews
Avg words: 13.55
Avg chars: 96.26
Unique reviews: 5,000+
```

**Time estimate:** 30 seconds

### 2.2 Examine Annotated ABSA Dataset

```bash
python scripts/explore_datasets.py --analyze
```

**What it does:**
- Downloads 6,175 annotated examples from HuggingFace
- Analyzes polarity distribution
- Shows aspect statistics

**Expected output:**
```
Loaded 6,175 annotated examples
Total aspects: ~15,500
Avg aspects per example: 2.53
Polarity distribution:
  Positive: ~64%
  Negative: ~16%
  Neutral: ~20%
```

**Time estimate:** 1-2 minutes (includes HF download)

### 2.3 Merge & Match Datasets (Optional)

```bash
python scripts/explore_datasets.py \
    --raw-file ./data/raw/reviews.csv \
    --merge \
    --fuzzy-threshold 0.95
```

**What it does:**
- Attempts to match raw reviews with annotated examples
- Shows how many matches found

**Expected output:**
```
Fuzzy matching results:
  Exact matches: ~200
  High confidence (>95%): ~150
  Total matched: ~350 / 5,058
```

**Time estimate:** 2-3 minutes

---

## STEP 3: Prepare Dataset for Training

### 3.1 Quick Test (100 Examples)

```bash
python scripts/prepare_complete_dataset.py \
    --max-examples 100 \
    --output-dir ./data/test_processed \
    --val-size 0.1
```

**What it does:**
- Downloads annotated dataset
- Converts to instruction-tuning format (ChatML)
- Creates train/validation splits
- Saves to `./data/test_processed/`

**Expected output:**
```
Loaded 100 examples
Converted: 99 valid, 1 failed
Train: 89 examples
Validation: 10 examples
Saved to ./data/test_processed/
```

**Time estimate:** 2-3 minutes

### 3.2 Full Dataset Preparation

```bash
python scripts/prepare_complete_dataset.py \
    --output-dir ./data/processed \
    --val-size 0.1
```

**What it does:**
- Prepares ALL 6,175 examples
- Creates train/validation splits (5,558 train / 617 val)
- Saves processed data for training

**Expected output:**
```
Loaded 6,175 examples
Converted: 6,174 valid, 1 failed
Train: 5,557 examples
Validation: 618 examples
Saved to ./data/processed/
```

**Time estimate:** 5-10 minutes

**Output files created:**
```
./data/processed/
├── train/
│   ├── dataset_info.json
│   └── state.json
├── validation/
│   ├── dataset_info.json
│   └── state.json
├── metadata.json          # Split info
└── conversion_stats.json  # Conversion results
```

---

## STEP 4: Train a Model

### 4.1 Single GPU Training (Fastest)

**On GPU 0 (RTX A6000, 46GB):**

```bash
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --gpu-id 0 \
    --dataset ./data/processed \
    --batch-size 8 \
    --grad-accum 2 \
    --max-steps 1000 \
    --learning-rate 2e-4 \
    --output-dir ./outputs/qwen2.5-7b-v1 \
    --run-name "qwen2.5-7b-single-gpu"
```

**What it does:**
- Loads Qwen 2.5-7B model (4-bit quantized)
- Applies LoRA adapters (r=16, alpha=32)
- Trains on GPU 0
- Saves checkpoints every 100 steps
- Logs to WandB (disable with `--no-wandb`)

**Expected output:**
```
Found 1 GPU(s) available:
  GPU 0: NVIDIA RTX A6000 (46.0 GB)

Starting training...
Step 10/1000: loss=2.34, grad_norm=0.89
Step 20/1000: loss=2.12, grad_norm=0.76
...
Training complete!
Model saved to ./outputs/qwen2.5-7b-v1/
```

**Time estimate:** 30-40 minutes (single GPU on 6,175 examples)

### 4.2 Multi-GPU Training (Faster, Recommended)

**On GPUs 0 & 1 (92GB total):**

```bash
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --multi-gpu \
    --dataset ./data/processed \
    --batch-size 8 \
    --grad-accum 2 \
    --max-steps 1000 \
    --learning-rate 2e-4 \
    --output-dir ./outputs/qwen2.5-7b-multi \
    --run-name "qwen2.5-7b-multi-gpu"
```

**What it does:**
- Uses both GPUs with DistributedDataParallel
- Effective batch size doubles → faster convergence
- Same model and setup, just distributed

**Expected output:** (same as single-GPU, but 2x faster)

**Time estimate:** 15-20 minutes

### 4.3 Test Training (Small Dataset)

**Quick verification before full training:**

```bash
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --gpu-id 0 \
    --dataset ./data/test_processed \
    --batch-size 4 \
    --grad-accum 1 \
    --max-steps 50 \
    --learning-rate 2e-4 \
    --output-dir ./outputs/qwen2.5-7b-test \
    --run-name "test-run"
```

**Time estimate:** 2-3 minutes

### 4.4 Try Different Models

**Llama 3.1 8B:**
```bash
python scripts/train_unsloth.py \
    --model llama3.1-8b \
    --gpu-id 0 \
    --dataset ./data/processed \
    --max-steps 1000
```

**DeepSeek 7B:**
```bash
python scripts/train_unsloth.py \
    --model deepseek-7b \
    --gpu-id 0 \
    --dataset ./data/processed \
    --max-steps 1000
```

**Mistral 7B:**
```bash
python scripts/train_unsloth.py \
    --model mistral-7b \
    --gpu-id 0 \
    --dataset ./data/processed \
    --max-steps 1000
```

---

## STEP 5: Monitor Training

### 5.1 View Training Logs

```bash
# Show latest training log
Get-Content training_*.log -Tail 20

# Or continuously tail logs
Get-Content training_*.log -Wait
```

### 5.2 WandB Dashboard (If Enabled)

```bash
# Login if not already logged in
wandb login

# Open project in browser
start https://wandb.ai/your-username/uzabsa-llm
```

**Metrics tracked:**
- Training loss
- Validation loss
- Learning rate schedule
- Gradient norm
- Training speed (examples/sec)

### 5.3 Check GPU Usage In Real-Time

**In another terminal:**

```bash
# Windows: Monitor GPU (nvidia-smi once)
nvidia-smi

# Or watch continuously (every 1 second)
while ($true) { nvidia-smi; Start-Sleep -Seconds 1 }
```

**Look for:**
- GPU memory usage (should reach ~40-60% of 46GB)
- GPU utilization (should be 80-95%)
- Temperature (should be <80°C)

---

## STEP 6: Run Inference on Trained Model

### 6.1 Extract Aspects from Text

```python
from src.inference import load_model, extract_aspects
import json

# Load your trained model
model, tokenizer = load_model("./outputs/qwen2.5-7b-v1/merged_model")

# Test sentence
text = "Bu restoranning ovqatlari juda mazali, lekin narxlari qimmat."

# Extract aspects
result = extract_aspects(model, tokenizer, text)

print(json.dumps(result, indent=2, ensure_ascii=False))
```

**Expected output:**
```json
{
  "aspects": [
    {
      "term": "ovqatlari",
      "category": "food",
      "polarity": "positive"
    },
    {
      "term": "narxlari",
      "category": "price",
      "polarity": "negative"
    }
  ]
}
```

### 6.2 Batch Inference (Multiple Texts)

```python
from src.inference import load_model, extract_aspects_batch

model, tokenizer = load_model("./outputs/qwen2.5-7b-v1/merged_model")

texts = [
    "Bu telefon juda yaxshi!",
    "Juda qimmat va xato.",
    "O'rtacha mahsulot.",
]

results = extract_aspects_batch(model, tokenizer, texts, batch_size=4)

for i, result in enumerate(results):
    print(f"Text {i}: {result}")
```

**Time estimate:** 5-10 seconds for 100 texts

### 6.3 Save Inference Results

```python
import json
from pathlib import Path

results = extract_aspects_batch(model, tokenizer, texts)

output_file = Path("./outputs/inference_results.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved to {output_file}")
```

---

## STEP 7: Evaluate Model

### 7.1 Compute Metrics on Validation Set

```bash
python scripts/evaluate.py \
    --model-path ./outputs/qwen2.5-7b-v1/merged_model \
    --test-data ./data/processed \
    --output-file ./outputs/qwen2.5-7b-v1/evaluation_results.json
```

**What it computes:**
- Aspect Term Extraction (ATE) F1 score
- Aspect-Polarity Pair F1 score
- Sentiment classification accuracy
- Precision, Recall per metric
- Per-class breakdown

**Expected output:**
```
Evaluating model on 618 validation examples...

Aspect Term Extraction (ATE):
  Precision: 0.72
  Recall: 0.68
  F1: 0.70

Aspect-Polarity Pairs:
  Precision: 0.65
  Recall: 0.61
  F1: 0.63

Sentiment Classification:
  Accuracy: 0.81
  Macro F1: 0.78

Results saved to ./outputs/qwen2.5-7b-v1/evaluation_results.json
```

**Time estimate:** 5-10 minutes

### 7.2 Compare Multiple Models

```bash
# Train and evaluate 3-4 models, then compare

for model in "qwen2.5-7b" "llama3.1-8b" "deepseek-7b"
do
  echo "Training $model..."
  python scripts/train_unsloth.py \
    --model $model \
    --gpu-id 0 \
    --dataset ./data/processed \
    --max-steps 1000 \
    --output-dir "./outputs/$model"
  
  echo "Evaluating $model..."
  python scripts/evaluate.py \
    --model-path "./outputs/$model/merged_model" \
    --test-data ./data/processed
done
```

**Create comparison table:**

```python
import json
import pandas as pd

results = {}
for model in ["qwen2.5-7b", "llama3.1-8b", "deepseek-7b"]:
    with open(f"./outputs/{model}/evaluation_results.json") as f:
        results[model] = json.load(f)

df = pd.DataFrame({
    model: {
        "ATE F1": results[model]["ate_f1"],
        "Pair F1": results[model]["pair_f1"],
        "Sentiment Acc": results[model]["sentiment_accuracy"]
    }
    for model in results
}).T

print(df.to_string())
```

---

## STEP 8: Prepare Results for Research Paper

### 8.1 Update Research Log

Open `RESEARCH_LOG.md` and add new entries:

```markdown
## LOG 016 — Experiment 1 Results: Model Comparison
Date: Feb 2026

### Results
| Model | ATE F1 | Pair F1 | Sent. Acc | Training Time |
|-------|--------|---------|-----------|-----------------|
| Qwen 2.5-7B | 0.70 | 0.63 | 0.81 | 35 min |
| Llama 3.1-8B | 0.68 | 0.61 | 0.79 | 38 min |
| DeepSeek-7B | 0.65 | 0.58 | 0.76 | 40 min |

### Key Findings
- Qwen 2.5-7B achieves best F1 score
- All models show reasonable performance
- Training time comparable across architectures
```

### 8.2 Generate Tables for Paper

```python
import pandas as pd
import json

# Load all results
results = {}
for model_dir in Path("./outputs").glob("*/"):
    model_name = model_dir.name
    with open(model_dir / "evaluation_results.json") as f:
        results[model_name] = json.load(f)

# Create Table 2: Model Comparison
df = pd.DataFrame(results).T
print("Table 2: Model Comparison Results")
print(df[["ate_precision", "ate_recall", "ate_f1", "pair_f1"]])

# Save as LaTeX
latex_table = df.to_latex()
with open("./outputs/table2_model_comparison.tex", 'w') as f:
    f.write(latex_table)
```

### 8.3 Generate Figures

```python
import matplotlib.pyplot as plt

# Figure 2: F1 Scores by Model
models = list(results.keys())
f1_scores = [results[m]["pair_f1"] for m in models]

plt.figure(figsize=(10, 6))
plt.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel("Aspect-Polarity Pair F1")
plt.title("Model Comparison: ABSA Performance")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("./outputs/figure2_model_comparison.png", dpi=300)
plt.show()
```

---

## STEP 9: Save and Merge LoRA Adapters

### 9.1 Merge LoRA Adapters with Base Model

The training script does this automatically (`save_method: "merged"`), but you can also do it manually:

```python
from src.inference import load_model
from pathlib import Path

# Load trained model (with merged adapters)
model, tokenizer = load_model("./outputs/qwen2.5-7b-v1/merged_model")

# Model is ready for inference (no LoRA needed)
```

### 9.2 Convert to GGUF Format (Optional, for deployment)

```bash
# GGUF conversion happens during training if configured
# Check outputs for model_gguf/ directory

ls ./outputs/qwen2.5-7b-v1/
# Expected:
# - merged_model/
# - model_gguf/           # GGUF format (7B -> ~4-5 GB)
# - lora_adapters/        # LoRA adapter weights
```

---

## COMPLETE END-TO-END EXAMPLE (Copy & Run)

```bash
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Check GPU
python -m src.gpu_config --check
python -m src.gpu_config --recommend

# 3. Explore data
python scripts/explore_datasets.py --analyze

# 4. Prepare dataset (full)
python scripts/prepare_complete_dataset.py --output-dir ./data/processed

# 5. Train model (single GPU, 1000 steps)
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --gpu-id 0 `
    --dataset ./data/processed `
    --batch-size 8 `
    --grad-accum 2 `
    --max-steps 1000 `
    --output-dir ./outputs/qwen2.5-7b-v1 `
    --run-name "qwen2.5-7b-single-gpu"

# 6. Evaluate
python scripts/evaluate.py `
    --model-path ./outputs/qwen2.5-7b-v1/merged_model `
    --test-data ./data/processed

# 7. Test inference
python -c "
from src.inference import load_model, extract_aspects
import json

model, tokenizer = load_model('./outputs/qwen2.5-7b-v1/merged_model')
text = 'Bu restoranning ovqatlari juda mazali, lekin narxlari qimmat.'
result = extract_aspects(model, tokenizer, text)
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Solution: Reduce batch size
python scripts/train_unsloth.py \
    --batch-size 4 \
    --grad-accum 4 \
    ...
```

### Issue: Slow Training

```bash
# Solution: Use multi-GPU
python scripts/train_unsloth.py \
    --multi-gpu \
    --batch-size 8 \
    ...
```

### Issue: Model not found / Download fails

```bash
# Check internet connection
# Clear HuggingFace cache:
rm -r ~/.cache/huggingface/hub/

# Re-download
python scripts/train_unsloth.py ...
```

### Issue: Evaluation script not found

```bash
# Check file exists
ls scripts/evaluate.py

# If not, it may need to be created - use inference.py directly
python -c "from src.inference import load_model; ..."
```

---

## Timeline Estimate

| Step | Time |
|------|------|
| 1. GPU check | 30 sec |
| 2. Dataset exploration | 2-3 min |
| 3. Data preparation (100) | 2-3 min |
| 3b. Data preparation (full) | 5-10 min |
| 4a. Test train (50 steps) | 2-3 min |
| 4b. Full train (1000 steps, single GPU) | 30-40 min |
| 4c. Full train (1000 steps, dual GPU) | 15-20 min |
| 5. Inference | <1 min |
| 6. Evaluation | 5-10 min |
| **TOTAL (1 model, single GPU)** | **~1 hour** |
| **TOTAL (4 models, single GPU)** | **~4-5 hours** |
| **TOTAL (4 models, dual GPU)** | **~2-3 hours** |

---

## Next Steps After Running

1. Compare results across models in results table
2. Run ablation studies (LoRA rank, prompt language)
3. Document findings in RESEARCH_LOG.md (LOG 016+)
4. Prepare paper with generated tables and figures
5. Push to GitHub with comprehensive commit message

Good luck! 🚀
