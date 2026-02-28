# UzABSA-LLM

**Fine-tuning Open-Source LLMs for Uzbek Aspect-Based Sentiment Analysis**

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [GPU Configuration](#gpu-configuration)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Multi-Domain Annotation Pipeline](#multi-domain-annotation-pipeline)
- [Publishing Models to HuggingFace Hub](#publishing-models-to-huggingface-hub)
- [Completed Experiments](#completed-experiments)
- [Supported Models](#supported-models)
- [Training Arguments Reference](#training-arguments-reference)
- [Troubleshooting](#troubleshooting)
- [Research Paper Support](#research-paper-support)
- [Citation](#license--citation)

---

## Overview

UzABSA-LLM fine-tunes open-source Large Language Models on Uzbek Aspect-Based Sentiment Analysis (ABSA) tasks using QLoRA (4-bit quantized LoRA). Models learn to extract aspect terms, categories, and sentiment polarities from Uzbek text reviews, outputting structured JSON.

**Datasets:**
- **Annotated ABSA**: [Sanatbek/aspect-based-sentiment-analysis-uzbek](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek) — 6,175 examples (SemEVAL 2014 format)
- **Raw Reviews**: 5,058 unannotated reviews from sharh.commeta.uz for future semi-supervised learning

**Key Technologies:**
- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster training, 80% less memory
- QLoRA — 4-bit NormalFloat quantization + Low-Rank Adaptation
- HuggingFace Transformers, TRL (SFTTrainer), PEFT, bitsandbytes

**Fine-tuned Models on HuggingFace Hub:** [Sanatbek/UzABSA-LLM](https://huggingface.co/Sanatbek/UzABSA-LLM/tree/main)

---

## Project Structure

```
UzABSA-LLM/
├── src/                              # Core package
│   ├── __init__.py                   # Package exports
│   ├── data_prep.py                  # HuggingFace dataset loading & ChatML formatting
│   ├── dataset_utils.py              # Raw CSV loading, cleaning, stats
│   ├── format_converter.py           # SemEVAL 2014 format conversion
│   ├── inference.py                  # Model loading, single & batch inference
│   ├── evaluation.py                 # ATE/ASC/E2E metrics computation
│   ├── gpu_config.py                 # GPU detection & optimization
│   └── training_metrics.py           # Training callbacks & curve plotting
├── scripts/
│   ├── train_unsloth.py              # Main training script (QLoRA + Unsloth)
│   ├── prepare_complete_dataset.py   # End-to-end data preparation pipeline
│   ├── explore_datasets.py           # Dataset exploration & analysis
│   ├── evaluate.py                   # Model evaluation script
│   ├── annotate_reviews.py           # Batch ABSA annotation with checkpoint support
│   ├── llm_judge.py                  # LLM-as-Judge quality scoring (OpenAI/Anthropic)
│   ├── assemble_dataset.py           # Final dataset assembly with quality tiers
│   └── push_to_hub.py                # Upload models to HuggingFace Hub
├── configs/
│   └── training_config.yaml          # Hyperparameter presets
├── data/
│   ├── raw/reviews.csv               # 5,058 raw reviews
│   ├── raw/absa_subcategories.json   # Aspect taxonomy (119 subcategories)
│   ├── raw/business_categories.json  # Business domain mapping (630 businesses)
│   └── processed/                    # Processed train/val datasets
├── outputs/my_run/                   # Trained models & experiment artifacts
├── RESEARCH_LOG.md                   # Detailed methodology & experiment logs
├── requirements.txt                  # Dependencies
├── setup.py                          # Package setup & CLI entry points
└── LICENSE
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- NVIDIA GPU with 24+ GB VRAM (RTX A6000 recommended)

### Step 1: Clone & Create Environment

```powershell
git clone https://github.com/yourusername/UzABSA-LLM.git
cd UzABSA-LLM

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
# source .venv/bin/activate
```

### Step 2: Install PyTorch with CUDA

This step is **critical** — default `pip install torch` installs CPU-only.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Install Unsloth

```powershell
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 5: (Optional) WandB Setup

```powershell
wandb login
```

### Verify Complete Setup

```powershell
python -c "
import torch, transformers
from src.gpu_config import get_gpu_info
print('PyTorch', torch.__version__)
print('Transformers', transformers.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'GPUs: {torch.cuda.device_count()}')
print('All imports OK')
"
```

---

## GPU Configuration

### Check Available GPUs

```powershell
python -m src.gpu_config --check        # Display GPU status
python -m src.gpu_config --recommend    # Get training recommendations  
python -m src.gpu_config --estimate 7   # Estimate 7B model memory (~4.9 GB)
```

### Recommended Batch Sizes

| GPU | VRAM | batch_size | grad_accum | Effective Batch |
|-----|------|-----------|------------|-----------------|
| RTX A6000 (×1) | 46 GB | 4–8 | 2–4 | 8–16 |
| RTX A6000 (×2) | 92 GB | 8 | 2 | 32 (DDP) |
| RTX 4090 | 24 GB | 2 | 4 | 8 |
| RTX 3090 | 24 GB | 1 | 8 | 8 |

### Memory Optimization (enabled by default)

- 4-bit NF4 quantization via bitsandbytes
- Gradient checkpointing (Unsloth optimized)
- BFloat16 mixed precision
- 8-bit AdamW optimizer

---

## Dataset

### Explore Data

```powershell
# Examine raw reviews + annotated dataset
python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv --analyze
```

### Prepare for Training

```powershell
# Quick test (100 examples, <1 min)
python scripts/prepare_complete_dataset.py --max-examples 100 --output-dir ./data/test_processed

# Full dataset (6,175 examples, ~5 min)
python scripts/prepare_complete_dataset.py --output-dir ./data/processed
```

This downloads the annotated dataset from HuggingFace, converts it to ChatML instruction-response format, and creates train/validation splits.

### Dataset Statistics

| Split | Sentences | Aspect Terms | Aspect Categories |
|-------|----------:|-------------:|------------------:|
| Train (90%) | 5,480 | 6,574 | 6,857 |
| Validation (10%) | 609 | 755 | 790 |
| **Total** | **6,089** | **7,329** | **7,647** |

**Polarity distribution:** ~64% positive, 16% negative, 20% neutral  
**Language:** 92.8% Uzbek Latin, 5.2% Russian Cyrillic, 2.0% Uzbek Cyrillic  
**Categories:** ovqat (food), muhit (ambiance), xizmat (service), narx (price), boshqalar (misc)

### Data Format

**Input prompt (Uzbek):**
```
Siz o'zbek tilida matnlardan aspektlarni aniqlash bo'yicha mutaxassissiz...
Matn: "Bu restoranning ovqatlari juda mazali, lekin narxlari qimmat."
```

**Expected output (JSON):**
```json
{"aspects": [{"term": "ovqatlari", "category": "ovqat", "polarity": "positive"},
             {"term": "narxlari", "category": "narx", "polarity": "negative"}]}
```

---

## Training

### Quick Test (2–3 min)

```powershell
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/test_processed `
    --max-steps 50 `
    --output-dir ./outputs/test
```

### Full Training — Single GPU

```powershell
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --gpu-id 0 `
    --dataset ./data/processed `
    --batch-size 4 `
    --grad-accum 4 `
    --max-steps 1000 `
    --learning-rate 2e-4 `
    --output-dir ./outputs/my_run `
    --run-name "qwen2.5-7b-v1"
```

### Full Training — Multi-GPU

```powershell
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --multi-gpu `
    --dataset ./data/processed `
    --batch-size 8 `
    --grad-accum 2 `
    --max-steps 1000 `
    --output-dir ./outputs/my_run
```

### Train Multiple Models (Experiment Loop)

```powershell
foreach ($model in @("qwen2.5-7b", "llama3.1-8b", "deepseek-7b", "mistral-7b")) {
    python scripts/train_unsloth.py `
        --model $model `
        --gpu-id 0 `
        --dataset ./data/processed `
        --max-steps 1000 `
        --output-dir ./outputs/my_run `
        --run-name $model
}
```

### Training Output Files

| File | Purpose |
|------|---------|
| `merged_model/` | Full fine-tuned model (ready for inference) |
| `lora_adapters/` | LoRA weights only (smaller, for resuming) |
| `checkpoint-*/` | Training checkpoints |
| `training_curves.png` | Loss curves (300 DPI, paper-ready) |
| `lr_schedule.png` | Learning rate schedule plot |
| `gpu_memory.png` | GPU memory usage plot |
| `training_history.json` | Per-step metrics (loss, LR, grad norm, GPU mem) |
| `training_history.csv` | Same metrics as CSV |
| `experiment_summary.json` | Full config + results (reproducibility) |

---

## Inference

### Single Text

```python
from src.inference import load_model, extract_aspects
import json

model, tokenizer = load_model("./outputs/my_run/uzabsa_qwen2.5-7b_.../merged_model")
text = "Bu restoranning ovqatlari juda mazali, lekin narxlari qimmat."
result = extract_aspects(model, tokenizer, text)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### Batch Inference

```python
from src.inference import extract_aspects_batch

texts = [
    "Bu telefon juda yaxshi!",
    "Juda qimmat va xato.",
    "O'rtacha mahsulot.",
]
results = extract_aspects_batch(model, tokenizer, texts, batch_size=8)
```

---

## Evaluation

### Quick Test (5 samples, ~30 sec)

```powershell
python scripts/evaluate.py `
    --model-path ./outputs/my_run/<experiment>/merged_model `
    --test-data ./data/processed `
    --output-dir ./outputs/my_run/<experiment> `
    --max-samples 5
```

### Full Evaluation — All Three Models

```powershell
# Qwen 2.5-7B
python scripts/evaluate.py `
    --model-path ./outputs/my_run/uzabsa_qwen2.5-7b_20260222_001629/merged_model `
    --test-data ./data/processed `
    --output-dir ./outputs/my_run/uzabsa_qwen2.5-7b_20260222_001629

# Llama 3.1-8B
python scripts/evaluate.py `
    --model-path ./outputs/my_run/uzabsa_llama3.1-8b_20260222_182459/merged_model `
    --test-data ./data/processed `
    --output-dir ./outputs/my_run/uzabsa_llama3.1-8b_20260222_182459

# DeepSeek-R1-7B
python scripts/evaluate.py `
    --model-path ./outputs/my_run/uzabsa_deepseek-7b/merged_model `
    --test-data ./data/processed `
    --output-dir ./outputs/my_run/uzabsa_deepseek-7b
```

Each run takes ~25 min on the full 609 validation examples. Results are saved to `eval_results_YYYYMMDD_HHMMSS.json` in the output directory.

**Metrics computed:**
- **ATE** (Aspect Term Extraction): Precision, Recall, F1 — exact & partial match
- **ASC** (Aspect Sentiment Classification): Accuracy, Macro-F1
- **E2E-ABSA** (End-to-End): Aspect-polarity pair F1 (primary metric)
- **JSON parse success rate** — instruction-following reliability

---

## Completed Experiments

All experiments used identical configuration: LoRA r=16, alpha=32, dropout=0.05, lr=2e-4 (cosine), effective batch=16, 1000 steps, seed=42, single RTX A6000 (48 GB).

### Experiment 1: Qwen 2.5-7B (Feb 22, 2026)

| Metric | Value |
|--------|-------|
| Model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Initial → Final train loss | 3.0409 → 0.4010 (−86.8%) |
| Best eval loss | 0.3656 (step 1000) |
| Training compute time | **49 min** |
| Total wall time | **4.7 hours** (incl. eval/save) |
| Throughput | 5.44 samples/sec |
| GPU memory | ~5.4 GB allocated, ~7.8 GB reserved |

### Experiment 2: Llama 3.1-8B (Feb 22, 2026)

| Metric | Value |
|--------|-------|
| Model | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| Initial → Final train loss | 2.1197 → 0.2868 (−86.5%) |
| Best eval loss | **0.2785** (step 600) |
| Training compute time | **60 min** |
| Total wall time | **7.6 hours** (incl. eval/save) |
| Throughput | 4.46 samples/sec |
| GPU memory | ~5.6 GB allocated, ~7.6 GB reserved |

### Experiment 3: DeepSeek-R1-7B (Feb 22–23, 2026)

| Metric | Value |
|--------|-------|
| Model | `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` |
| Initial → Final train loss | 3.2900 → 0.2330 (−92.9%) |
| Best eval loss | 0.3363 (step 1000) |
| Training compute time | **62 min** |
| Total wall time | ~**8+ hours** (incl. eval/save) |
| Throughput | 4.28 samples/sec |
| GPU memory | ~5.5 GB allocated, ~8.5 GB reserved |

### 3-Model Comparison (Training Loss)

| Metric | Qwen 2.5-7B | DeepSeek-R1-7B | Llama 3.1-8B |
|--------|:-----------:|:--------------:|:------------:|
| Best eval loss | 0.3656 | 0.3363 | **0.2785** |
| Best eval step | 1000 | 1000 | **600** |
| Loss reduction | 86.8% | **92.9%** | 86.5% |
| Training speed | **5.44 s/s** | 4.28 s/s | 4.46 s/s |
| GPU memory | ~5.4 GB | ~5.5 GB | ~5.6 GB |
| Convergence | Gradual | Gradual | Fast |

> **Note:** Rankings based on cross-entropy loss only. ABSA evaluation (P/R/F1) is pending — see RESEARCH_LOG.md for full details. DeepSeek shares the Qwen-7B architecture but achieves 8% lower eval loss, likely due to R1 reasoning distillation.

---

## Supported Models

All models use 4-bit NF4 quantization via Unsloth/bitsandbytes.

| Shorthand | HuggingFace Path | Params | VRAM |
|-----------|-----------------|--------|------|
| `qwen2.5-7b` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 7B | ~6 GB |
| `qwen2.5-14b` | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` | 14B | ~10 GB |
| `qwen2.5-32b` | `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` | 32B | ~20 GB |
| `llama3-8b` | `unsloth/llama-3-8b-Instruct-bnb-4bit` | 8B | ~6 GB |
| `llama3.1-8b` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | 8B | ~6 GB |
| `llama3.2-3b` | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | 3B | ~3 GB |
| `deepseek-7b` | `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` | 7B | ~6 GB |
| `deepseek-14b` | `unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit` | 14B | ~10 GB |
| `mistral-7b` | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | 7B | ~6 GB |
| `gemma2-9b` | `unsloth/gemma-2-9b-it-bnb-4bit` | 9B | ~7 GB |

---

## Training Arguments Reference

```powershell
python scripts/train_unsloth.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `qwen2.5-7b` | Model shorthand (see table above) |
| `--model-path` | — | Custom HuggingFace path (overrides `--model`) |
| `--dataset` | `./data/processed` | Path to processed dataset |
| `--max-seq-length` | 2048 | Maximum sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha (scaling = alpha/r) |
| `--learning-rate` | 2e-4 | Learning rate |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--max-steps` | 1000 | Max training steps (−1 for epoch-based) |
| `--epochs` | 3 | Number of epochs (if max-steps=−1) |
| `--gpu-id` | — | Specific GPU to use (0, 1, 2, 3) |
| `--multi-gpu` | False | Enable DistributedDataParallel |
| `--device-map` | `auto` | Device mapping (auto, cuda:0, cpu) |
| `--output-dir` | `./outputs` | Output directory |
| `--run-name` | — | Experiment name |
| `--no-wandb` | False | Disable W&B logging |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"CUDA not found"** | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| **CUDA out of memory** | Reduce `--batch-size` to 2 or 1, increase `--grad-accum` |
| **Very slow training** | Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |
| **Module not found** | Activate venv: `.\.venv\Scripts\Activate.ps1` |
| **Unsloth install fails** | `pip install --upgrade pip` then retry |
| **"AttrsDescriptor" error** | Triton mismatch: `pip install triton-windows==3.2.0.post19` |
| **HuggingFace download fails** | Check internet; set `HF_HOME` env var |
| **Empty logs** | Check: `python -c "from unsloth import FastLanguageModel"` |

---

## Multi-Domain Annotation Pipeline

This project includes a complete 3-layer annotation pipeline for creating a silver-standard multi-domain Uzbek ABSA dataset from 5,038 raw reviews spanning 23 business domains.

### Architecture

```
Layer 1: Fine-tuned model (local GPU)  →  Extracts aspects + sentiments from raw reviews
Layer 2: LLM-as-Judge (API: GPT-4o-mini)  →  Scores annotation quality (1–5) on stratified sample
Layer 3: Dataset assembly  →  Filters by quality tiers, exports HuggingFace-compatible dataset
```

### Layer 1 — Batch Annotation

Annotate all 5,038 reviews using any of the three fine-tuned models. The script automatically resumes from checkpoints if interrupted.

**Qwen 2.5-7B** (recommended — 100% JSON parse rate, highest ATE F1):
```powershell
python scripts/annotate_reviews.py `
    --model-path ./outputs/my_run/uzabsa_qwen2.5-7b_20260222_001629/merged_model
```

**Llama 3.1-8B** (best sentiment accuracy, 95.9% JSON parse rate):
```powershell
python scripts/annotate_reviews.py `
    --model-path ./outputs/my_run/uzabsa_llama3.1-8b_20260222_182459/merged_model
```

**DeepSeek-R1-7B** (95.4% JSON parse rate):
```powershell
python scripts/annotate_reviews.py `
    --model-path ./outputs/my_run/uzabsa_deepseek-7b/merged_model
```

| Option | Description |
|--------|-------------|
| `--max-samples N` | Annotate only N reviews (for testing) |
| `--reviews-csv PATH` | Custom reviews CSV (default: `./data/raw/reviews.csv`) |
| `--categories-json PATH` | Business categories JSON (default: `./data/raw/business_categories.json`) |
| `--output-dir PATH` | Output directory (default: `./data/annotated`) |
| `--use-english` | Use English prompts instead of Uzbek |

**Output:** `data/annotated/reviews_annotated.json`, `annotation_stats.json`, `domain_distribution.json`

### Layer 2 — LLM-as-Judge Quality Scoring

Scores a stratified sample (~300 reviews across 23 domains) using an external LLM. The judge reads each review alongside the model's prediction and rates quality on 5 dimensions (1–5 scale): Completeness, Accuracy, Sentiment, Relevance, Overall.

```powershell
# Set API key first
$env:OPENAI_API_KEY = "sk-..."

# Run with GPT-4o-mini (OpenAI)
python scripts/llm_judge.py `
    --annotations ./data/annotated/reviews_annotated.json `
    --provider openai `
    --model gpt-4o-mini `
    --sample-size 300 `
    --output-dir ./data/judged
```

**Alternative providers:**
```powershell
# Anthropic Claude
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python scripts/llm_judge.py `
    --annotations ./data/annotated/reviews_annotated.json `
    --provider anthropic `
    --model claude-3-5-haiku-20241022 `
    --sample-size 300 `
    --output-dir ./data/judged

# Any OpenAI-compatible API (e.g., Together, Groq, local vLLM)
$env:OPENAI_API_KEY = "your-api-key"
python scripts/llm_judge.py `
    --annotations ./data/annotated/reviews_annotated.json `
    --provider openai `
    --model meta-llama/Meta-Llama-3.1-70B-Instruct `
    --api-base https://api.together.xyz/v1 `
    --sample-size 300 `
    --output-dir ./data/judged
```

| Option | Description |
|--------|-------------|
| `--sample-size N` | Number of reviews to judge (default: 300) |
| `--delay SECS` | Delay between API calls (default: 0.5) |
| `--seed N` | Random seed for stratified sampling (default: 42) |
| `--no-sample` | Judge ALL annotations (no sampling) |

**Output:** `data/judged/judge_results.json`, `judge_report.json`, `judge_summary.txt`

### Layer 3 — Final Dataset Assembly

Merges annotations with judge scores and applies quality thresholds:
- **Include** (overall score ≥ 3.5) — high-quality annotations
- **Flag** (2.5–3.5) — needs human review
- **Exclude** (< 2.5) — low quality

```powershell
python scripts/assemble_dataset.py `
    --annotations ./data/annotated/reviews_annotated.json `
    --judge-results ./data/judged/judge_results.json `
    --output-dir ./data/final_dataset
```

Can also run **without judge results** (all annotations become "unjudged" tier):
```powershell
python scripts/assemble_dataset.py `
    --annotations ./data/annotated/reviews_annotated.json `
    --output-dir ./data/final_dataset
```

| Option | Description |
|--------|-------------|
| `--include-threshold N` | Minimum judge score to include (default: 3.5) |

**Output:**
| File | Description |
|------|-------------|
| `uzbek_multi_domain_absa_full.json` | All annotations + judge scores |
| `uzbek_multi_domain_absa_silver.json` | Included + unjudged (silver standard) |
| `uzbek_multi_domain_absa_approved.json` | Judge-approved only (score ≥ 3.5) |
| `uzbek_multi_domain_absa.jsonl` | JSONL format for HuggingFace |
| `dataset_stats.json` | Comprehensive statistics |

---

## Publishing Models to HuggingFace Hub

All three fine-tuned models are published to a single HuggingFace repository using **branches**:

| Branch | Model | Size (merged + LoRA) |
|--------|-------|:----:|
| [`main`](https://huggingface.co/Sanatbek/UzABSA-LLM) | Project README + comparison | — |
| [`qwen2.5-7b`](https://huggingface.co/Sanatbek/UzABSA-LLM/tree/qwen2.5-7b) | Qwen 2.5-7B | 14.2 GB + 154 MB |
| [`llama3.1-8b`](https://huggingface.co/Sanatbek/UzABSA-LLM/tree/llama3.1-8b) | Llama 3.1-8B | 15.0 GB + 160 MB |
| [`deepseek-r1-7b`](https://huggingface.co/Sanatbek/UzABSA-LLM/tree/deepseek-r1-7b) | DeepSeek-R1-Distill-Qwen-7B | 14.2 GB + 154 MB |

### How to Upload

```powershell
# 1. Login to HuggingFace (one-time setup)
pip install huggingface_hub
huggingface-cli login

# 2. Dry-run first to preview what will be uploaded
python scripts/push_to_hub.py --all --dry-run

# 3. Upload everything (main README + all 3 models)
python scripts/push_to_hub.py --all

# Or upload one model at a time
python scripts/push_to_hub.py --branch main          # Project README only
python scripts/push_to_hub.py --branch qwen2.5-7b     # Qwen model
python scripts/push_to_hub.py --branch llama3.1-8b     # Llama model
python scripts/push_to_hub.py --branch deepseek-r1-7b  # DeepSeek model
```

### How Others Load Your Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pick a model branch
model = AutoModelForCausalLM.from_pretrained(
    "Sanatbek/UzABSA-LLM",
    revision="qwen2.5-7b",   # or "llama3.1-8b" or "deepseek-r1-7b"
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Sanatbek/UzABSA-LLM", revision="qwen2.5-7b")
```

---

## Research Paper Support

Detailed methodology, experiment logs, and reproducibility metadata are in [RESEARCH_LOG.md](RESEARCH_LOG.md) (25 entries covering dataset analysis, model selection, QLoRA config, evaluation metrics, hardware setup, experiment plans, and results).

### Planned Experiments (see RESEARCH_LOG.md)

| # | Experiment | Status |
|---|-----------|--------|
| 1 | Model comparison (Qwen/Llama/DeepSeek/Mistral at ~7B) | Qwen, Llama & DeepSeek done |
| 2 | LoRA rank ablation (r ∈ {4, 8, 16, 32, 64}) | Planned |
| 3 | Prompt language ablation (Uzbek vs English) | Planned |
| 4 | Model scale comparison (3B vs 7B vs 14B) | Planned |
| 5 | Zero-shot / few-shot baselines | Planned |

### Paper-Ready Outputs

- `training_curves.png` — 300 DPI loss plots
- `experiment_summary.json` — full reproducibility metadata
- `training_history.csv` — per-step metrics for custom plotting
- W&B dashboard — interactive experiment tracking

---

## Citation

```bibtex
@misc{uzabsa-llm-2026,
  title={UzABSA-LLM: Fine-tuned Large Language Models for Uzbek Aspect-Based Sentiment Analysis},
  author={Matlatipov, Sanatbek},
  year={2026},
  url={https://huggingface.co/Sanatbek/UzABSA-LLM},
  note={QLoRA fine-tuned Qwen 2.5-7B, Llama 3.1-8B, and DeepSeek-R1-Distill-Qwen-7B}
}
```

### Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) — Optimized LLM training
- [Hugging Face](https://huggingface.co/) — Transformers ecosystem
- [Sanatbek](https://huggingface.co/Sanatbek) — Uzbek ABSA dataset
