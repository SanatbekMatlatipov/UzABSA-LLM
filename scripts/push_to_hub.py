#!/usr/bin/env python3
"""
Push UzABSA-LLM fine-tuned models to HuggingFace Hub.

Repository structure (single repo, multiple branches):
  - main branch:        Project README + evaluation comparison (no weights)
  - qwen2.5-7b branch:  Merged model + LoRA adapters + model card
  - llama3.1-8b branch: Merged model + LoRA adapters + model card
  - deepseek-r1-7b branch: Merged model + LoRA adapters + model card

Usage:
  # Push everything (all branches)
  python scripts/push_to_hub.py --all

  # Push only the main branch (README + comparison)
  python scripts/push_to_hub.py --branch main

  # Push a specific model branch
  python scripts/push_to_hub.py --branch qwen2.5-7b

  # Dry-run (shows what would be uploaded without actually uploading)
  python scripts/push_to_hub.py --all --dry-run

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   # or set HF_TOKEN env variable
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID = "Sanatbek/UzABSA-LLM"

MODELS = {
    "qwen2.5-7b": {
        "branch": "qwen2.5-7b",
        "local_dir": "outputs/my_run/uzabsa_qwen2.5-7b_20260222_001629",
        "display_name": "UzABSA Qwen 2.5-7B",
        "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "base_model_full": "Qwen/Qwen2.5-7B-Instruct",
        "architecture": "Qwen2ForCausalLM",
        "params": "7.6B",
        "merged_size": "14.2 GB",
        "lora_size": "154 MB",
    },
    "llama3.1-8b": {
        "branch": "llama3.1-8b",
        "local_dir": "outputs/my_run/uzabsa_llama3.1-8b_20260222_182459",
        "display_name": "UzABSA Llama 3.1-8B",
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "base_model_full": "meta-llama/Llama-3.1-8B-Instruct",
        "architecture": "LlamaForCausalLM",
        "params": "8.0B",
        "merged_size": "15.0 GB",
        "lora_size": "160 MB",
    },
    "deepseek-r1-7b": {
        "branch": "deepseek-r1-7b",
        "local_dir": "outputs/my_run/uzabsa_deepseek-7b",
        "display_name": "UzABSA DeepSeek-R1-Distill-Qwen-7B",
        "base_model": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
        "base_model_full": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "architecture": "Qwen2ForCausalLM",
        "params": "7.6B",
        "merged_size": "14.2 GB",
        "lora_size": "154 MB",
    },
}

# Shared training hyperparameters
TRAINING_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": "2e-4",
    "batch_size": 4,
    "grad_accum": 4,
    "effective_batch_size": 16,
    "max_steps": 1000,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "optimizer": "adamw_8bit",
    "precision": "bf16",
    "quantization": "4-bit NF4 (QLoRA)",
    "max_seq_length": 2048,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_eval_results(model_dir: Path) -> dict | None:
    """Load the latest eval_results JSON from the model directory."""
    eval_files = sorted(model_dir.glob("eval_results_*.json"))
    if not eval_files:
        return None
    with open(eval_files[-1]) as f:
        return json.load(f)


def load_experiment_summary(model_dir: Path) -> dict | None:
    """Load experiment_summary.json from the model directory."""
    summary_path = model_dir / "experiment_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model card generators
# ---------------------------------------------------------------------------

def generate_main_readme() -> str:
    """Generate the main branch README with project overview + comparison."""
    root = get_project_root()

    # Load all eval results
    results = {}
    for key, cfg in MODELS.items():
        model_dir = root / cfg["local_dir"]
        results[key] = load_eval_results(model_dir)

    card = f"""---
language:
  - uz
license: apache-2.0
tags:
  - aspect-based-sentiment-analysis
  - absa
  - uzbek
  - nlp
  - qlora
  - unsloth
  - fine-tuned
datasets:
  - Sanatbek/aspect-based-sentiment-analysis-uzbek
pipeline_tag: text-generation
---

# UzABSA-LLM — Uzbek Aspect-Based Sentiment Analysis

Three open-source LLMs fine-tuned for **Aspect-Based Sentiment Analysis (ABSA)** in Uzbek, using QLoRA on [Sanatbek/aspect-based-sentiment-analysis-uzbek](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek) (6,175 examples, SemEVAL 2014 format).

Given an Uzbek text review, each model extracts **aspect terms**, **aspect categories**, and **sentiment polarities** as structured JSON.

## Available Models

Each model is stored on a separate branch. Load with `revision=`:

| Model | Branch | Load Command |
|-------|--------|--------------|
| **Qwen 2.5-7B** | `qwen2.5-7b` | `AutoModelForCausalLM.from_pretrained("{REPO_ID}", revision="qwen2.5-7b")` |
| **Llama 3.1-8B** | `llama3.1-8b` | `AutoModelForCausalLM.from_pretrained("{REPO_ID}", revision="llama3.1-8b")` |
| **DeepSeek-R1-Distill-Qwen-7B** | `deepseek-r1-7b` | `AutoModelForCausalLM.from_pretrained("{REPO_ID}", revision="deepseek-r1-7b")` |

> Each branch contains both the **full merged model** (safetensors) and the **LoRA adapter** (in `lora_adapters/`).

## Evaluation Results

All models evaluated on the same held-out validation set (609 examples).

| Metric | Qwen 2.5-7B | Llama 3.1-8B | DeepSeek-R1-7B | Best |
|--------|:-----------:|:------------:|:--------------:|:----:|
| **ATE Exact F1** | **0.6603** | 0.6549 | 0.6034 | Qwen |
| **ATE Partial F1** | **0.7705** | 0.7591 | 0.7279 | Qwen |
| **Pair F1** | 0.5795 | **0.5805** | 0.5018 | Llama |
| **Sentiment Accuracy** | 0.8777 | **0.8864** | 0.8317 | Llama |
| **Sentiment Macro-F1** | 0.8113 | **0.8435** | 0.7717 | Llama |
| **JSON Parse Rate** | **100.0%** | 95.89% | 95.40% | Qwen |

**Key findings:**
- **Qwen 2.5-7B** leads in aspect term extraction (ATE) and achieves a perfect 100% JSON parse rate — the most reliable for structured output.
- **Llama 3.1-8B** leads in sentiment classification accuracy and end-to-end pair-level F1, making it the best choice when sentiment precision matters most.
- **DeepSeek-R1-Distill-Qwen-7B** trails on all metrics, suggesting R1 distillation doesn't benefit structured ABSA extraction tasks.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the best model (Qwen 2.5-7B) for aspect extraction
model_name = "{REPO_ID}"
revision = "qwen2.5-7b"  # or "llama3.1-8b" or "deepseek-r1-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=revision,
    torch_dtype="auto",
    device_map="auto",
)

# Prepare input (ChatML format)
messages = [
    {{"role": "system", "content": "Siz aspect-based sentiment analysis mutaxassisisiz. Berilgan sharh matnidan aspect termlarni, ularning kategoriyalarini va sentiment polarligini JSON formatida chiqaring."}},
    {{"role": "user", "content": "Sharh: Ovqat mazali, lekin xizmat juda sekin edi."}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

**Expected output:**
```json
{{
  "aspects": [
    {{"aspect": "ovqat", "category": "Taom sifati", "sentiment": "ijobiy"}},
    {{"aspect": "xizmat", "category": "Xizmat ko'rsatish", "sentiment": "salbiy"}}
  ]
}}
```

## Using LoRA Adapters Only

For smaller downloads (~154 MB vs ~14 GB), use the LoRA adapter with the original base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Apply LoRA adapter from the repo
model = PeftModel.from_pretrained(
    base_model,
    "{REPO_ID}",
    revision="qwen2.5-7b",
    subfolder="lora_adapters",
)
```

## Training Details

All three models were trained with **identical hyperparameters** for fair comparison:

| Parameter | Value |
|-----------|-------|
| Framework | Unsloth + HuggingFace TRL (SFTTrainer) |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Learning rate | 2e-4 |
| Effective batch size | 16 (4 × 4 gradient accumulation) |
| Max steps | 1,000 |
| Warmup | 10% |
| Scheduler | Cosine |
| Optimizer | AdamW 8-bit |
| Precision | bf16 |
| Max sequence length | 2,048 tokens |
| Seed | 42 |

**Training data:** [Sanatbek/aspect-based-sentiment-analysis-uzbek](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek) — 6,175 Uzbek restaurant reviews with aspect-level annotations (SemEVAL 2014 format), split 90/10 into 5,480 train / 609 validation.

**Hardware:** NVIDIA RTX A6000 (48 GB VRAM), Windows 11, CUDA 12.4, PyTorch 2.6.0

| Model | Training Time | Final Train Loss | Best Eval Loss |
|-------|:------------:|:----------------:|:--------------:|
| Qwen 2.5-7B | 49 min | 0.401 | 0.366 |
| Llama 3.1-8B | 60 min | 0.287 | 0.279 |
| DeepSeek-R1-7B | 62 min | 0.415 | 0.336 |

## Task Format

**Input:** Uzbek text review + structured system prompt (ChatML)

**Output:** JSON with extracted aspects:
```json
{{
  "aspects": [
    {{
      "aspect": "<aspect term in original language>",
      "category": "<aspect category>",
      "sentiment": "ijobiy | salbiy | neytral"
    }}
  ]
}}
```

## Citation

If you use these models in your research, please cite:

```bibtex
@misc{{uzabsa-llm-2026,
  title={{UzABSA-LLM: Fine-tuned Large Language Models for Uzbek Aspect-Based Sentiment Analysis}},
  author={{Sanatbek}},
  year={{2026}},
  url={{https://huggingface.co/{REPO_ID}}},
  note={{QLoRA fine-tuned Qwen 2.5-7B, Llama 3.1-8B, and DeepSeek-R1-Distill-Qwen-7B}}
}}
```

## Project Repository

Full source code, training scripts, evaluation pipeline, and research log:
**[GitHub: UzABSA-LLM](https://github.com/Sanatbek/UzABSA-LLM)**

## License

Apache 2.0
"""
    return card.strip() + "\n"


def generate_model_readme(model_key: str) -> str:
    """Generate a model card for a specific model branch."""
    cfg = MODELS[model_key]
    root = get_project_root()
    model_dir = root / cfg["local_dir"]

    eval_data = load_eval_results(model_dir) or {}
    exp_data = load_experiment_summary(model_dir) or {}

    # Extract metrics safely
    ate = eval_data.get("aspect_term_extraction", {})
    pairs = eval_data.get("aspect_polarity_pairs", {})
    exact = ate.get("exact_match", {})
    partial = ate.get("partial_match", {})

    ate_exact_f1 = exact.get("f1", "N/A")
    ate_partial_f1 = partial.get("f1", "N/A")
    pair_f1 = pairs.get("pair_f1", "N/A")
    sent_acc = pairs.get("sentiment_accuracy", "N/A")
    sent_f1 = pairs.get("sentiment_macro_f1", "N/A")
    parse_rate = eval_data.get("json_parse_rate", "N/A")

    # Training info
    tr = exp_data.get("training_results", {})
    curves = exp_data.get("training_curves_summary", {})
    final_loss = tr.get("final_train_loss", "N/A")
    best_eval = curves.get("best_eval_loss", "N/A")
    train_time_min = curves.get("total_training_time_min", "N/A")

    # Format numbers
    def fmt(v, decimals=4):
        if isinstance(v, (int, float)):
            return f"{v:.{decimals}f}"
        return str(v)

    card = f"""---
language:
  - uz
license: apache-2.0
base_model: {cfg['base_model_full']}
tags:
  - aspect-based-sentiment-analysis
  - absa
  - uzbek
  - nlp
  - qlora
  - unsloth
  - fine-tuned
  - lora
  - peft
datasets:
  - Sanatbek/aspect-based-sentiment-analysis-uzbek
pipeline_tag: text-generation
model-index:
  - name: {cfg['display_name']}
    results:
      - task:
          type: token-classification
          name: Aspect Term Extraction
        dataset:
          name: Uzbek ABSA (SemEVAL format)
          type: Sanatbek/aspect-based-sentiment-analysis-uzbek
          split: validation
        metrics:
          - type: f1
            value: {fmt(ate_exact_f1)}
            name: ATE Exact F1
          - type: f1
            value: {fmt(ate_partial_f1)}
            name: ATE Partial F1
      - task:
          type: text-classification
          name: Aspect Sentiment Classification
        dataset:
          name: Uzbek ABSA (SemEVAL format)
          type: Sanatbek/aspect-based-sentiment-analysis-uzbek
          split: validation
        metrics:
          - type: accuracy
            value: {fmt(sent_acc)}
            name: Sentiment Accuracy
          - type: f1
            value: {fmt(sent_f1)}
            name: Sentiment Macro-F1
---

# {cfg['display_name']}

A **{cfg['params']}** parameter LLM fine-tuned for **Aspect-Based Sentiment Analysis (ABSA)** in Uzbek using QLoRA. Extracts aspect terms, categories, and sentiment polarities from Uzbek text reviews as structured JSON.

> This is the **`{cfg['branch']}`** branch of [{REPO_ID}](https://huggingface.co/{REPO_ID}). See the [main branch](https://huggingface.co/{REPO_ID}) for a comparison of all models.

## Model Details

| Property | Value |
|----------|-------|
| Base model | [{cfg['base_model_full']}](https://huggingface.co/{cfg['base_model_full']}) |
| Architecture | {cfg['architecture']} |
| Parameters | {cfg['params']} |
| Fine-tuning method | QLoRA (4-bit NF4) |
| Framework | Unsloth + HuggingFace TRL |
| Task | Aspect-Based Sentiment Analysis |
| Language | Uzbek (uz) |
| Merged model size | {cfg['merged_size']} |
| LoRA adapter size | {cfg['lora_size']} |

## Evaluation Results

Evaluated on 609 held-out validation examples:

| Metric | Score |
|--------|:-----:|
| **ATE Exact F1** | {fmt(ate_exact_f1)} |
| **ATE Partial F1** | {fmt(ate_partial_f1)} |
| **Pair F1** (aspect + sentiment) | {fmt(pair_f1)} |
| **Sentiment Accuracy** | {fmt(sent_acc)} |
| **Sentiment Macro-F1** | {fmt(sent_f1)} |
| **JSON Parse Rate** | {parse_rate}% |

## Usage

### Load Merged Model (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{REPO_ID}",
    revision="{cfg['branch']}",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{REPO_ID}", revision="{cfg['branch']}")

messages = [
    {{"role": "system", "content": "Siz aspect-based sentiment analysis mutaxassisisiz. Berilgan sharh matnidan aspect termlarni, ularning kategoriyalarini va sentiment polarligini JSON formatida chiqaring."}},
    {{"role": "user", "content": "Sharh: Ovqat mazali, lekin xizmat juda sekin edi."}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

### Load LoRA Adapter Only (~{cfg['lora_size']})

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "{cfg['base_model_full']}",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{cfg['base_model_full']}")

model = PeftModel.from_pretrained(
    base,
    "{REPO_ID}",
    revision="{cfg['branch']}",
    subfolder="lora_adapters",
)
```

### With Unsloth (2x Faster Inference)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "{REPO_ID}",
    revision="{cfg['branch']}",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | {TRAINING_CONFIG['lora_r']} |
| LoRA alpha | {TRAINING_CONFIG['lora_alpha']} |
| LoRA dropout | {TRAINING_CONFIG['lora_dropout']} |
| Target modules | q, k, v, o, gate, up, down projections |
| Learning rate | {TRAINING_CONFIG['learning_rate']} |
| Effective batch size | {TRAINING_CONFIG['effective_batch_size']} (4 × 4) |
| Max steps | {TRAINING_CONFIG['max_steps']} |
| Warmup | {TRAINING_CONFIG['warmup_ratio']} |
| Scheduler | {TRAINING_CONFIG['scheduler']} |
| Optimizer | {TRAINING_CONFIG['optimizer']} |
| Precision | {TRAINING_CONFIG['precision']} |
| Max seq length | {TRAINING_CONFIG['max_seq_length']} |
| Seed | {TRAINING_CONFIG['seed']} |
| Final train loss | {fmt(final_loss)} |
| Best eval loss | {fmt(best_eval) if isinstance(best_eval, float) else best_eval} |
| Training time | {fmt(train_time_min, 1) if isinstance(train_time_min, (int, float)) else train_time_min} min |

## Training Data

[Sanatbek/aspect-based-sentiment-analysis-uzbek](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek) — 6,175 Uzbek restaurant reviews with aspect-level annotations (SemEVAL 2014 format).

- **Train split:** 5,480 examples
- **Validation split:** 609 examples
- **Format:** ChatML instruction-response pairs

## Hardware

- **GPU:** NVIDIA RTX A6000 (48 GB VRAM)
- **OS:** Windows 11
- **CUDA:** 12.4
- **PyTorch:** 2.6.0

## Repository Contents

This branch contains:

```
├── config.json                  # Model config
├── model-00001-of-00004.safetensors   # Merged weights (shard 1/4)
├── model-00002-of-00004.safetensors   # Merged weights (shard 2/4)
├── model-00003-of-00004.safetensors   # Merged weights (shard 3/4)
├── model-00004-of-00004.safetensors   # Merged weights (shard 4/4)
├── model.safetensors.index.json       # Weight index
├── tokenizer.json               # Tokenizer
├── tokenizer_config.json        # Tokenizer config
├── special_tokens_map.json      # Special tokens
├── lora_adapters/               # LoRA adapter (lightweight alternative)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
└── eval_results.json            # Full evaluation metrics
```

## Citation

```bibtex
@misc{{uzabsa-llm-2026,
  title={{UzABSA-LLM: Fine-tuned Large Language Models for Uzbek Aspect-Based Sentiment Analysis}},
  author={{Sanatbek}},
  year={{2026}},
  url={{https://huggingface.co/{REPO_ID}}},
  note={{{cfg['display_name']} — QLoRA fine-tuned on Uzbek ABSA}}
}}
```

## License

Apache 2.0
"""
    return card.strip() + "\n"


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------

def push_main_branch(repo_id: str, dry_run: bool = False):
    """Push the main branch with project README."""
    from huggingface_hub import HfApi

    api = HfApi()
    readme_content = generate_main_readme()

    print(f"\n{'=' * 60}")
    print(f"  Pushing MAIN branch to {repo_id}")
    print(f"{'=' * 60}")
    print(f"  Content: Project README + evaluation comparison")
    print(f"  Size: ~{len(readme_content) / 1024:.1f} KB")

    if dry_run:
        print("\n  [DRY RUN] Would upload README.md to main branch")
        # Save locally for review
        preview_path = get_project_root() / "outputs" / "hub_preview" / "main_README.md"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(readme_content, encoding="utf-8")
        print(f"  Preview saved to: {preview_path}")
        return

    # Ensure repo exists
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    # Upload README to main branch
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update project README with evaluation comparison",
    )
    print(f"\n  ✓ Main branch updated: https://huggingface.co/{repo_id}")


def push_model_branch(model_key: str, repo_id: str, dry_run: bool = False):
    """Push a model branch with merged weights + LoRA adapters."""
    from huggingface_hub import HfApi

    cfg = MODELS[model_key]
    api = HfApi()
    root = get_project_root()
    model_dir = root / cfg["local_dir"]
    merged_dir = model_dir / "merged_model"
    lora_dir = model_dir / "lora_adapters"
    branch = cfg["branch"]

    print(f"\n{'=' * 60}")
    print(f"  Pushing {cfg['display_name']} to {repo_id} (branch: {branch})")
    print(f"{'=' * 60}")

    # Validate paths
    if not merged_dir.exists():
        print(f"  ERROR: Merged model not found at {merged_dir}")
        return False
    if not lora_dir.exists():
        print(f"  ERROR: LoRA adapters not found at {lora_dir}")
        return False

    # Generate model card
    readme_content = generate_model_readme(model_key)

    # Gather files to upload
    files_to_upload = []

    # 1. Merged model files (root of branch)
    print(f"\n  Merged model files ({cfg['merged_size']}):")
    for f in sorted(merged_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name:45s} {size_mb:>10.1f} MB")
            files_to_upload.append((str(f), f.name))

    # 2. LoRA adapter files (lora_adapters/ subfolder)
    print(f"\n  LoRA adapter files ({cfg['lora_size']}):")
    for f in sorted(lora_dir.iterdir()):
        if f.is_file() and f.name != "README.md":  # Skip default PEFT readme
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    lora_adapters/{f.name:35s} {size_mb:>10.1f} MB")
            files_to_upload.append((str(f), f"lora_adapters/{f.name}"))

    # 3. Eval results (copy to branch root)
    eval_files = sorted(model_dir.glob("eval_results_*.json"))
    if eval_files:
        latest_eval = eval_files[-1]
        print(f"\n  Evaluation results:")
        print(f"    {latest_eval.name} -> eval_results.json")
        files_to_upload.append((str(latest_eval), "eval_results.json"))

    # 4. Experiment summary
    summary_file = model_dir / "experiment_summary.json"
    if summary_file.exists():
        print(f"    experiment_summary.json")
        files_to_upload.append((str(summary_file), "experiment_summary.json"))

    total_files = len(files_to_upload) + 1  # +1 for README
    print(f"\n  Total: {total_files} files")

    if dry_run:
        print(f"\n  [DRY RUN] Would upload {total_files} files to branch '{branch}'")
        # Save model card for review
        preview_path = root / "outputs" / "hub_preview" / f"{branch}_README.md"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(readme_content, encoding="utf-8")
        print(f"  Preview saved to: {preview_path}")
        return True

    # Ensure repo exists
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    # Create branch if it doesn't exist
    try:
        api.create_branch(repo_id=repo_id, branch=branch, repo_type="model")
        print(f"\n  Created branch: {branch}")
    except Exception as e:
        if "already exists" in str(e).lower() or "reference already exists" in str(e).lower():
            print(f"\n  Branch '{branch}' already exists, will update")
        else:
            print(f"\n  Branch creation note: {e}")

    # Upload README first
    print(f"\n  Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Add model card for {cfg['display_name']}",
    )

    # Upload all model files using upload_folder for efficiency
    # First, create a temporary commit message
    print(f"\n  Uploading model files (this may take a while for {cfg['merged_size']})...")

    # Upload merged model folder
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Upload merged model weights for {cfg['display_name']}",
    )
    print(f"  ✓ Merged model uploaded")

    # Upload LoRA adapters to subfolder
    api.upload_folder(
        folder_path=str(lora_dir),
        path_in_repo="lora_adapters",
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Upload LoRA adapters for {cfg['display_name']}",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )
    print(f"  ✓ LoRA adapters uploaded")

    # Upload eval results and summary
    for local_path, repo_path in files_to_upload:
        if repo_path in ("eval_results.json", "experiment_summary.json"):
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                commit_message=f"Add {repo_path} for {cfg['display_name']}",
            )
            print(f"  ✓ {repo_path} uploaded")

    print(f"\n  ✓ Branch '{branch}' complete: https://huggingface.co/{repo_id}/tree/{branch}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Push UzABSA-LLM models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/push_to_hub.py --all                    # Push everything
  python scripts/push_to_hub.py --all --dry-run          # Preview without uploading
  python scripts/push_to_hub.py --branch main            # Update main README only
  python scripts/push_to_hub.py --branch qwen2.5-7b      # Push Qwen model only
  python scripts/push_to_hub.py --branch llama3.1-8b     # Push Llama model only
  python scripts/push_to_hub.py --branch deepseek-r1-7b  # Push DeepSeek model only
        """,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Push all branches (main + all 3 models)"
    )
    parser.add_argument(
        "--branch", type=str, choices=["main", "qwen2.5-7b", "llama3.1-8b", "deepseek-r1-7b"],
        help="Push a specific branch only"
    )
    parser.add_argument(
        "--repo-id", type=str, default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be uploaded without actually uploading"
    )

    args = parser.parse_args()

    if not args.all and not args.branch:
        parser.error("Specify --all or --branch <name>")

    # Show plan
    print(f"\n{'#' * 60}")
    print(f"  UzABSA-LLM → HuggingFace Hub Upload")
    print(f"  Repository: {args.repo_id}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if args.dry_run:
        print(f"  Mode: DRY RUN (no actual uploads)")
    print(f"{'#' * 60}")

    branches_to_push = []
    if args.all:
        branches_to_push = ["main", "qwen2.5-7b", "llama3.1-8b", "deepseek-r1-7b"]
    elif args.branch:
        branches_to_push = [args.branch]

    print(f"\n  Branches to push: {', '.join(branches_to_push)}")

    # Execute
    for branch in branches_to_push:
        if branch == "main":
            push_main_branch(args.repo_id, dry_run=args.dry_run)
        else:
            push_model_branch(branch, args.repo_id, dry_run=args.dry_run)

    print(f"\n{'#' * 60}")
    print(f"  Done! Visit: https://huggingface.co/{args.repo_id}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
