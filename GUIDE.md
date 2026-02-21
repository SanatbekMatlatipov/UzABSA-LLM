# UzABSA-LLM Quick Reference Guide

Complete guide for working with raw reviews, annotated datasets, and training models.

## Quick Setup

```bash
# Activate virtual environment
.venv\Scripts\activate     # Windows
source .venv/bin/activate # Linux/Mac

# Install dependencies (if not already done)
pip install -r requirements.txt
```

## Dataset Overview

### Annotations Structure (SemEVAL 2014)

The Hugging Face dataset follows SemEVAL 2014 format:

```python
{
    "sentence_id": "2771#1",
    "text": "Juda yaxshi ovqat va kayfiyat",
    "aspect_terms": [
        {
            "term": "ovqat",           # Aspect term
            "polarity": "positive",    # positive/negative/neutral
            "from": 12,                # Character position start
            "to": 17                   # Character position end
        }
    ],
    "aspect_categories": [
        {
            "category": "food",        # Category name
            "polarity": "positive"     # Category-level polarity
        }
    ]
}
```

### Data Statistics

**Raw Reviews** (sharh.commeta.uz):
- 5,058 reviews
- Average: 13.55 words, 96.26 characters per review
- Multiple organizations and domains

**Annotated ABSA Dataset**:
- 6,175 examples (train split)
- 2.53 aspects per example on average
- 6.42 words per text
- Polarity distribution: ~64% positive, 16% negative, 20% neutral

## Usage Examples

### 1. Explore Datasets

```bash
# View both raw and annotated datasets
python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv --analyze --merge
```

**Output**: Statistics for raw reviews, annotated examples, and matching results.

### 2. Python API for Dataset Loading

```python
from src.dataset_utils import (
    load_raw_reviews_csv,
    load_annotated_absa_dataset,
    clean_raw_reviews,
    analyze_dataset_stats,
)

# Load raw reviews
raw_df = load_raw_reviews_csv("./data/raw/reviews.csv")
print(f"Loaded {len(raw_df)} raw reviews")

# Clean reviews (remove nulls, short texts, duplicates)
raw_df = clean_raw_reviews(raw_df, min_length=10)

# Analyze
stats = analyze_dataset_stats(raw_df, text_field="review_text")
print(f"Avg words: {stats['avg_words']}")

# Load annotated dataset
dataset = load_annotated_absa_dataset(split="train")
print(f"Loaded {len(dataset)} annotated examples")
```

### 3. Format Conversion

Convert from SemEVAL 2014 format to instruction-tuning format:

```python
from src.format_converter import (
    convert_semeval_to_instruction_format,
    convert_dataset,
    analyze_converted_dataset,
)

# Load and convert
dataset = load_annotated_absa_dataset(split="train")
converted = convert_dataset(dataset, format_type="semeval")

# Check structure
print(converted[0])
# Output:
# {
#     "aspects": [
#         {"term": "ovqat", "polarity": "positive"},
#         {"category": "food", "polarity": "positive"}
#     ],
#     "text": "Juda yaxshi ovqat va kayfiyat"
# }

# Analyze
stats = analyze_converted_dataset(converted)
print(f"Total aspects: {stats['total_aspects']}")
print(f"Polarity dist: {stats['polarity_distribution']}")
```

### 4. Complete Data Preparation

One-command pipeline to prepare data for training:

```bash
# Prepare with full dataset
python scripts/prepare_complete_dataset.py --output-dir ./data/processed

# Or with a subset for testing
python scripts/prepare_complete_dataset.py --max-examples 500 --output-dir ./data/test
```

**Output**: Train/validation splits with ChatML formatting, ready for training.

### 5. GPU Configuration (Recommended for 4x RTX A6000)

Check available GPUs and get training recommendations:

```bash
# Display GPU status
python -m src.gpu_config --check

# Output:
# GPU 0: NVIDIA RTX A6000 (46.0 GB)
# GPU 1: NVIDIA RTX A6000 (46.0 GB)
# ...

# Get recommended training settings
python -m src.gpu_config --recommend

# Output:
# Recommended Training Configuration
# ==================================
# num_gpus        : 2
# total_memory_gb : 92.0
# batch_size      : 8
# grad_accum      : 2
# device_map      : auto
# note            : Multi-GPU training (2 GPUs)
```

Single GPU training (RTX A6000):

```bash
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --gpu-id 0 \
    --batch-size 8 \
    --grad-accum 2 \
    --dataset ./data/processed
```

Multi-GPU training (recommended for faster convergence):

```bash
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --multi-gpu \
    --batch-size 8 \
    --grad-accum 2 \
    --dataset ./data/processed
```

**GPU Arguments**:
- `--gpu-id <int>`: Use specific GPU (0, 1, 2, 3, etc.)
- `--device-map <str>`: Device strategy (auto, cuda:0, cuda:1, cpu)
- `--multi-gpu`: Enable multi-GPU DistributedDataParallel training

### 6. Model Training

```bash
# Quick start training
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --dataset ./data/processed \
    --max-steps 500

# Full custom training
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --dataset ./data/processed \
    --lora-r 16 \
    --lora-alpha 32 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --grad-accum 4 \
    --max-steps 1000 \
    --output-dir ./outputs/my_experiment \
    --run-name "qwen_absa_v1"
```

### 6. Model Training

```bash
# Quick start training
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --dataset ./data/processed \
    --max-steps 500

# Full custom training
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --dataset ./data/processed \
    --lora-r 16 \
    --lora-alpha 32 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --grad-accum 4 \
    --max-steps 1000 \
    --output-dir ./outputs/my_experiment \
    --run-name "qwen_absa_v1"
```

### 7. Inference

```python
from src.inference import load_model, extract_aspects
import json

# Load model
model, tokenizer = load_model("./outputs/my_experiment/merged_model")

# Extract aspects from text
text = "Bu restoranning ovqatlari juda mazali, lekin narxlari qimmat."
result = extract_aspects(model, tokenizer, text)

print(json.dumps(result, indent=2, ensure_ascii=False))
# Output:
# {
#     "aspects": [
#         {
#             "term": "ovqatlari",
#             "polarity": "positive"
#         },
#         {
#             "term": "narxlari",
#             "polarity": "negative"
#         }
#     ]
# }
```

### 8. Batch Inference

```python
from src.inference import extract_aspects_batch

texts = [
    "Bu telefon juda yaxshi!",
    "Juda qimmat va xato.",
    "O'rtacha mahsulot.",
]

results = extract_aspects_batch(model, tokenizer, texts)
for i, result in enumerate(results):
    print(f"Text {i}: {result}")
```

### 9. Model Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --model-path ./outputs/my_experiment/merged_model \
    --test-data ./data/processed

# With max samples for quick evaluation
python scripts/evaluate.py \
    --model-path ./outputs/my_experiment/merged_model \
    --test-data ./data/processed \
    --max-samples 100
```

## Project Structure

```
src/
├── __init__.py              # Package exports
├── data_prep.py             # Data loading from HuggingFace
├── dataset_utils.py         # Raw dataset utilities
├── format_converter.py       # SemEVAL to instruction format
├── inference.py             # Model loading and inference
└── evaluation.py            # Evaluation metrics

scripts/
├── explore_datasets.py       # Dataset exploration
├── prepare_complete_dataset.py  # End-to-end pipeline
├── train_unsloth.py         # Training script
└── evaluate.py              # Evaluation script

configs/
└── training_config.yaml     # YAML config template

data/
├── raw/
│   └── reviews.csv         # Raw reviews (5,000+)
└── processed/              # Prepared datasets

outputs/
└── {run_name}/             # Trained models & checkpoints
```

## Tips and Best Practices

### Data Preparation

1. **Always clean raw data first**:
   ```python
   raw_df = clean_raw_reviews(raw_df, min_length=15)
   ```

2. **Check data statistics before training**:
   ```python
   stats = analyze_dataset_stats(dataset)
   ```

3. **Verify format conversion**:
   ```python
   converted = convert_dataset(dataset[:10])  # Test on small subset
   ```

### Training

1. **Start with small dataset for testing**:
   ```bash
   python scripts/prepare_complete_dataset.py --max-examples 100
   python scripts/train_unsloth.py --max-steps 100 --batch-size 1
   ```

2. **Use validation split for early stopping**:
   - Default val_size=0.1 (10%)
   - Set lower val_size for larger datasets

3. **Monitor training**:
   - Check WandB dashboard for loss curves
   - Watch for overfitting with eval metrics

4. **Save final model**:
   - LoRA adapters: `outputs/merged_model/lora_adapters/`
   - Merged model: `outputs/merged_model/`
   - GGUF for inference: `outputs/merged_model/model_gguf/`

### Inference

1. **Batch inference for efficiency**:
   ```python
   results = extract_aspects_batch(model, tokenizer, texts, batch_size=8)
   ```

2. **Use appropriate system prompts**:
   - Uzbek: `use_uzbek=True` (default)
   - English: `use_uzbek=False`

3. **Handle edge cases**:
   - Very short texts (< 5 words)
   - Non-Uzbek text
   - Special characters

## Troubleshooting

### Dataset Issues

**Problem**: Raw CSV not loading
```
Solution: Check encoding, use --raw-file with correct path
```

**Problem**: HuggingFace download fails
```
Solution: Set HF_HOME, check internet, or download manually
```

### Training Issues

**Problem**: CUDA out of memory
```
Solution: Reduce batch_size, increase gradient_accumulation_steps
```

**Problem**: Slow training
```
Solution: Use Unsloth for 2x speedup, enable packing for short sequences
```

### Inference Issues

**Problem**: Model outputs invalid JSON
```
Solution: Check parse_model_output, use extract_aspects_from_text fallback
```

## Configuration Files

**training_config.yaml** - Predefined configurations for different models:

```yaml
# Qwen 2.5 7B (fast & good)
model:
  name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  max_seq_length: 2048
lora:
  r: 16
  alpha: 32
training:
  learning_rate: 2e-4
  max_steps: 1000
```

## Further Reading

- [SemEVAL 2014 Task 4: Aspect-Based Sentiment Analysis](https://www.semantic-proceedings.org/semeval2014/task4/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## Support

For issues or questions:
1. Check the logs: `training_*.log` or `data_prep_*.log`
2. Review dataset statistics: `conversion_stats.json`, `metadata.json`
3. Inspect failed examples and edge cases
