# UzABSA-LLM

<p align="center">
  <strong>Fine-tuning Large Language Models for Uzbek Aspect-Based Sentiment Analysis</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#models">Models</a>
</p>

---

## Overview

UzABSA-LLM is a project for fine-tuning open-source Large Language Models (LLMs) on Uzbek Aspect-Based Sentiment Analysis (ABSA) tasks. The goal is to train models that can extract aspect terms, categories, and sentiment polarities from Uzbek text reviews.

**Dataset**: [Sanatbek/aspect-based-sentiment-analysis-uzbek](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek) - 6,000 annotated Uzbek reviews

**Key Technologies**:
- 🚀 **Unsloth** - 2x faster training with 80% less memory
- 🔧 **QLoRA** - 4-bit quantization + Low-Rank Adaptation
- 🤗 **Hugging Face** - Transformers, Datasets, TRL, PEFT

## Features

- ✅ Support for multiple LLMs (Qwen, Llama, DeepSeek, Mistral, Gemma)
- ✅ Memory-efficient 4-bit quantization
- ✅ Parameter-efficient fine-tuning with LoRA
- ✅ Structured JSON output for aspect extraction
- ✅ Bilingual prompts (Uzbek/English)
- ✅ WandB integration for experiment tracking
- ✅ Easy model swapping and configuration

## Project Structure

```
UzABSA-LLM/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py           # Data loading and formatting
│   ├── evaluation.py          # Evaluation metrics
│   └── inference.py           # Inference utilities
├── scripts/
│   ├── train_unsloth.py       # Main training script
│   └── evaluate.py            # Evaluation script
├── configs/
│   └── training_config.yaml   # Training configurations
├── outputs/                    # Model checkpoints and logs
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/UzABSA-LLM.git
cd UzABSA-LLM
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or using conda
conda create -n uzabsa python=3.10
conda activate uzabsa
```

### Step 3: Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install Unsloth (for GPU training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

### Step 4: (Optional) Setup WandB

```bash
wandb login
```

## Quick Start

### 1. Prepare the Dataset

```bash
python -m src.data_prep --output-dir ./data/processed --val-size 0.1
```

This will:
- Download the dataset from Hugging Face
- Format it for instruction tuning
- Create train/validation splits
- Save to `./data/processed/`

### 2. Run Training

```bash
python scripts/train_unsloth.py \
    --model qwen2.5-7b \
    --dataset ./data/processed \
    --max-steps 1000 \
    --batch-size 2 \
    --learning-rate 2e-4 \
    --output-dir ./outputs
```

### 3. Run Inference

```python
from src.inference import load_model, extract_aspects

model, tokenizer = load_model("./outputs/your_run/merged_model")
text = "Bu restoranning ovqatlari juda mazali, lekin narxlari qimmat."
result = extract_aspects(model, tokenizer, text)
print(result)
# Output: {"aspects": [{"term": "ovqatlari", "category": "food", "polarity": "positive"}, ...]}
```

## Usage

### Training Options

```bash
python scripts/train_unsloth.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `qwen2.5-7b` | Model shorthand (qwen2.5-7b, llama3-8b, deepseek-7b, etc.) |
| `--model-path` | None | Custom model path (overrides --model) |
| `--dataset` | `./data/processed` | Path to processed dataset |
| `--max-seq-length` | 2048 | Maximum sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--learning-rate` | 2e-4 | Learning rate |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--max-steps` | 1000 | Max training steps (-1 for epochs) |
| `--epochs` | 3 | Number of epochs (if max-steps=-1) |
| `--output-dir` | `./outputs` | Output directory |
| `--no-wandb` | False | Disable WandB logging |

### Supported Models

| Shorthand | Full Model Name | VRAM Required |
|-----------|----------------|---------------|
| `qwen2.5-7b` | unsloth/Qwen2.5-7B-Instruct-bnb-4bit | ~6 GB |
| `qwen2.5-14b` | unsloth/Qwen2.5-14B-Instruct-bnb-4bit | ~10 GB |
| `llama3-8b` | unsloth/llama-3-8b-Instruct-bnb-4bit | ~6 GB |
| `llama3.1-8b` | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | ~6 GB |
| `deepseek-7b` | unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit | ~6 GB |
| `mistral-7b` | unsloth/mistral-7b-instruct-v0.3-bnb-4bit | ~6 GB |
| `gemma2-9b` | unsloth/gemma-2-9b-it-bnb-4bit | ~7 GB |

### Data Format

The training data is formatted as instruction-response pairs:

**Input (Prompt)**:
```
Quyidagi o'zbek tilidagi matndan aspektlarni, kategoriyalarni va hissiyot polaritesini aniqlang:

Matn: "Bu telefon juda yaxshi, lekin batareyasi tez tugaydi."
```

**Output (Response)**:
```json
{
    "aspects": [
        {
            "term": "telefon",
            "category": "general",
            "polarity": "positive"
        },
        {
            "term": "batareyasi",
            "category": "battery",
            "polarity": "negative"
        }
    ]
}
```

## Evaluation

```bash
python scripts/evaluate.py \
    --model-path ./outputs/your_run/merged_model \
    --test-data ./data/processed/test
```

Metrics:
- Aspect Term Extraction (F1)
- Aspect Category Classification (Accuracy)
- Sentiment Polarity Classification (Accuracy, F1)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{uzabsa-llm,
  title={UzABSA-LLM: Fine-tuning LLMs for Uzbek Aspect-Based Sentiment Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/UzABSA-LLM}
}
```

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - For the amazing optimization library
- [Hugging Face](https://huggingface.co/) - For the transformers ecosystem
- [Sanatbek](https://huggingface.co/Sanatbek) - For the Uzbek ABSA dataset