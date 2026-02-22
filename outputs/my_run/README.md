# Training Outputs

This directory contains fine-tuned model checkpoints and training artifacts.

## Large Files (Not Committed)

The following large files are excluded from Git due to size:
- `*/merged_model/` — Full merged 16-bit LLM (4-8 GB per model)
- `*/lora_adapters/` — LoRA adapter weights (~100 MB)
- `*/checkpoint-*/` — Training checkpoints with optimizer state
- `*/*.png` — Generated plots
- `*/*.csv`/`*.json` — Training history

These files are generated during local training runs. To access the best trained models, see below.

## Best Models on HuggingFace

All best-performing fine-tuned models are hosted on HuggingFace Hub:

🤗 **[Sanatbek/UzABSA-LLM](https://huggingface.co/Sanatbek/UzABSA-LLM/tree/main)**

This hub contains:
- **Best performing models** across different base architectures
- **LoRA adapters** for each model
- **Model cards** with training configuration and performance metrics
- **Training curves** and experimental results

### Available Models
- `Qwen2.5-7B-UzABSA` — Default recommendation (fast, accurate)
- `Llama-3.1-8B-UzABSA` — Largest, best for complex tasks
- `DeepSeek-7B-UzABSA` — Reasoning-focused variant
- `Mistral-7B-UzABSA` — Light-weight alternative

### Load Models
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Using merged model
model = AutoModelForCausalLM.from_pretrained(
    "Sanatbek/UzABSA-LLM/Qwen2.5-7B-UzABSA-merged"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Sanatbek/UzABSA-LLM/Qwen2.5-7B-UzABSA-merged"
)

# Using LoRA adapter (more efficient)
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
model = PeftModel.from_pretrained(base_model, "Sanatbek/UzABSA-LLM/Qwen2.5-7B-UzABSA-lora")
```

### Run Inference
```bash
python scripts/inference.py \
    --model Sanatbek/UzABSA-LLM/Qwen2.5-7B-UzABSA-merged \
    --text "Bu telefon juda barakali lekin qimmat"
```

---

## Local Training Logs

For **local training artifacts** (if running locally):
- Check timestamps in folder names (e.g., `uzabsa_qwen2.5-7b_20260222_001629`)
- Open `experiment_summary.json` for full training config + results
- View `training_curves.png` for loss curves
- Check W&B logs: [wandb.ai/Sanatbek/uzabsa-llm](https://wandb.ai/Sanatbek/uzabsa-llm)

---

See [RESEARCH_LOG.md](../../RESEARCH_LOG.md) for the experiment roadmap and methodology.
