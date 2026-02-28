---
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
| **Qwen 2.5-7B** | `qwen2.5-7b` | `AutoModelForCausalLM.from_pretrained("Sanatbek/UzABSA-LLM", revision="qwen2.5-7b")` |
| **Llama 3.1-8B** | `llama3.1-8b` | `AutoModelForCausalLM.from_pretrained("Sanatbek/UzABSA-LLM", revision="llama3.1-8b")` |
| **DeepSeek-R1-Distill-Qwen-7B** | `deepseek-r1-7b` | `AutoModelForCausalLM.from_pretrained("Sanatbek/UzABSA-LLM", revision="deepseek-r1-7b")` |

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
model_name = "Sanatbek/UzABSA-LLM"
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
    {"role": "system", "content": "Siz aspect-based sentiment analysis mutaxassisisiz. Berilgan sharh matnidan aspect termlarni, ularning kategoriyalarini va sentiment polarligini JSON formatida chiqaring."},
    {"role": "user", "content": "Sharh: Ovqat mazali, lekin xizmat juda sekin edi."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

**Expected output:**
```json
{
  "aspects": [
    {"aspect": "ovqat", "category": "Taom sifati", "sentiment": "ijobiy"},
    {"aspect": "xizmat", "category": "Xizmat ko'rsatish", "sentiment": "salbiy"}
  ]
}
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
    "Sanatbek/UzABSA-LLM",
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
{
  "aspects": [
    {
      "aspect": "<aspect term in original language>",
      "category": "<aspect category>",
      "sentiment": "ijobiy | salbiy | neytral"
    }
  ]
}
```

## Citation

If you use these models in your research, please cite:

```bibtex
@misc{uzabsa-llm-2026,
  title={UzABSA-LLM: Fine-tuned Large Language Models for Uzbek Aspect-Based Sentiment Analysis},
  author={Sanatbek},
  year={2026},
  url={https://huggingface.co/Sanatbek/UzABSA-LLM},
  note={QLoRA fine-tuned Qwen 2.5-7B, Llama 3.1-8B, and DeepSeek-R1-Distill-Qwen-7B}
}
```

## Project Repository

Full source code, training scripts, evaluation pipeline, and research log:
**[GitHub: UzABSA-LLM](https://github.com/Sanatbek/UzABSA-LLM)**

## License

Apache 2.0
