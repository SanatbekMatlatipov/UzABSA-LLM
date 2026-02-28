---
language:
  - uz
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
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
  - name: UzABSA Qwen 2.5-7B
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
            value: 0.6603
            name: ATE Exact F1
          - type: f1
            value: 0.7705
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
            value: 0.8777
            name: Sentiment Accuracy
          - type: f1
            value: 0.8113
            name: Sentiment Macro-F1
---

# UzABSA Qwen 2.5-7B

A **7.6B** parameter LLM fine-tuned for **Aspect-Based Sentiment Analysis (ABSA)** in Uzbek using QLoRA. Extracts aspect terms, categories, and sentiment polarities from Uzbek text reviews as structured JSON.

> This is the **`qwen2.5-7b`** branch of [Sanatbek/UzABSA-LLM](https://huggingface.co/Sanatbek/UzABSA-LLM). See the [main branch](https://huggingface.co/Sanatbek/UzABSA-LLM) for a comparison of all models.

## Model Details

| Property | Value |
|----------|-------|
| Base model | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Architecture | Qwen2ForCausalLM |
| Parameters | 7.6B |
| Fine-tuning method | QLoRA (4-bit NF4) |
| Framework | Unsloth + HuggingFace TRL |
| Task | Aspect-Based Sentiment Analysis |
| Language | Uzbek (uz) |
| Merged model size | 14.2 GB |
| LoRA adapter size | 154 MB |

## Evaluation Results

Evaluated on 609 held-out validation examples:

| Metric | Score |
|--------|:-----:|
| **ATE Exact F1** | 0.6603 |
| **ATE Partial F1** | 0.7705 |
| **Pair F1** (aspect + sentiment) | 0.5795 |
| **Sentiment Accuracy** | 0.8777 |
| **Sentiment Macro-F1** | 0.8113 |
| **JSON Parse Rate** | 100.0% |

## Usage

### Load Merged Model (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Sanatbek/UzABSA-LLM",
    revision="qwen2.5-7b",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Sanatbek/UzABSA-LLM", revision="qwen2.5-7b")

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

### Load LoRA Adapter Only (~154 MB)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model = PeftModel.from_pretrained(
    base,
    "Sanatbek/UzABSA-LLM",
    revision="qwen2.5-7b",
    subfolder="lora_adapters",
)
```

### With Unsloth (2x Faster Inference)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Sanatbek/UzABSA-LLM",
    revision="qwen2.5-7b",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Learning rate | 2e-4 |
| Effective batch size | 16 (4 × 4) |
| Max steps | 1000 |
| Warmup | 0.1 |
| Scheduler | cosine |
| Optimizer | adamw_8bit |
| Precision | bf16 |
| Max seq length | 2048 |
| Seed | 42 |
| Final train loss | 0.4010 |
| Best eval loss | 0.3656 |
| Training time | 282.6 min |

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
@misc{uzabsa-llm-2026,
  title={UzABSA-LLM: Fine-tuned Large Language Models for Uzbek Aspect-Based Sentiment Analysis},
  author={Sanatbek},
  year={2026},
  url={https://huggingface.co/Sanatbek/UzABSA-LLM},
  note={UzABSA Qwen 2.5-7B — QLoRA fine-tuned on Uzbek ABSA}
}
```

## License

Apache 2.0
