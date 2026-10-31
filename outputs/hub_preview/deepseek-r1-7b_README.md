---
language:
  - uz
license: apache-2.0
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
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
  - name: UzABSA DeepSeek-R1-Distill-Qwen-7B
    results:
      - task:
          type: token-classification
          name: Aspect Term Extraction
        dataset:
          name: UzABSA (SemEval-2014 format)
          type: Sanatbek/aspect-based-sentiment-analysis-uzbek
          split: validation
        metrics:
          - type: f1
            value: 0.6776
            name: ATE Exact F1
          - type: f1
            value: 0.7892
            name: ATE Partial F1 (one-to-one)
          - type: f1
            value: 0.6034
            name: Pair F1
      - task:
          type: text-classification
          name: Aspect Sentiment Classification
        dataset:
          name: UzABSA (SemEval-2014 format)
          type: Sanatbek/aspect-based-sentiment-analysis-uzbek
          split: validation
        metrics:
          - type: accuracy
            value: 0.8905
            name: Sentiment Accuracy
          - type: f1
            value: 0.8314
            name: Sentiment Macro-F1
---

# UzABSA DeepSeek-R1-Distill-Qwen-7B

A **7.6B** parameter LLM fine-tuned with QLoRA for **aspect-based sentiment analysis (ABSA)** of Uzbek reviews. Given a review, it returns JSON listing the aspect terms and aspect categories it mentions, each with its sentiment polarity.

> This is the **`deepseek-r1-7b`** branch of [Sanatbek/UzABSA-LLM](https://huggingface.co/Sanatbek/UzABSA-LLM). The [main page](https://huggingface.co/Sanatbek/UzABSA-LLM) compares the three models.

> [!IMPORTANT]
> **Corrected release (2026-08-29).** These weights supersede an earlier upload. Two defects in the data
> pipeline were found and fixed, and every model was retrained from scratch:
>
> 1. **Malformed system prompt.** The preprocessing script passed `system_prompt=None`, which overrode the
>    default instead of selecting it, so all 5,480 training examples carried the literal string `"None"` as
>    their system prompt while inference served the real Uzbek instruction.
> 2. **Train/evaluation overlap.** The split was drawn over annotation records without deduplication, so 61
>    of the 609 held-out items (10.0%) had an exact-text twin in training. The split is now grouped by
>    normalised review text; sizes are unchanged (5,480/609).
>
> Scores from the earlier weights are not comparable with the ones below.

## Model Details

| Property | Value |
|----------|-------|
| Base model | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| Architecture | Qwen2ForCausalLM |
| Parameters | 7.6B |
| Fine-tuning method | QLoRA (4-bit NF4), Unsloth + TRL |
| Task | Aspect-based sentiment analysis |
| Language | Uzbek (uz) |
| Merged model size | 15.2 GB (bf16 safetensors) |
| LoRA adapter size | 162 MB (`lora_adapters/`) |

## Evaluation Results

Scores on the 609 held-out UzABSA reviews (737 gold aspect terms), next to the base model prompted zero-shot with the same prompt:

| Metric | This model (fine-tuned) | Base model (zero-shot) |
|--------|:---:|:---:|
| ATE P | **0.7143** | 0.2080 |
| ATE R | **0.6445** | 0.2266 |
| ATE exact F1 | **0.6776** | 0.2169 |
| ATE partial F1 | **0.7892** | 0.2790 |
| Pair F1 | **0.6034** | 0.1540 |
| Sentiment acc. | **0.8905** | 0.7126 |
| Sentiment macro-F1 | **0.8314** | 0.5683 |
| JSON parse | **96.9%** | 46.3% |

ATE is aspect term extraction, scored on exact matches (P: precision, R: recall) unless marked partial; partial F1 also credits a predicted term that contains the gold term or lies inside it, with one-to-one matching. Pair F1 counts a term as correct only with the right polarity. Sentiment accuracy and macro-F1 are computed on the exactly matched terms. `eval_results.json` holds the scores written at training time; its partial F1 uses greedy instead of one-to-one matching.

## Usage

### Merged model

```bash
pip install transformers accelerate
```

```python
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

repo, branch = "Sanatbek/UzABSA-LLM", "deepseek-r1-7b"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=branch)
model = AutoModelForCausalLM.from_pretrained(repo, revision=branch, dtype="auto", device_map="auto")

SYSTEM_PROMPT = """Siz o'zbek tilida matnlardan aspektlarni va ularning hissiyotlarini aniqlash bo'yicha mutaxassissiz.

Berilgan matndan barcha aspekt terminlarini, ularning kategoriyalarini va hissiyot polaritesini (positive, negative, neutral) ajratib oling.

Javobni quyidagi Python dictionary formatida qaytaring:
{
    "aspects": [
        {
            "term": "aspekt termin",
            "category": "kategoriya",
            "polarity": "positive/negative/neutral"
        }
    ]
}"""


def build_prompt(review):
    # The ChatML text all three models were trained on. Do not use
    # tokenizer.apply_chat_template: Llama and DeepSeek have other templates.
    user = ("Quyidagi o'zbek tilidagi matndan aspektlarni, kategoriyalarni va hissiyot polaritesini aniqlang:"
            f'\n\nMatn: "{review}"')
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")


inputs = tokenizer(build_prompt("Ovqat mazali, lekin xizmat juda sekin edi."), return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,  # greedy decoding, as in the evaluation
    eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False),
    pad_token_id=tokenizer.eos_token_id,
)
answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer)
aspects = json.loads(answer[: answer.rfind("}") + 1])["aspects"]
```

The scores above come from `scripts/evaluate.py` in the [GitHub repository](https://github.com/SanatbekMatlatipov/UzABSA-LLM), which loads the merged model in 4-bit with Unsloth and decodes greedily; bf16 outputs can differ slightly.

### 4-bit with Unsloth, as in the evaluation

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Sanatbek/UzABSA-LLM",
    revision="deepseek-r1-7b",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
# then build the prompt and generate as above
```

### LoRA adapter only (162 MB)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
model = PeftModel.from_pretrained(base, "Sanatbek/UzABSA-LLM", revision="deepseek-r1-7b", subfolder="lora_adapters")
# then build the prompt and generate as above
```

## Output Format

The answer holds two kinds of records, as UzABSA does:

- **term records**, `{"term": ..., "polarity": ...}`: a phrase of the review and its polarity;
- **category records**, `{"category": ..., "polarity": ...}`: one of five categories, `ovqat` (food), `muhit` (ambience), `xizmat` (service), `narx` (price) or `boshqalar` (other), and its polarity.

Polarity is `positive`, `negative` or `neutral`. UzABSA annotates terms and categories as independent layers, so a category record is not linked to a term. The scores count the term records.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | Unsloth + TRL (SFTTrainer) + PEFT |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q, k, v, o, gate, up and down projections |
| Learning rate | 2e-4, cosine schedule, 10% warm-up |
| Effective batch size | 16 (4 × 4 gradient accumulation) |
| Steps | 1,000 |
| Optimizer | `adamw_8bit`, weight decay 0.01 |
| Precision | bf16 |
| Max sequence length | 2,048 tokens |
| Seed | 42 |
| Checkpoint kept | lowest loss on the 609 held-out reviews, evaluated every 100 steps |

**Data.** [UzABSA](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek) holds 6,175 Uzbek restaurant-review sentences in the SemEval-2014 format. Conversion to instruction examples, with the *conflict* polarity removed, yields 6,089 examples, split 90/10 (seed 42) into 5,480 training and 609 held-out reviews. Identical normalised texts are kept on the same side of the split.

**Hardware.** One NVIDIA RTX A6000 (48 GB), CUDA 12.4, PyTorch 2.6.0. The run's log is in `experiment_summary.json`.

## Repository Contents

```
├── config.json                           # model config
├── model-0000X-of-00004.safetensors      # merged weights (4 shards)
├── model.safetensors.index.json          # weight index
├── tokenizer files
├── lora_adapters/                        # LoRA adapter (lightweight alternative)
├── eval_results.json                     # scores written at training time
└── experiment_summary.json               # training configuration and log
```

## Code

Training, evaluation and significance tests: [github.com/SanatbekMatlatipov/UzABSA-LLM](https://github.com/SanatbekMatlatipov/UzABSA-LLM)

## Citation

```bibtex
@misc{matlatipov2026uzabsallm,
  title        = {UzABSA-LLM: Fine-tuned Large Language Models for Uzbek Aspect-Based Sentiment Analysis},
  author       = {Matlatipov, Sanatbek},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Sanatbek/UzABSA-LLM}}
}
```

The training data:

```bibtex
@inproceedings{matlatipov-etal-2024-uzabsa,
  title     = {{UzABSA}: Aspect-Based Sentiment Analysis for the {Uzbek} Language},
  author    = {Matlatipov, Sanatbek Gayratovich and Rajabov, Jaloliddin and Kuriyozov, Elmurod and Aripov, Mersaid},
  booktitle = {Proceedings of the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages @ LREC-COLING 2024},
  year      = {2024},
  pages     = {394--403},
  url       = {https://aclanthology.org/2024.sigul-1.47/}
}
```

## License

Apache 2.0.
