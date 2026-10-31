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

# UzABSA-LLM: Uzbek Aspect-Based Sentiment Analysis

Three open 7–8B LLMs fine-tuned with QLoRA for **aspect-based sentiment analysis (ABSA)** of Uzbek reviews. Given a review, a model returns JSON listing the **aspect terms** and **aspect categories** it mentions, each with its **sentiment polarity**.

## Models

Each model is stored on its own branch; load it with `revision="<branch>"`.

| Model | Branch | Base model | License |
|-------|--------|------------|---------|
| **Qwen 2.5-7B** | `qwen2.5-7b` | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache 2.0 |
| **Llama 3.1-8B** | `llama3.1-8b` | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Llama 3.1 Community License |
| **DeepSeek-R1-7B** | `deepseek-r1-7b` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | Apache 2.0 |

Each branch holds the merged model (bf16 safetensors, 15–16 GB), the LoRA adapter (`lora_adapters/`, about 165 MB), and the run's `eval_results.json` and `experiment_summary.json`. The branch cards also show how to load only the adapter, or the model in 4-bit with Unsloth.

## Results

Scores on the 609 held-out UzABSA reviews (737 gold aspect terms). Bold marks the best single fine-tuned system in each column.

| System | ATE P | ATE R | ATE exact F1 | ATE partial F1 | Pair F1 | Sentiment acc. | Sentiment macro-F1 | JSON parse |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5-7B**, fine-tuned (`qwen2.5-7b`) | 0.7141 | 0.7015 | **0.7077** | **0.8036** | 0.6448 | 0.9110 | 0.8622 | **100.0%** |
| **Llama 3.1-8B**, fine-tuned (`llama3.1-8b`) | 0.7185 | 0.6893 | 0.7036 | 0.8019 | **0.6510** | **0.9252** | **0.8867** | 97.4% |
| **DeepSeek-R1-7B**, fine-tuned (`deepseek-r1-7b`) | 0.7143 | 0.6445 | 0.6776 | 0.7892 | 0.6034 | 0.8905 | 0.8314 | 96.9% |
| BERTbek, fine-tuned encoder | 0.6802 | **0.7042** | 0.6920 | 0.7995 | 0.6080 | 0.8786 | 0.8126 | – |
| TahrirchiBERT, fine-tuned encoder | **0.7287** | 0.6744 | 0.7005 | 0.7865 | 0.6225 | 0.8893 | 0.8140 | – |
| Majority vote of the five fine-tuned systems | 0.8003 | 0.7069 | 0.7507 | 0.8213 | 0.6902 | 0.9194 | 0.8699 | – |
| Qwen 2.5-7B, zero-shot | 0.2504 | 0.4518 | 0.3222 | 0.5571 | 0.2100 | 0.6517 | 0.4719 | 99.8% |
| Llama 3.1-8B, zero-shot | 0.2271 | 0.3799 | 0.2843 | 0.4718 | 0.1999 | 0.7036 | 0.6280 | 95.4% |
| DeepSeek-R1-7B, zero-shot | 0.2080 | 0.2266 | 0.2169 | 0.2790 | 0.1540 | 0.7126 | 0.5683 | 46.3% |

ATE is aspect term extraction, scored on exact matches (P: precision, R: recall) unless marked partial; partial F1 also credits a predicted term that contains the gold term or lies inside it, with one-to-one matching. Pair F1 counts a term as correct only with the right polarity. Sentiment accuracy and macro-F1 are computed on the exactly matched terms.

- **Fine-tuning is essential:** compared with zero-shot prompting of the same base models, it multiplies exact ATE F1 by 2.2–3.1 and pair F1 by 3.1–3.9.
- **Qwen and Llama form the top tier and are statistically tied** (paired bootstrap), so either is a sound choice. Qwen is the only LLM that returns valid JSON for every review.
- **DeepSeek trails both.** Its pair-F1 gap to Llama is the only difference between fine-tuned systems that stays significant after Holm correction; it misses more aspects (lower recall) rather than producing broken output.
- **Encoders versus LLMs:** on extraction alone, the 110M-parameter Uzbek BERT encoders are competitive (TahrirchiBERT has the highest precision and BERTbek the highest recall of the single systems). On joint prediction and sentiment, Qwen and Llama score higher than both encoders, by up to 4.3 pair-F1 and 7.4 macro-F1 points.
- **A majority vote** that keeps a term when at least three of the five fine-tuned systems predict it reaches 0.751 exact ATE F1 and 0.690 pair F1, significantly above every single system, mainly through higher precision (0.800).

## Quick start

```bash
pip install transformers accelerate
```

```python
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

repo, branch = "Sanatbek/UzABSA-LLM", "qwen2.5-7b"  # or "llama3.1-8b", "deepseek-r1-7b"
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

## Output format

The answer holds two kinds of records, as UzABSA does:

- **term records**, `{"term": ..., "polarity": ...}`: a phrase of the review and its polarity;
- **category records**, `{"category": ..., "polarity": ...}`: one of five categories, `ovqat` (food), `muhit` (ambience), `xizmat` (service), `narx` (price) or `boshqalar` (other), and its polarity.

Polarity is `positive`, `negative` or `neutral`. UzABSA annotates terms and categories as independent layers, so a category record is not linked to a term. The scores count the term records.

## Training

The three models share one configuration, fixed in advance:

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

**Hardware.** One NVIDIA RTX A6000 (48 GB), CUDA 12.4, PyTorch 2.6.0.

**Corrected release (2026-08-29).** All three models were retrained after two data-pipeline fixes: the training examples had carried the literal string `"None"` as their system prompt, and 61 of the 609 held-out reviews had an exact-text twin in training. Scores of weights uploaded before that date are not comparable with the ones above; each branch card has the details.

## Reproduction and seed variance

The scores above come from one run per system, and those runs' checkpoints were chosen on the same 609 reviews they are scored on. A stricter rerun is under way in [`experiments/v3/`](https://github.com/SanatbekMatlatipov/UzABSA-LLM/tree/main/experiments/v3): it carves a separate dev split out of the training data, chooses every checkpoint and the corpus annotator on dev alone, and repeats each system over three seeds so results can be reported as mean ± sd rather than a single number. The test split is these same 609 reviews, byte for byte, so the two sets of numbers stay comparable.

Expect small downward corrections wherever a published score benefited from choosing a checkpoint on the test set. `experiments/v3/SUMMARY.md` and `experiments/v3/registry.json` hold the current state, both rebuilt from the prediction files rather than written by hand.

## Code and data

- **Code** (training, evaluation, significance tests and the annotation pipeline): [github.com/SanatbekMatlatipov/UzABSA-LLM](https://github.com/SanatbekMatlatipov/UzABSA-LLM)
- **Training data:** [Sanatbek/aspect-based-sentiment-analysis-uzbek](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek)
- **Multi-domain silver corpus** (5,038 reviews from 23 business domains, annotated by an earlier Qwen checkpoint and scored by a panel of LLM judges): [doi:10.5281/zenodo.22146677](https://doi.org/10.5281/zenodo.22146677)

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

The `qwen2.5-7b` and `deepseek-r1-7b` branches are released under Apache 2.0. **Built with Llama.** The `llama3.1-8b` branch is governed by the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) and the [Llama 3.1 Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/).
