#!/usr/bin/env python3
"""
Push UzABSA-LLM fine-tuned models to HuggingFace Hub.

Repository structure (single repo, multiple branches):
  - main branch:        Project README + evaluation comparison (no weights)
  - qwen2.5-7b branch:  Merged model + LoRA adapters + model card
  - llama3.1-8b branch: Merged model + LoRA adapters + model card
  - deepseek-r1-7b branch: Merged model + LoRA adapters + model card

The cards read their scores from paper_materials/revision_v2/robustness/robustness.json
and the majority-vote metrics in experiments/v3/results/ensembles/.

Usage:
  # Push everything (all branches)
  python scripts/push_to_hub.py --all

  # Push only the main branch (README + comparison)
  python scripts/push_to_hub.py --branch main

  # Push a specific model branch
  python scripts/push_to_hub.py --branch qwen2.5-7b

  # Update only the cards, not the weights (works without the merged models)
  python scripts/push_to_hub.py --all --card-only

  # Dry-run (shows what would be uploaded without actually uploading)
  python scripts/push_to_hub.py --all --dry-run

Prerequisites:
  pip install huggingface_hub
  hf auth login   # or huggingface-cli login, or set HF_TOKEN env variable
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Windows terminals default to cp1252, which cannot encode the arrows and box
# glyphs this script prints. Without this the script dies on its first banner
# line — before uploading anything — with a UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID = "Sanatbek/UzABSA-LLM"
GITHUB_URL = "https://github.com/SanatbekMatlatipov/UzABSA-LLM"
DATASET_ID = "Sanatbek/aspect-based-sentiment-analysis-uzbek"
CORPUS_DOI = "10.5281/zenodo.22146677"

ROBUSTNESS_JSON = "paper_materials/revision_v2/robustness/robustness.json"
VOTE_JSON = "experiments/v3/results/ensembles/majority_of_5_v2_systems_metrics.json"
VOTE_NAME = "Majority of 5 (v2 systems)"

# Sizes follow from the parameter counts: bf16 merged weights, fp32 adapters
# (40.4M trainable parameters for the Qwen architecture, 41.9M for Llama).
MODELS = {
    "qwen2.5-7b": {
        "branch": "qwen2.5-7b",
        "local_dir": "outputs/rerun_v2/uzabsa_qwen2.5-7b_v2",
        "paper_name": "Qwen 2.5-7B",  # system name in robustness.json
        "display_name": "UzABSA Qwen 2.5-7B",
        "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "base_model_full": "Qwen/Qwen2.5-7B-Instruct",
        "architecture": "Qwen2ForCausalLM",
        "params": "7.6B",
        "merged_size": "15.2 GB",
        "lora_size": "162 MB",
        "license": "apache-2.0",
    },
    "llama3.1-8b": {
        "branch": "llama3.1-8b",
        "local_dir": "outputs/rerun_v2/uzabsa_llama3.1-8b_v2",
        "paper_name": "Llama 3.1-8B",
        # the Llama 3.1 license asks for "Llama" at the start of a derived model's name
        "display_name": "Llama 3.1-8B UzABSA",
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "base_model_full": "meta-llama/Llama-3.1-8B-Instruct",
        "architecture": "LlamaForCausalLM",
        "params": "8.0B",
        "merged_size": "16.1 GB",
        "lora_size": "168 MB",
        "license": "llama3.1",
    },
    "deepseek-r1-7b": {
        "branch": "deepseek-r1-7b",
        "local_dir": "outputs/rerun_v2/uzabsa_deepseek-7b_v2",
        "paper_name": "DeepSeek-R1-7B",
        "display_name": "UzABSA DeepSeek-R1-Distill-Qwen-7B",
        "base_model": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
        "base_model_full": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "architecture": "Qwen2ForCausalLM",
        "params": "7.6B",
        "merged_size": "15.2 GB",
        "lora_size": "162 MB",
        "license": "apache-2.0",
    },
}

LICENSE_NAMES = {
    "apache-2.0": "Apache 2.0",
    "llama3.1": "Llama 3.1 Community License",
}

LLAMA_NOTICE = (
    "**Built with Llama.** This model is a fine-tune of Llama 3.1 8B Instruct. Its use is governed by the "
    "[Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) and the "
    "[Llama 3.1 Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/)."
)

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


def load_results() -> dict:
    """Held-out scores of every system in the main results table, by system name.

    scripts/robustness_analysis.py and scripts/ensemble_preds.py recompute them
    from the per-example predictions with the evaluation's metric code. The
    eval_results files written at training time use an older partial-match score.
    """
    root = get_project_root()
    with open(root / ROBUSTNESS_JSON, encoding="utf-8") as f:
        robustness = json.load(f)
    with open(root / VOTE_JSON, encoding="utf-8") as f:
        vote = json.load(f)
    n_reviews = robustness["gold"]["reviews"]
    results = {}
    for name, s in dict(robustness["systems"], **{VOTE_NAME: vote["scores"][VOTE_NAME]}).items():
        c = s["canonical"]
        failures = robustness["format_vs_extraction"].get(name, {}).get("parse_failures")
        results[name] = {"p": c["ate_exact_p"], "r": c["ate_exact_r"], "f1": c["ate_exact_f1"],
                         "partial": s["partial_1to1"]["f1"], "pair": c["pair_f1"],
                         "acc": c["sentiment_accuracy"], "macro": c["sentiment_macro_f1"],
                         "parse": None if failures is None else 1 - failures / n_reviews}
    return results


# ---------------------------------------------------------------------------
# Card sections shared by all cards
# ---------------------------------------------------------------------------

EXAMPLE_REVIEW = "Ovqat mazali, lekin xizmat juda sekin edi."

# The qwen2.5-7b branch's answer to EXAMPLE_REVIEW from the quick-start code, pasted
# from a real run; until one is recorded, None leaves it off the cards.
EXAMPLE_OUTPUT = None

# The prompt part of the quick start. render_quickstart() checks that it builds
# exactly the training prompt, src.inference.create_inference_prompt.
PROMPT_CODE = r'''SYSTEM_PROMPT = """@SYSTEM_PROMPT@"""


def build_prompt(review):
    # The ChatML text all three models were trained on. Do not use
    # tokenizer.apply_chat_template: Llama and DeepSeek have other templates.
    user = ("Quyidagi o'zbek tilidagi matndan aspektlarni, kategoriyalarni va hissiyot polaritesini aniqlang:"
            f'\n\nMatn: "{review}"')
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")'''

QUICKSTART = r'''```python
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

repo, branch = "@REPO@", "@BRANCH@"@OTHER_BRANCHES@
tokenizer = AutoTokenizer.from_pretrained(repo, revision=branch)
model = AutoModelForCausalLM.from_pretrained(repo, revision=branch, dtype="auto", device_map="auto")

@PROMPT_CODE@


inputs = tokenizer(build_prompt("@EXAMPLE@"), return_tensors="pt").to(model.device)
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
```'''

OUTPUT_FORMAT = """The answer holds two kinds of records, as UzABSA does:

- **term records**, `{"term": ..., "polarity": ...}`: a phrase of the review and its polarity;
- **category records**, `{"category": ..., "polarity": ...}`: one of five categories, `ovqat` (food), `muhit` (ambience), `xizmat` (service), `narx` (price) or `boshqalar` (other), and its polarity.

Polarity is `positive`, `negative` or `neutral`. UzABSA annotates terms and categories as independent layers, so a category record is not linked to a term. The scores count the term records."""

EVAL_NOTE = (
    f"The scores above come from `scripts/evaluate.py` in the [GitHub repository]({GITHUB_URL}), "
    "which loads the merged model in 4-bit with Unsloth and decodes greedily; bf16 outputs can differ slightly."
)

METRICS_NOTE = (
    "ATE is aspect term extraction, scored on exact matches (P: precision, R: recall) unless marked partial; "
    "partial F1 also credits a predicted term that contains the gold term or lies inside it, with one-to-one "
    "matching. Pair F1 counts a term as correct only with the right polarity. Sentiment accuracy and macro-F1 "
    "are computed on the exactly matched terms."
)

DATA_TEXT = (
    f"**Data.** [UzABSA](https://huggingface.co/datasets/{DATASET_ID}) holds 6,175 Uzbek restaurant-review "
    "sentences in the SemEval-2014 format. Conversion to instruction examples, with the *conflict* polarity "
    "removed, yields 6,089 examples, split 90/10 (seed 42) into 5,480 training and 609 held-out reviews. "
    "Identical normalised texts are kept on the same side of the split."
)

CORRECTED_RELEASE = """> [!IMPORTANT]
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
> Scores from the earlier weights are not comparable with the ones below."""

CITATION = """```bibtex
@misc{matlatipov2026uzabsallm,
  title        = {UzABSA-LLM: Fine-tuned Large Language Models for Uzbek Aspect-Based Sentiment Analysis},
  author       = {Matlatipov, Sanatbek},
  year         = {2026},
  howpublished = {\\url{https://huggingface.co/@REPO@}}
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
```""".replace("@REPO@", REPO_ID)


def render_quickstart(branch: str, other_branches: str = "") -> str:
    """The quick-start code for a branch, after checking that it builds the training prompt."""
    sys.path.insert(0, str(get_project_root()))
    from src.inference import SYSTEM_PROMPT_UZ, create_inference_prompt
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # src.inference turns on INFO logging, which floods the output

    prompt_code = PROMPT_CODE.replace("@SYSTEM_PROMPT@", SYSTEM_PROMPT_UZ)
    namespace = {}
    exec(prompt_code, namespace)
    if namespace["build_prompt"](EXAMPLE_REVIEW) != create_inference_prompt(EXAMPLE_REVIEW):
        sys.exit("The card's prompt differs from the training prompt in src/inference.py; update PROMPT_CODE")
    return (QUICKSTART.replace("@PROMPT_CODE@", prompt_code)
            .replace("@REPO@", REPO_ID)
            .replace("@BRANCH@", branch)
            .replace("@OTHER_BRANCHES@", other_branches)
            .replace("@EXAMPLE@", EXAMPLE_REVIEW))


def example_answer() -> str:
    if not EXAMPLE_OUTPUT:
        return ""
    return f"\nThe `qwen2.5-7b` model answers:\n\n```json\n{EXAMPLE_OUTPUT}\n```\n"


def training_table() -> str:
    c = TRAINING_CONFIG
    return f"""| Parameter | Value |
|-----------|-------|
| Framework | Unsloth + TRL (SFTTrainer) + PEFT |
| Quantization | {c['quantization']} |
| LoRA rank / alpha / dropout | {c['lora_r']} / {c['lora_alpha']} / {c['lora_dropout']} |
| Target modules | q, k, v, o, gate, up and down projections |
| Learning rate | {c['learning_rate']}, {c['scheduler']} schedule, {c['warmup_ratio']:.0%} warm-up |
| Effective batch size | {c['effective_batch_size']} ({c['batch_size']} × {c['grad_accum']} gradient accumulation) |
| Steps | {c['max_steps']:,} |
| Optimizer | `{c['optimizer']}`, weight decay {c['weight_decay']} |
| Precision | {c['precision']} |
| Max sequence length | {c['max_seq_length']:,} tokens |
| Seed | {c['seed']} |
| Checkpoint kept | lowest loss on the 609 held-out reviews, evaluated every 100 steps |"""


# ---------------------------------------------------------------------------
# Model card generators
# ---------------------------------------------------------------------------

# Columns of the results tables: (header, key in load_results())
_RESULT_COLUMNS = [
    ("ATE P", "p"), ("ATE R", "r"), ("ATE exact F1", "f1"), ("ATE partial F1", "partial"),
    ("Pair F1", "pair"), ("Sentiment acc.", "acc"), ("Sentiment macro-F1", "macro"), ("JSON parse", "parse"),
]

# Rows of the main-branch results table: (label, system name in the results files)
_RESULT_ROWS = [
    ("**Qwen 2.5-7B**, fine-tuned (`qwen2.5-7b`)", "Qwen 2.5-7B"),
    ("**Llama 3.1-8B**, fine-tuned (`llama3.1-8b`)", "Llama 3.1-8B"),
    ("**DeepSeek-R1-7B**, fine-tuned (`deepseek-r1-7b`)", "DeepSeek-R1-7B"),
    ("BERTbek, fine-tuned encoder", "BERTbek"),
    ("TahrirchiBERT, fine-tuned encoder", "TahrirchiBERT"),
    ("Majority vote of the five fine-tuned systems", VOTE_NAME),
    ("Qwen 2.5-7B, zero-shot", "Qwen 2.5-7B (ZS)"),
    ("Llama 3.1-8B, zero-shot", "Llama 3.1-8B (ZS)"),
    ("DeepSeek-R1-7B, zero-shot", "DeepSeek-R1-7B (ZS)"),
]
LLMS = [cfg["paper_name"] for cfg in MODELS.values()]
ENCODERS = ["BERTbek", "TahrirchiBERT"]


def _cell(key: str, value) -> str:
    if value is None:
        return "–"
    return f"{100 * value:.1f}%" if key == "parse" else f"{value:.4f}"


def build_results_table(results: dict) -> str:
    """Render the main-branch results table, bold marking the best single fine-tuned
    system in each column.

    Hardcoding these numbers previously let the published card drift away from the
    artifacts; generating them means the card is wrong only if the runs are.
    """
    fine_tuned = LLMS + ENCODERS
    best = {k: max(results[n][k] for n in fine_tuned if results[n][k] is not None) for _, k in _RESULT_COLUMNS}
    lines = ["| System | " + " | ".join(h for h, _ in _RESULT_COLUMNS) + " |",
             "|--------|" + "|".join([":---:"] * len(_RESULT_COLUMNS)) + "|"]
    for label, name in _RESULT_ROWS:
        cells = []
        for _, k in _RESULT_COLUMNS:
            v = results[name][k]
            bold = name in fine_tuned and v is not None and v == best[k]
            cells.append(f"**{_cell(k, v)}**" if bold else _cell(k, v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_model_table(results: dict, model_key: str) -> str:
    """One model's scores next to those of its zero-shot base model."""
    name = MODELS[model_key]["paper_name"]
    ft, zs = results[name], results[f"{name} (ZS)"]
    lines = ["| Metric | This model (fine-tuned) | Base model (zero-shot) |", "|--------|:---:|:---:|"]
    for header, k in _RESULT_COLUMNS:
        lines.append(f"| {header} | **{_cell(k, ft[k])}** | {_cell(k, zs[k])} |")
    return "\n".join(lines)


def generate_main_readme() -> str:
    """Generate the main branch README with project overview + comparison."""
    r = load_results()
    vote = r[VOTE_NAME]
    exact_gain = [r[n]["f1"] / r[f"{n} (ZS)"]["f1"] for n in LLMS]
    pair_gain = [r[n]["pair"] / r[f"{n} (ZS)"]["pair"] for n in LLMS]
    top = ["Qwen 2.5-7B", "Llama 3.1-8B"]
    pair_lead = max(r[a]["pair"] - r[b]["pair"] for a in top for b in ENCODERS)
    macro_lead = max(r[a]["macro"] - r[b]["macro"] for a in top for b in ENCODERS)

    model_rows = "\n".join(
        f"| **{cfg['paper_name']}** | `{cfg['branch']}` | "
        f"[{cfg['base_model_full']}](https://huggingface.co/{cfg['base_model_full']}) | "
        f"{LICENSE_NAMES[cfg['license']]} |"
        for cfg in MODELS.values())
    licenses = {}
    for cfg in MODELS.values():
        licenses.setdefault(cfg["license"], []).append(f"`{cfg['branch']}`")
    license_text = " ".join(
        f"**Built with Llama.** The {' and '.join(b)} branch is governed by the "
        "[Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) and the "
        "[Llama 3.1 Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/)."
        if lic == "llama3.1" else
        f"The {' and '.join(b)} {'branches are' if len(b) > 1 else 'branch is'} released under "
        f"{LICENSE_NAMES[lic]}."
        for lic, b in licenses.items())

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
  - {DATASET_ID}
pipeline_tag: text-generation
---

# UzABSA-LLM: Uzbek Aspect-Based Sentiment Analysis

Three open 7–8B LLMs fine-tuned with QLoRA for **aspect-based sentiment analysis (ABSA)** of Uzbek reviews. Given a review, a model returns JSON listing the **aspect terms** and **aspect categories** it mentions, each with its **sentiment polarity**.

## Models

Each model is stored on its own branch; load it with `revision="<branch>"`.

| Model | Branch | Base model | License |
|-------|--------|------------|---------|
{model_rows}

Each branch holds the merged model (bf16 safetensors, 15–16 GB), the LoRA adapter (`lora_adapters/`, about 165 MB), and the run's `eval_results.json` and `experiment_summary.json`. The branch cards also show how to load only the adapter, or the model in 4-bit with Unsloth.

## Results

Scores on the 609 held-out UzABSA reviews (737 gold aspect terms). Bold marks the best single fine-tuned system in each column.

{build_results_table(r)}

{METRICS_NOTE}

- **Fine-tuning is essential:** compared with zero-shot prompting of the same base models, it multiplies exact ATE F1 by {min(exact_gain):.1f}–{max(exact_gain):.1f} and pair F1 by {min(pair_gain):.1f}–{max(pair_gain):.1f}.
- **Qwen and Llama form the top tier and are statistically tied** (paired bootstrap), so either is a sound choice. Qwen is the only LLM that returns valid JSON for every review.
- **DeepSeek trails both.** Its pair-F1 gap to Llama is the only difference between fine-tuned systems that stays significant after Holm correction; it misses more aspects (lower recall) rather than producing broken output.
- **Encoders versus LLMs:** on extraction alone, the 110M-parameter Uzbek BERT encoders are competitive (TahrirchiBERT has the highest precision and BERTbek the highest recall of the single systems). On joint prediction and sentiment, Qwen and Llama score higher than both encoders, by up to {100 * pair_lead:.1f} pair-F1 and {100 * macro_lead:.1f} macro-F1 points.
- **A majority vote** that keeps a term when at least three of the five fine-tuned systems predict it reaches {vote['f1']:.3f} exact ATE F1 and {vote['pair']:.3f} pair F1, significantly above every single system, mainly through higher precision ({vote['p']:.3f}).

## Quick start

```bash
pip install transformers accelerate
```

{render_quickstart("qwen2.5-7b", '  # or "llama3.1-8b", "deepseek-r1-7b"')}
{example_answer()}
{EVAL_NOTE}

## Output format

{OUTPUT_FORMAT}

## Training

The three models share one configuration, fixed in advance:

{training_table()}

{DATA_TEXT}

**Hardware.** One NVIDIA RTX A6000 (48 GB), CUDA 12.4, PyTorch 2.6.0.

**Corrected release (2026-08-29).** All three models were retrained after two data-pipeline fixes: the training examples had carried the literal string `"None"` as their system prompt, and 61 of the 609 held-out reviews had an exact-text twin in training. Scores of weights uploaded before that date are not comparable with the ones above; each branch card has the details.

## Reproduction and seed variance

The scores above come from one run per system, and those runs' checkpoints were chosen on the same 609 reviews they are scored on. A stricter rerun is under way in [`experiments/v3/`]({GITHUB_URL}/tree/main/experiments/v3): it carves a separate dev split out of the training data, chooses every checkpoint and the corpus annotator on dev alone, and repeats each system over three seeds so results can be reported as mean ± sd rather than a single number. The test split is these same 609 reviews, byte for byte, so the two sets of numbers stay comparable.

Expect small downward corrections wherever a published score benefited from choosing a checkpoint on the test set. `experiments/v3/SUMMARY.md` and `experiments/v3/registry.json` hold the current state, both rebuilt from the prediction files rather than written by hand.

## Code and data

- **Code** (training, evaluation, significance tests and the annotation pipeline): [{GITHUB_URL.removeprefix("https://")}]({GITHUB_URL})
- **Training data:** [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID})
- **Multi-domain silver corpus** (5,038 reviews from 23 business domains, annotated by an earlier Qwen checkpoint and scored by a panel of LLM judges): [doi:{CORPUS_DOI}](https://doi.org/{CORPUS_DOI})

## Citation

{CITATION}

## License

{license_text}
"""
    return card.strip() + "\n"


def generate_model_readme(model_key: str) -> str:
    """Generate a model card for a specific model branch."""
    cfg = MODELS[model_key]
    results = load_results()
    s = results[cfg["paper_name"]]
    notice = LLAMA_NOTICE + "\n\n" if cfg["license"] == "llama3.1" else ""
    example = example_answer() if model_key == "qwen2.5-7b" else ""

    card = f"""---
language:
  - uz
license: {cfg['license']}
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
  - {DATASET_ID}
pipeline_tag: text-generation
model-index:
  - name: {cfg['display_name']}
    results:
      - task:
          type: token-classification
          name: Aspect Term Extraction
        dataset:
          name: UzABSA (SemEval-2014 format)
          type: {DATASET_ID}
          split: validation
        metrics:
          - type: f1
            value: {s['f1']}
            name: ATE Exact F1
          - type: f1
            value: {s['partial']}
            name: ATE Partial F1 (one-to-one)
          - type: f1
            value: {s['pair']}
            name: Pair F1
      - task:
          type: text-classification
          name: Aspect Sentiment Classification
        dataset:
          name: UzABSA (SemEval-2014 format)
          type: {DATASET_ID}
          split: validation
        metrics:
          - type: accuracy
            value: {s['acc']}
            name: Sentiment Accuracy
          - type: f1
            value: {s['macro']}
            name: Sentiment Macro-F1
---

# {cfg['display_name']}

{notice}{'An' if cfg['params'].startswith('8') else 'A'} **{cfg['params']}**-parameter LLM fine-tuned with QLoRA for **aspect-based sentiment analysis (ABSA)** of Uzbek reviews. Given a review, it returns JSON listing the aspect terms and aspect categories it mentions, each with its sentiment polarity.

> This is the **`{cfg['branch']}`** branch of [{REPO_ID}](https://huggingface.co/{REPO_ID}). The [main page](https://huggingface.co/{REPO_ID}) compares the three models.

{CORRECTED_RELEASE}

## Model Details

| Property | Value |
|----------|-------|
| Base model | [{cfg['base_model_full']}](https://huggingface.co/{cfg['base_model_full']}) |
| Architecture | {cfg['architecture']} |
| Parameters | {cfg['params']} |
| Fine-tuning method | QLoRA (4-bit NF4), Unsloth + TRL |
| Task | Aspect-based sentiment analysis |
| Language | Uzbek (uz) |
| Merged model size | {cfg['merged_size']} (bf16 safetensors) |
| LoRA adapter size | {cfg['lora_size']} (`lora_adapters/`) |

## Evaluation Results

Scores on the 609 held-out UzABSA reviews (737 gold aspect terms), next to the base model prompted zero-shot with the same prompt:

{build_model_table(results, model_key)}

{METRICS_NOTE} `eval_results.json` holds the scores written at training time; its partial F1 uses greedy instead of one-to-one matching.

## Usage

### Merged model

```bash
pip install transformers accelerate
```

{render_quickstart(cfg['branch'])}
{example}
{EVAL_NOTE}

### 4-bit with Unsloth, as in the evaluation

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "{REPO_ID}",
    revision="{cfg['branch']}",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
# then build the prompt and generate as above
```

### LoRA adapter only ({cfg['lora_size']})

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{cfg['base_model_full']}", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{cfg['base_model_full']}")
model = PeftModel.from_pretrained(base, "{REPO_ID}", revision="{cfg['branch']}", subfolder="lora_adapters")
# then build the prompt and generate as above
```

## Output Format

{OUTPUT_FORMAT}

## Training Configuration

{training_table()}

{DATA_TEXT}

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

Training, evaluation and significance tests: [{GITHUB_URL.removeprefix("https://")}]({GITHUB_URL})

## Citation

{CITATION}

## License

{LLAMA_NOTICE if cfg['license'] == 'llama3.1' else LICENSE_NAMES[cfg['license']] + '.'}
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


def push_model_branch(model_key: str, repo_id: str, dry_run: bool = False, card_only: bool = False):
    """Push a model branch with merged weights + LoRA adapters, or only its card."""
    from huggingface_hub import HfApi

    cfg = MODELS[model_key]
    api = HfApi()
    root = get_project_root()
    model_dir = root / cfg["local_dir"]
    merged_dir = model_dir / "merged_model"
    lora_dir = model_dir / "lora_adapters"
    branch = cfg["branch"]
    preview_path = root / "outputs" / "hub_preview" / f"{branch}_README.md"

    print(f"\n{'=' * 60}")
    print(f"  Pushing {cfg['display_name']} to {repo_id} (branch: {branch}){' [card only]' if card_only else ''}")
    print(f"{'=' * 60}")

    # Generate model card
    readme_content = generate_model_readme(model_key)

    if card_only:
        if dry_run:
            print(f"\n  [DRY RUN] Would upload README.md to branch '{branch}'")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_text(readme_content, encoding="utf-8")
            print(f"  Preview saved to: {preview_path}")
            return True
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message=f"Update model card for {cfg['display_name']}",
        )
        print(f"\n  ✓ Card updated: https://huggingface.co/{repo_id}/tree/{branch}")
        return True

    # Validate paths
    if not merged_dir.exists():
        print(f"  ERROR: Merged model not found at {merged_dir}")
        return False
    if not lora_dir.exists():
        print(f"  ERROR: LoRA adapters not found at {lora_dir}")
        return False

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

    # Weights first and the card last, so an interrupted upload never leaves a
    # card (or an experiment_summary.json) that describes weights not yet there
    print(f"\n  Uploading model files (this may take a while for {cfg['merged_size']})...")

    # Upload merged model folder
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Upload merged model weights for {cfg['display_name']}",
        ignore_patterns=["README.md"],
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

    print(f"\n  Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Add model card for {cfg['display_name']}",
    )

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
  python scripts/push_to_hub.py --all --card-only        # Update the cards, not the weights
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
    parser.add_argument(
        "--card-only", action="store_true",
        help="Upload only the model cards (README.md), not the weights"
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
            push_model_branch(branch, args.repo_id, dry_run=args.dry_run, card_only=args.card_only)

    print(f"\n{'#' * 60}")
    print(f"  Done! Visit: https://huggingface.co/{args.repo_id}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
