# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UzABSA-LLM is a research codebase (targeting an academic paper, currently in revision — see `paper_materials/revision_v2/REVISION_LOG.md`) for Uzbek Aspect-Based Sentiment Analysis (ABSA). It fine-tunes open-source LLMs (Qwen 2.5-7B, Llama 3.1-8B, DeepSeek-R1-Distill-7B) via QLoRA/Unsloth to extract aspect terms, categories, and sentiment polarities from Uzbek review text as structured JSON, and compares them against fine-tuned Uzbek BERT encoder baselines (BIO-tagging).

There is no application server or frontend here — this is an ML research pipeline: data prep → training → evaluation → significance testing → paper artifacts.

## Environment Setup

PyTorch **must** be installed with the CUDA index URL first, or it silently installs CPU-only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

For the BERT baselines specifically (Mac M2/Apple Silicon, no CUDA), see `paper_materials/revision_v2/baselines/README_RUN_ON_MAC.md` and `paper_materials/revision_v2/baselines/requirements_bert_baseline.txt` — these run on MPS, not the CUDA/Unsloth stack above.

Verify GPU setup: `python -m src.gpu_config --check` (also `--recommend`, `--estimate <billions>`).

There is no automated test suite (no `pytest` files, no CI config) — validation happens via the scripts' own `--max-samples`/`--max-examples` quick-run flags and, for `significance_test.py`, a built-in `--selftest`.

## Common Commands

### Dataset preparation (LLM pipeline)
```bash
python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv --analyze
python scripts/prepare_complete_dataset.py --max-examples 100 --output-dir ./data/test_processed   # quick
python scripts/prepare_complete_dataset.py --output-dir ./data/processed                            # full
```

### Training (LLM, QLoRA + Unsloth)
```bash
python scripts/train_unsloth.py --model qwen2.5-7b --dataset ./data/processed \
    --batch-size 4 --grad-accum 4 --max-steps 1000 --learning-rate 2e-4 \
    --output-dir ./outputs/my_run --run-name "qwen2.5-7b-v1"
```
Model shorthands (`--model`) are resolved in `src/gpu_config.py`/`scripts/train_unsloth.py` — see README.md "Supported Models" table (qwen2.5-7b/14b/32b, llama3-8b, llama3.1-8b, llama3.2-3b, deepseek-7b/14b, mistral-7b, gemma2-9b). Use `--model-path` to override with an arbitrary HF path.

### Evaluation & significance
```bash
python scripts/evaluate.py --model-path <run>/merged_model --test-data ./data/processed --output-dir <run>
python scripts/dump_llm_preds.py   # emit per-example preds.jsonl for significance testing
python scripts/significance_test.py --a a_preds.jsonl --a-name Qwen --b b_preds.jsonl --b-name Llama \
    --out paper_materials/revision_v2/significance/qwen_vs_llama.json
python scripts/significance_test.py --selftest   # sanity-check the bootstrap implementation
```

### BERT baselines (encoder, BIO-polarity token classification, runs on Mac MPS)
```bash
python scripts/prepare_bio_dataset.py                 # build data/bio_processed/ (BIO tags, same train/val split as LLMs)
python scripts/train_bert_baseline.py --model tahrirchi/tahrirchi-bert-base \
    --out paper_materials/revision_v2/baselines/tahrirchi-bert
python scripts/eval_bert_baseline.py \
    --model-dir paper_materials/revision_v2/baselines/tahrirchi-bert/model \
    --data ./data/processed --out paper_materials/revision_v2/baselines/tahrirchi-bert
```

### Annotation pipeline (silver-standard multi-domain dataset)
```bash
python scripts/annotate_reviews.py --model-path <merged_model>          # Layer 1: batch-annotate raw reviews (resumable)
python scripts/llm_judge.py --annotations ./data/annotated/reviews_annotated.json \
    --provider openai --model gpt-4o-mini --sample-size 300 --output-dir ./data/judged   # Layer 2: LLM-as-Judge
python scripts/assemble_dataset.py --annotations ./data/annotated/reviews_annotated.json \
    --judge-results ./data/judged/judge_results.json --output-dir ./data/final_dataset   # Layer 3: quality-tiered assembly
```

### Publishing
```bash
python scripts/push_to_hub.py --all --dry-run
python scripts/push_to_hub.py --branch qwen2.5-7b   # models are branches of one HF repo, not separate repos
```

## Architecture

### `src/` — shared library, imported by scripts
- `data_prep.py` — loads the HF dataset `Sanatbek/aspect-based-sentiment-analysis-uzbek`, converts to ChatML instruction/response pairs, creates train/val splits.
- `dataset_utils.py` — raw CSV review loading/cleaning (`sharh.commeta.uz` scrape), stats, merging raw+annotated.
- `format_converter.py` — converts SemEVAL-2014-style annotations to the instruction-tuning JSON target format.
- `evaluation.py` — the canonical metric implementations (ATE exact/partial F1, aspect-sentiment pair F1, sentiment accuracy). Both the LLM eval (`scripts/evaluate.py`) and the BERT eval (`scripts/eval_bert_baseline.py`) import from here via `importlib` so results are directly comparable across model families.
- `inference.py` — model loading + single/batch inference; contains the canonical system prompts (`SYSTEM_PROMPT_UZ`) that **must match what training used** — changing them requires retraining, not just re-inference.
- `gpu_config.py` — GPU detection, batch-size recommendations, the model-shorthand → HF-path table.
- `training_metrics.py` — Unsloth/Trainer callback that logs loss/LR/GPU-mem per step and produces paper-ready plots (`training_curves.png`, 300 DPI) + `experiment_summary.json` for reproducibility.

### `scripts/` — CLI entry points, one per pipeline stage
Two parallel model families are evaluated on the **same** 609-example validation split for a fair comparison:
1. **LLM family** (decoder, generative JSON output): `prepare_complete_dataset.py` → `train_unsloth.py` → `evaluate.py` → `dump_llm_preds.py`.
2. **BERT family** (encoder, BIO-polarity token classification): `prepare_bio_dataset.py` → `train_bert_baseline.py` → `eval_bert_baseline.py`.

Both families' per-example predictions feed `significance_test.py`, which runs paired bootstrap tests (not naive metric comparison) — this exists specifically because early results showed near-tied metrics (e.g. Qwen vs Llama pair-F1 gap of 0.001) that needed a real significance test rather than a bolded "winner" in the paper.

`annotate_reviews.py` → `llm_judge.py` → `assemble_dataset.py` is a separate three-layer pipeline for producing a **silver-standard multi-domain dataset** (23 business domains) from unannotated reviews, distinct from the training/eval pipeline above: a fine-tuned model annotates, an external LLM-as-Judge scores quality per example, and assembly filters into include/flag/exclude tiers by score threshold.

### Data flow directories
- `data/raw/` — original scraped reviews + taxonomy JSONs (`absa_subcategories.json`, `business_categories.json`).
- `data/processed/` / `data/test_processed/` — ChatML-formatted train/val splits used for LLM training and evaluation.
- `data/bio_processed/` — BIO-tag-aligned version of the same split, used for BERT baseline training/eval.
- `data/annotated/` → `data/judged/` → `data/final_dataset/` — outputs of the three-layer silver-dataset pipeline.
- `outputs/` — training run artifacts (`merged_model/`, `lora_adapters/`, checkpoints, curves, `experiment_summary.json`).
- `paper_materials/` — LaTeX sources for submitted/revised papers (MDPI, Springer, current `revision_v2`) plus baseline run artifacts, human-validation data, and significance-test outputs referenced directly in the paper tables. Treat `revision_v2/REVISION_LOG.md` and `revision_v2/writing/pending_edits.md` as the source of truth for what's still open in the current revision.
- `RESEARCH_LOG.md` — the authoritative methodology/experiment log (research questions, dataset stats, config decisions, results). Consult this before changing methodology, metrics, or hyperparameters, since it documents *why* choices were made, not just what they are.

### Key invariant
Evaluation numbers across model families (LLM vs BERT) and across papers are only comparable because everything is evaluated against the **same held-out reference set** — reconstructed from the original ChatML validation split, not a per-model-format subset. When adding a new baseline or model, reuse `src/evaluation.py`'s metric functions and the existing 609-example split rather than recomputing references.
