#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Dump per-example LLM predictions  (P2-B, feeds P2-C)
# =============================================================================
"""
Regenerates a fine-tuned LLM's per-example predictions on the 609-example split in
the SAME JSONL format as ``eval_bert_baseline.py``, so paired-bootstrap significance
tests (``significance_test.py``) can compare LLMs to each other and to the BERT
baselines. The original evaluation did live inference without persisting per-example
predictions; this fills that gap.

Runs in the project env. On a Mac M2 Max, load the **merged fp16** model (each HF
branch of ``Sanatbek/UzABSA-LLM`` ships one) — 4-bit/bitsandbytes is unavailable on
Apple Silicon. Expect ~30-60 min per model over 609 examples.

Usage (merged model from HF Hub branch):
    python scripts/dump_llm_preds.py \
        --model Sanatbek/UzABSA-LLM --revision qwen2.5-7b \
        --data ./data/processed \
        --out paper_materials/revision_v2/results/qwen_preds.jsonl

Usage (base model, no adapter — for zero-shot baseline, P2-A):
    python scripts/dump_llm_preds.py --model Qwen/Qwen2.5-7B-Instruct \
        --out paper_materials/revision_v2/results/qwen_zeroshot_preds.jsonl

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
from pathlib import Path

_ev = ilu.spec_from_file_location(
    "uzabsa_evaluation", Path(__file__).parent.parent / "src" / "evaluation.py")
_evaluation = ilu.module_from_spec(_ev); _ev.loader.exec_module(_evaluation)
parse_chatml_example = _evaluation.parse_chatml_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path (merged model or base)")
    ap.add_argument("--revision", default=None, help="HF branch (e.g. qwen2.5-7b)")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    ap.add_argument("--data", default="./data/processed")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import load_from_disk
    # load_model/extract_aspects import transformers/unsloth lazily inside src.inference
    from src.inference import load_model, extract_aspects

    model_path = args.model
    # allow revision by passing "model@revision" style if load_model doesn't accept revision
    if args.revision:
        print(f"NOTE: loading {args.model} (revision {args.revision}). If load_model does "
              f"not pass revision through, clone the branch locally first.")
    model, tokenizer = load_model(model_path, adapter_path=args.adapter)

    ds = load_from_disk(args.data)
    split = ds[args.split] if args.split in ds else ds[list(ds.keys())[0]]
    if args.max_samples:
        split = split.select(range(min(args.max_samples, len(split))))
    print(f"Dumping predictions for {len(split)} {args.split} examples -> {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_parse_ok = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(split):
            text, gold = parse_chatml_example(ex["text"])
            gold_terms = [a for a in gold if a.get("term")]
            result = extract_aspects(model, tokenizer, text, use_uzbek=True)
            pred = [a for a in result.get("aspects", []) if a.get("term")]
            n_parse_ok += 1 if result.get("parse_success") else 0
            f.write(json.dumps({"idx": i, "text": text, "pred_aspects": pred,
                                "gold_aspects": gold_terms,
                                "parse_success": result.get("parse_success", False)},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(split)}  parse_ok={n_parse_ok}")
    print(f"Done. JSON parse rate: {n_parse_ok}/{len(split)} = {n_parse_ok/len(split):.1%}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
