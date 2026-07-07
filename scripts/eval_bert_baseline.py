#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Encoder (BERT) Baseline Evaluation  (P1)
# =============================================================================
"""
Evaluates a fine-tuned BERT ABSA baseline on the SAME 609-example split the LLMs
were scored on, producing metrics directly comparable to the paper's Table.

Crucially, the *reference* aspects are reconstructed from the original ChatML
validation split (ALL term-bearing gold aspects, via parse_chatml_example) — the
identical reference set used for the LLM evaluation — NOT the reduced BIO-aligned
subset. This keeps the head-to-head fair.

Pipeline: for each validation review -> tokenize text into words -> BERT token
classification -> decode BIO-polarity tags into {term, polarity} -> compare to gold
with the shared metric functions from src/evaluation.py. Saves per-example
predictions for the significance tests (P2-C).

Outputs (to --out):
    eval_results.json          ATE exact/partial, pair F1, sentiment, matching the LLM table
    preds_per_example.jsonl    {review_id?, text, pred_aspects, gold_aspects}

Usage:
    python scripts/eval_bert_baseline.py \
        --model-dir paper_materials/revision_v2/baselines/tahrirchi-bert/model \
        --data ./data/processed \
        --out paper_materials/revision_v2/baselines/tahrirchi-bert

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
from pathlib import Path

import numpy as np

# Reuse metric + parsing + BIO helpers without the heavy package __init__.
_root = Path(__file__).parent.parent
_ev = ilu.spec_from_file_location("uzabsa_evaluation", _root / "src" / "evaluation.py")
_evaluation = ilu.module_from_spec(_ev); _ev.loader.exec_module(_evaluation)
compute_ate_metrics = _evaluation.compute_ate_metrics
compute_aspect_polarity_metrics = _evaluation.compute_aspect_polarity_metrics
parse_chatml_example = _evaluation.parse_chatml_example

_bp = ilu.spec_from_file_location("uzabsa_bio", _root / "scripts" / "prepare_bio_dataset.py")
_bio = ilu.module_from_spec(_bp); _bp.loader.exec_module(_bio)
tokenize_words = _bio.tokenize_words
decode_bio = _bio.decode_bio
LABELS = _bio.LABELS


def get_device(pref):
    import torch
    if pref == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def predict_words(model, tokenizer, words, device, max_length=192):
    """Return one predicted tag id per input word (first-subword prediction)."""
    import torch
    if not words:
        return []
    enc = tokenizer([words], is_split_into_words=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred = logits.argmax(-1).tolist()
    tags, prev = [], None
    for wid, p in zip(word_ids, pred):
        if wid is None or wid == prev:
            prev = wid
            continue
        tags.append(p)
        prev = wid
    # pad/truncate to len(words) (truncation may drop trailing words)
    if len(tags) < len(words):
        tags += [0] * (len(words) - len(tags))
    return tags[:len(words)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", default="./data/processed")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    device = get_device(args.device if args.device != "auto" else "auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device).eval()
    print(f"Loaded {args.model_dir} on {device}")

    ds = load_from_disk(args.data)
    split = ds[args.split] if args.split in ds else ds[list(ds.keys())[0]]
    print(f"Evaluating on {len(split)} {args.split} examples")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    preds, refs = [], []
    fpred = open(out / "preds_per_example.jsonl", "w", encoding="utf-8")
    for i, ex in enumerate(split):
        text, gold = parse_chatml_example(ex["text"])
        gold_terms = [a for a in gold if a.get("term")]
        words = tokenize_words(text)
        tag_ids = predict_words(model, tokenizer, words, device, args.max_length)
        pred_aspects = decode_bio(words, tag_ids)
        preds.append(pred_aspects)
        refs.append(gold_terms)
        fpred.write(json.dumps({"idx": i, "text": text,
                                "pred_aspects": pred_aspects,
                                "gold_aspects": gold_terms}, ensure_ascii=False) + "\n")
    fpred.close()

    pred_terms = [[a["term"] for a in p] for p in preds]
    ref_terms = [[a["term"] for a in r] for r in refs]
    results = {
        "model_dir": args.model_dir,
        "num_examples": len(preds),
        "aspect_term_extraction": {
            "exact_match": compute_ate_metrics(pred_terms, ref_terms),
            "partial_match": compute_ate_metrics(pred_terms, ref_terms, partial_match=True),
        },
        "aspect_polarity_pairs": compute_aspect_polarity_metrics(preds, refs),
    }
    json.dump(results, open(out / "eval_results.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out/'eval_results.json'} and per-example preds "
          f"-> {out/'preds_per_example.jsonl'}")


if __name__ == "__main__":
    main()
