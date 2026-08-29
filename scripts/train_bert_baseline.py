#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Encoder (BERT) Baseline Trainer  (P1)
# =============================================================================
"""
Fine-tunes an Uzbek BERT encoder as a joint ABSA baseline via BIO-polarity token
classification. Designed to run on a Mac M2 Max (Apple MPS) — no CUDA / bitsandbytes.

Recommended encoders (both BERT-base, ~110M params, fit comfortably on M2 Max):
    tahrirchi/tahrirchi-bert-base
    elmurod1202/bertbek-news-big-cased

Consumes the BIO dataset from ``scripts/prepare_bio_dataset.py`` (data/bio_processed/)
and trains on the SAME train split the LLMs used, so the 609-example evaluation is
directly comparable.

Usage (run once per model):
    python scripts/prepare_bio_dataset.py            # build data/bio_processed first
    python scripts/train_bert_baseline.py --model tahrirchi/tahrirchi-bert-base \
        --out paper_materials/revision_v2/baselines/tahrirchi-bert
    python scripts/train_bert_baseline.py --model elmurod1202/bertbek-news-big-cased \
        --out paper_materials/revision_v2/baselines/bertbek

All console output is also written to <out>/training_log.txt.
If MPS runs out of memory, lower --batch-size (try 8 or 4) or add --device cpu.

Author: UzABSA Team
License: MIT
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def get_device(pref):
    import torch
    if pref == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def align_labels(tokenizer, examples, label_all_subwords=False, max_length=192):
    """Tokenize pre-split words and align BIO word-labels to subword tokens.
    Only the first subword of each word gets the label; the rest get -100."""
    tok = tokenizer(examples["tokens"], is_split_into_words=True,
                    truncation=True, max_length=max_length)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        prev, out = None, []
        for wid in word_ids:
            if wid is None:
                out.append(-100)
            elif wid != prev:
                out.append(labels[wid])
            else:
                out.append(labels[wid] if label_all_subwords else -100)
            prev = wid
        all_labels.append(out)
    tok["labels"] = all_labels
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (Uzbek BERT)")
    ap.add_argument("--bio-dir", default="./data/bio_processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out / "training_log.txt", mode="w")])
    log = logging.getLogger("bert-baseline")

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments, set_seed)
    # NOTE: we call seqeval directly rather than the HuggingFace `evaluate` library.
    # Running `python scripts/train_bert_baseline.py` puts scripts/ on sys.path[0], which
    # shadows the pip `evaluate` package with this repo's scripts/evaluate.py — so
    # `import evaluate` would fail with "no attribute 'load'". seqeval avoids the clash.
    from seqeval.metrics import (precision_score as seq_precision,
                                 recall_score as seq_recall,
                                 f1_score as seq_f1,
                                 accuracy_score as seq_accuracy)

    set_seed(args.seed)
    device = get_device(args.device if args.device != "auto" else "auto")
    log.info(f"Model: {args.model} | device: {device} | torch {torch.__version__}")

    bio = Path(args.bio_dir)
    label2id = json.load(open(bio / "label_map.json"))
    id2label = {v: k for k, v in label2id.items()}
    train = json.load(open(bio / "train.json", encoding="utf-8"))
    val = json.load(open(bio / "validation.json", encoding="utf-8"))
    log.info(f"train={len(train)} val={len(val)} labels={list(label2id)}")

    # TahrirchiBERT ships a RoBERTa-style BPE tokenizer, which refuses
    # is_split_into_words=True unless add_prefix_space is set. WordPiece
    # tokenizers accept and ignore the flag, so pass it unconditionally and
    # fall back only for tokenizers that reject the kwarg outright.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    except (TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds_train = Dataset.from_list(train).map(
        lambda e: align_labels(tokenizer, e, max_length=args.max_length),
        batched=True, remove_columns=["tokens", "ner_tags", "text", "aspects"])
    ds_val = Dataset.from_list(val).map(
        lambda e: align_labels(tokenizer, e, max_length=args.max_length),
        batched=True, remove_columns=["tokens", "ner_tags", "text", "aspects"])

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(label2id), id2label=id2label, label2id=label2id)

    label_list = [id2label[i] for i in range(len(id2label))]

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        true_preds, true_labs = [], []
        for pred, lab in zip(preds, p.label_ids):
            tp = [label_list[pr] for pr, l in zip(pred, lab) if l != -100]
            tl = [label_list[l] for pr, l in zip(pred, lab) if l != -100]
            true_preds.append(tp); true_labs.append(tl)
        # seqeval signature is (y_true, y_pred)
        return {"precision": seq_precision(true_labs, true_preds),
                "recall": seq_recall(true_labs, true_preds),
                "f1": seq_f1(true_labs, true_preds),
                "accuracy": seq_accuracy(true_labs, true_preds)}

    targs = TrainingArguments(
        output_dir=str(out / "hf_trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=25, seed=args.seed,
        fp16=False, bf16=False,  # MPS: keep fp32
        dataloader_pin_memory=False, report_to=[],
        use_cpu=(device == "cpu"),
    )
    # Newer transformers (>=4.46) renamed Trainer's `tokenizer` arg to `processing_class`;
    # older versions only accept `tokenizer`. Pick whichever this install supports.
    import inspect
    trainer_kwargs = dict(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    log.info("Starting training...")
    trainer.train()
    metrics = trainer.evaluate()
    log.info(f"Final seqeval (token-span) metrics: {metrics}")

    save_dir = out / "model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    json.dump({"model": args.model, "seqeval_span_metrics": metrics,
               "args": vars(args)},
              open(out / "train_summary.json", "w"), indent=2, default=str)
    log.info(f"Saved model -> {save_dir}")
    log.info("NOTE: seqeval numbers above are token-span level; run "
             "eval_bert_baseline.py for the ABSA metrics comparable to the LLMs.")


if __name__ == "__main__":
    main()
