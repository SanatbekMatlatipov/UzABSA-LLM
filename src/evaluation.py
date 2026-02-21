#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Evaluation Module
# =============================================================================
"""
Evaluation module for Uzbek Aspect-Based Sentiment Analysis.

This module provides metrics and utilities for evaluating:
1. Aspect Term Extraction (ATE) - F1, Precision, Recall
2. Aspect Category Detection (ACD) - Accuracy, F1
3. Sentiment Polarity Classification - Accuracy, Macro-F1

Author: UzABSA Team
License: MIT
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Aspect Term Extraction Metrics
# =============================================================================

def compute_ate_metrics(
    predictions: List[List[str]],
    references: List[List[str]],
    partial_match: bool = False,
) -> Dict[str, float]:
    """
    Compute Aspect Term Extraction (ATE) metrics.

    Evaluates how well the model identifies aspect terms in text.

    Args:
        predictions: List of predicted aspect term lists.
        references: List of ground truth aspect term lists.
        partial_match: If True, use partial string matching.

    Returns:
        Dictionary with precision, recall, and F1 scores.

    Example:
        >>> preds = [["telefon", "batareya"], ["narx"]]
        >>> refs = [["telefon", "ekran"], ["narx", "sifat"]]
        >>> metrics = compute_ate_metrics(preds, refs)
        >>> print(metrics)
        {"precision": 0.67, "recall": 0.50, "f1": 0.57}
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for pred_terms, ref_terms in zip(predictions, references):
        # Normalize terms (lowercase, strip whitespace)
        pred_set = set(t.lower().strip() for t in pred_terms)
        ref_set = set(t.lower().strip() for t in ref_terms)
        
        if partial_match:
            # Partial matching: term is correct if it's a substring or superstring
            matched_refs = set()
            for pred in pred_set:
                for ref in ref_set:
                    if pred in ref or ref in pred:
                        matched_refs.add(ref)
                        break
            
            true_positives = len(matched_refs)
            false_positives = len(pred_set) - true_positives
            false_negatives = len(ref_set) - true_positives
        else:
            # Exact matching
            true_positives = len(pred_set & ref_set)
            false_positives = len(pred_set - ref_set)
            false_negatives = len(ref_set - pred_set)
        
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
    
    # Compute metrics
    precision = (
        total_true_positives / (total_true_positives + total_false_positives)
        if (total_true_positives + total_false_positives) > 0
        else 0.0
    )
    
    recall = (
        total_true_positives / (total_true_positives + total_false_negatives)
        if (total_true_positives + total_false_negatives) > 0
        else 0.0
    )
    
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# =============================================================================
# Aspect-Polarity Pair Metrics
# =============================================================================

def compute_aspect_polarity_metrics(
    predictions: List[List[Dict[str, str]]],
    references: List[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """
    Compute metrics for aspect-polarity pair extraction.

    Evaluates both aspect term extraction and sentiment classification together.
    An aspect is correct only if both term and polarity match.

    Args:
        predictions: List of predicted aspect lists (each aspect has term, polarity).
        references: List of ground truth aspect lists.

    Returns:
        Dictionary with pair-level and individual metrics.

    Example:
        >>> preds = [[{"term": "telefon", "polarity": "positive"}]]
        >>> refs = [[{"term": "telefon", "polarity": "positive"}]]
        >>> metrics = compute_aspect_polarity_metrics(preds, refs)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    polarity_true = []
    polarity_pred = []
    
    for pred_aspects, ref_aspects in zip(predictions, references):
        # Create sets of (term, polarity) tuples
        pred_pairs = set(
            (a.get("term", "").lower().strip(), a.get("polarity", "").lower().strip())
            for a in pred_aspects
        )
        ref_pairs = set(
            (a.get("term", "").lower().strip(), a.get("polarity", "").lower().strip())
            for a in ref_aspects
        )
        
        # Count matches
        matches = pred_pairs & ref_pairs
        total_tp += len(matches)
        total_fp += len(pred_pairs - ref_pairs)
        total_fn += len(ref_pairs - pred_pairs)
        
        # Collect polarities for matched terms only
        pred_terms = {t for t, p in pred_pairs}
        ref_terms = {t for t, p in ref_pairs}
        
        for term in pred_terms & ref_terms:
            pred_pol = next((p for t, p in pred_pairs if t == term), None)
            ref_pol = next((p for t, p in ref_pairs if t == term), None)
            if pred_pol and ref_pol:
                polarity_pred.append(pred_pol)
                polarity_true.append(ref_pol)
    
    # Pair-level metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Sentiment classification metrics (for matched aspects)
    sentiment_accuracy = (
        accuracy_score(polarity_true, polarity_pred)
        if polarity_true
        else 0.0
    )
    sentiment_f1 = (
        f1_score(polarity_true, polarity_pred, average="macro", zero_division=0)
        if polarity_true
        else 0.0
    )
    
    return {
        "pair_precision": round(precision, 4),
        "pair_recall": round(recall, 4),
        "pair_f1": round(f1, 4),
        "sentiment_accuracy": round(sentiment_accuracy, 4),
        "sentiment_macro_f1": round(sentiment_f1, 4),
        "total_predictions": total_tp + total_fp,
        "total_references": total_tp + total_fn,
    }


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    text_column: str = "text",
    aspects_column: str = "aspects",
    batch_size: int = 8,
    use_uzbek: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation on a test dataset.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        test_dataset: Dataset with test examples.
        text_column: Column name for input text.
        aspects_column: Column name for ground truth aspects.
        batch_size: Batch size for inference.
        use_uzbek: Whether to use Uzbek prompts.

    Returns:
        Comprehensive evaluation results.
    """
    from .inference import extract_aspects
    from tqdm import tqdm
    
    logger.info(f"Evaluating on {len(test_dataset)} examples...")
    
    all_predictions = []
    all_references = []
    
    for example in tqdm(test_dataset, desc="Evaluating"):
        text = example[text_column]
        ref_aspects = example[aspects_column]
        
        # Handle string JSON
        if isinstance(ref_aspects, str):
            ref_aspects = json.loads(ref_aspects)
        
        # Run inference
        result = extract_aspects(model, tokenizer, text, use_uzbek)
        pred_aspects = result.get("aspects", [])
        
        all_predictions.append(pred_aspects)
        all_references.append(ref_aspects)
    
    # Compute metrics
    pred_terms = [[a.get("term", "") for a in aspects] for aspects in all_predictions]
    ref_terms = [[a.get("term", "") for a in aspects] for aspects in all_references]
    
    ate_metrics = compute_ate_metrics(pred_terms, ref_terms)
    ate_metrics_partial = compute_ate_metrics(pred_terms, ref_terms, partial_match=True)
    pair_metrics = compute_aspect_polarity_metrics(all_predictions, all_references)
    
    results = {
        "aspect_term_extraction": {
            "exact_match": ate_metrics,
            "partial_match": ate_metrics_partial,
        },
        "aspect_polarity_pairs": pair_metrics,
        "num_examples": len(test_dataset),
    }
    
    # Print report
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nAspect Term Extraction (Exact Match):")
    print(f"  Precision: {ate_metrics['precision']:.4f}")
    print(f"  Recall:    {ate_metrics['recall']:.4f}")
    print(f"  F1:        {ate_metrics['f1']:.4f}")
    print(f"\nAspect Term Extraction (Partial Match):")
    print(f"  Precision: {ate_metrics_partial['precision']:.4f}")
    print(f"  Recall:    {ate_metrics_partial['recall']:.4f}")
    print(f"  F1:        {ate_metrics_partial['f1']:.4f}")
    print(f"\nAspect-Polarity Pairs:")
    print(f"  Precision: {pair_metrics['pair_precision']:.4f}")
    print(f"  Recall:    {pair_metrics['pair_recall']:.4f}")
    print(f"  F1:        {pair_metrics['pair_f1']:.4f}")
    print(f"\nSentiment Classification (on matched aspects):")
    print(f"  Accuracy:  {pair_metrics['sentiment_accuracy']:.4f}")
    print(f"  Macro F1:  {pair_metrics['sentiment_macro_f1']:.4f}")
    print("=" * 60)
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Evaluation CLI."""
    import argparse
    from datasets import load_from_disk
    from .inference import load_model
    
    parser = argparse.ArgumentParser(description="Evaluate ABSA model")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--uzbek", action="store_true", default=True, help="Use Uzbek prompts")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Load test data
    test_dataset = load_from_disk(args.test_data)
    if hasattr(test_dataset, "keys"):
        test_dataset = test_dataset.get("test", test_dataset.get("validation"))
    
    # Run evaluation
    results = evaluate_model(model, tokenizer, test_dataset, use_uzbek=args.uzbek)
    
    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
