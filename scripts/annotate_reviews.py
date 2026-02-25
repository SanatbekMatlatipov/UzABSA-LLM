#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Batch Annotation Script for reviews.csv
# =============================================================================
"""
Annotate raw reviews from reviews.csv using a fine-tuned ABSA model.
Produces a multi-domain annotated dataset ready for HuggingFace upload.

Usage:
    python scripts/annotate_reviews.py \
        --model-path ./outputs/my_run/uzabsa_qwen2.5-7b_20260222_001629/merged_model \
        --reviews-csv ./data/raw/reviews.csv \
        --categories-json ./data/raw/business_categories.json \
        --output-dir ./data/annotated

Author: UzABSA Team
License: MIT
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import load_model, extract_aspects

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_reviews(
    csv_path: str,
    categories_path: str,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load and prepare reviews.csv with business categories."""
    logger.info(f"Loading reviews from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Add business categories
    with open(categories_path, encoding="utf-8") as f:
        cats = json.load(f)
    cat_map = {c["object_name"]: c["business_category"] for c in cats}
    df["business_category"] = df["object_name"].map(cat_map)
    
    # Drop duplicates
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["review_text"], keep="first")
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate reviews")
    
    # Filter empty/very short reviews
    df = df[df["review_text"].str.len() >= 10].copy()
    
    # Add review_id
    df = df.reset_index(drop=True)
    df["review_id"] = [f"rev_{i:05d}" for i in range(len(df))]
    
    logger.info(f"Loaded {len(df)} reviews across {df['business_category'].nunique()} domains")
    return df


def annotate_batch(
    model,
    tokenizer,
    df: pd.DataFrame,
    use_uzbek: bool = True,
    checkpoint_dir: Path = None,
    checkpoint_every: int = 100,
) -> list:
    """Run inference on all reviews with checkpoint support for crash recovery."""
    annotations = []
    start_idx = 0
    
    # Resume from checkpoint if available
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "annotation_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            annotations = checkpoint["annotations"]
            start_idx = checkpoint["next_idx"]
            logger.info(f"Resuming from checkpoint: {start_idx}/{len(df)} already done")
    
    parse_successes = sum(1 for a in annotations if a.get("parse_success"))
    parse_failures = sum(1 for a in annotations if not a.get("parse_success"))
    
    rows = list(df.iterrows())
    for i in tqdm(range(start_idx, len(rows)), total=len(rows) - start_idx,
                  initial=0, desc="Annotating"):
        idx, row = rows[i]
        review_text = str(row["review_text"]).strip()
        
        # Run ABSA inference
        result = extract_aspects(model, tokenizer, review_text, use_uzbek=use_uzbek)
        
        aspects = result.get("aspects", [])
        parse_ok = result.get("parse_success", False)
        
        if parse_ok:
            parse_successes += 1
        else:
            parse_failures += 1
        
        annotation = {
            "review_id": row["review_id"],
            "text": review_text,
            "business_name": row["object_name"],
            "business_category": row.get("business_category", "Boshqa"),
            "user_rating": int(row["rating_value"]),
            "aspects": aspects,
            "num_aspects": len(aspects),
            "annotation_source": "qwen2.5-7b-finetuned",
            "parse_success": parse_ok,
            "raw_output": result.get("raw_output", ""),
        }
        annotations.append(annotation)
        
        # Save checkpoint
        if checkpoint_dir and (len(annotations) % checkpoint_every == 0):
            checkpoint_file = checkpoint_dir / "annotation_checkpoint.json"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"annotations": annotations, "next_idx": i + 1}, f, ensure_ascii=False)
            logger.info(f"Checkpoint saved: {len(annotations)}/{len(df)} reviews")
    
    total = len(annotations)
    rate = parse_successes / total * 100 if total > 0 else 0
    logger.info(f"Annotation complete: {total} reviews")
    logger.info(f"JSON parse rate: {rate:.1f}% ({parse_successes}/{total})")
    logger.info(f"Parse failures: {parse_failures}")
    
    # Remove checkpoint file on successful completion
    if checkpoint_dir:
        checkpoint_file = checkpoint_dir / "annotation_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Checkpoint file removed (annotation complete)")
    
    return annotations


def quality_filter(annotations: list) -> dict:
    """Filter annotations by quality and return stats."""
    total = len(annotations)
    
    # Categorize
    good = []       # parse_success=True AND >=1 aspect
    no_aspects = [] # parse_success=True but 0 aspects
    failed = []     # parse_success=False
    flagged = []    # >5 aspects (potential hallucination)
    
    for ann in annotations:
        if not ann["parse_success"]:
            failed.append(ann)
        elif ann["num_aspects"] == 0:
            no_aspects.append(ann)
        else:
            good.append(ann)
            if ann["num_aspects"] > 5:
                flagged.append(ann)
    
    stats = {
        "total_reviews": total,
        "good_annotations": len(good),
        "no_aspects_extracted": len(no_aspects),
        "parse_failures": len(failed),
        "flagged_many_aspects": len(flagged),
        "usable_rate": round(len(good) / total * 100, 2) if total > 0 else 0,
    }
    
    logger.info(f"Quality filter: {len(good)}/{total} usable ({stats['usable_rate']}%)")
    logger.info(f"  No aspects: {len(no_aspects)}, Parse failures: {len(failed)}, Flagged (>5 aspects): {len(flagged)}")
    
    return {
        "good": good,
        "no_aspects": no_aspects,
        "failed": failed,
        "flagged": flagged,
        "stats": stats,
    }


def save_dataset(annotations: list, output_dir: Path, stats: dict):
    """Save annotated dataset in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Full annotated dataset (JSON) — stripped of raw_output for size
    clean_annotations = []
    for ann in annotations:
        clean = {k: v for k, v in ann.items() if k != "raw_output"}
        clean_annotations.append(clean)
    
    dataset_path = output_dir / "reviews_annotated.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(clean_annotations, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved annotated dataset: {dataset_path} ({len(clean_annotations)} reviews)")
    
    # 2. Full dataset with raw outputs (for debugging)
    full_path = output_dir / f"reviews_annotated_full_{timestamp}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    # 3. Stats
    stats_path = output_dir / "annotation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved annotation stats: {stats_path}")
    
    # 4. Domain distribution of annotated data
    domain_stats = {}
    for ann in clean_annotations:
        cat = ann["business_category"]
        if cat not in domain_stats:
            domain_stats[cat] = {"count": 0, "avg_aspects": 0, "total_aspects": 0}
        domain_stats[cat]["count"] += 1
        domain_stats[cat]["total_aspects"] += ann["num_aspects"]
    
    for cat in domain_stats:
        c = domain_stats[cat]["count"]
        domain_stats[cat]["avg_aspects"] = round(domain_stats[cat]["total_aspects"] / c, 2) if c > 0 else 0
    
    domain_path = output_dir / "domain_distribution.json"
    with open(domain_path, "w", encoding="utf-8") as f:
        json.dump(domain_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved domain distribution: {domain_path}")
    
    return dataset_path


def main():
    parser = argparse.ArgumentParser(description="Annotate reviews.csv with ABSA model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to merged model")
    parser.add_argument("--reviews-csv", type=str, default="./data/raw/reviews.csv", help="Path to reviews CSV")
    parser.add_argument("--categories-json", type=str, default="./data/raw/business_categories.json", help="Business categories JSON")
    parser.add_argument("--output-dir", type=str, default="./data/annotated", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max reviews to annotate (for testing)")
    parser.add_argument("--use-english", action="store_true", help="Use English prompts")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("UzABSA-LLM: Multi-Domain Review Annotation")
    logger.info("=" * 60)
    
    # 1. Load reviews
    df = load_reviews(args.reviews_csv, args.categories_json)
    
    if args.max_samples and args.max_samples < len(df):
        df = df.head(args.max_samples)
        logger.info(f"Limited to {args.max_samples} samples")
    
    # 2. Load model
    logger.info("\nLoading model...")
    model, tokenizer = load_model(args.model_path)
    
    # 3. Annotate (with checkpoint support for crash recovery)
    logger.info("\nStarting annotation...")
    annotations = annotate_batch(
        model, tokenizer, df,
        use_uzbek=not args.use_english,
        checkpoint_dir=output_dir,
        checkpoint_every=100,
    )
    
    # 4. Quality filter
    logger.info("\nFiltering by quality...")
    filtered = quality_filter(annotations)
    
    # 5. Save — use the "good" annotations (parse_success + >=1 aspect)
    logger.info("\nSaving dataset...")
    all_annotations = filtered["good"] + filtered["no_aspects"]  # include no-aspect reviews too
    save_dataset(all_annotations, output_dir, filtered["stats"])
    
    # Print summary
    stats = filtered["stats"]
    print("\n" + "=" * 60)
    print("ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Total reviews processed: {stats['total_reviews']}")
    print(f"Good annotations (≥1 aspect): {stats['good_annotations']}")
    print(f"No aspects extracted: {stats['no_aspects_extracted']}")
    print(f"Parse failures: {stats['parse_failures']}")
    print(f"Flagged (>5 aspects): {stats['flagged_many_aspects']}")
    print(f"Usable rate: {stats['usable_rate']}%")
    print(f"\nOutput: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
