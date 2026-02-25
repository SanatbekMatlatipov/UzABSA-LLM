#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Final Dataset Assembly
# =============================================================================
"""
Assemble the final multi-domain ABSA dataset by combining:
1. Model annotations (from annotate_reviews.py)
2. LLM-as-Judge quality scores (from llm_judge.py)
3. Quality filtering by judge thresholds

Output: HuggingFace-compatible dataset ready for upload.

Usage:
    python scripts/assemble_dataset.py \
        --annotations ./data/annotated/reviews_annotated.json \
        --judge-results ./data/judged/judge_results.json \
        --output-dir ./data/final_dataset

Author: UzABSA Team
License: MIT
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Quality thresholds from LOG 024
INCLUDE_THRESHOLD = 3.5    # Overall score >= 3.5 → include in dataset
FLAG_THRESHOLD = 2.5       # 2.5 <= score < 3.5 → flag for human review
# score < 2.5 → exclude


def load_annotations(path: str) -> List[Dict]:
    """Load the annotated reviews JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judge_results(path: str) -> Dict[str, Dict]:
    """Load judge results and index by review_id."""
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {r["review_id"]: r for r in results if r.get("judge_scores")}


def merge_annotations_with_scores(
    annotations: List[Dict],
    judge_index: Dict[str, Dict],
) -> List[Dict]:
    """Merge annotations with their judge scores (if available)."""
    merged = []
    for ann in annotations:
        rid = ann["review_id"]
        record = {
            "review_id": rid,
            "text": ann["text"],
            "business_name": ann.get("business_name", ""),
            "business_category": ann.get("business_category", "Boshqa"),
            "user_rating": ann.get("user_rating"),
            "aspects": ann.get("aspects", []),
            "num_aspects": ann.get("num_aspects", 0),
            "annotation_source": ann.get("annotation_source", "qwen2.5-7b-finetuned"),
        }

        if rid in judge_index:
            scores = judge_index[rid]["judge_scores"]
            record["judge_completeness"] = scores.get("completeness")
            record["judge_accuracy"] = scores.get("accuracy")
            record["judge_sentiment"] = scores.get("sentiment")
            record["judge_relevance"] = scores.get("relevance")
            record["judge_overall"] = scores.get("overall")
            record["judge_explanation"] = scores.get("explanation", "")
            record["human_verified"] = False
            record["quality_tier"] = _classify_tier(scores.get("overall", 0))
        else:
            # Not in judge sample — mark as unjudged
            record["judge_overall"] = None
            record["quality_tier"] = "unjudged"
            record["human_verified"] = False

        merged.append(record)

    return merged


def _classify_tier(overall_score: float) -> str:
    """Classify a review into a quality tier."""
    if overall_score >= INCLUDE_THRESHOLD:
        return "include"
    elif overall_score >= FLAG_THRESHOLD:
        return "flag_for_review"
    else:
        return "exclude"


def apply_quality_filter(
    merged: List[Dict],
    include_unjudged: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Split merged records by quality tier.

    Args:
        merged: List of merged annotation records.
        include_unjudged: If True, unjudged annotations are included in 'silver'.
    """
    included = []
    flagged = []
    excluded = []
    unjudged = []

    for rec in merged:
        tier = rec.get("quality_tier", "unjudged")
        if tier == "include":
            included.append(rec)
        elif tier == "flag_for_review":
            flagged.append(rec)
        elif tier == "exclude":
            excluded.append(rec)
        else:
            unjudged.append(rec)

    return {
        "included": included,
        "flagged": flagged,
        "excluded": excluded,
        "unjudged": unjudged,
    }


def compute_dataset_stats(
    merged: List[Dict],
    splits: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics."""
    total = len(merged)
    judged = [r for r in merged if r.get("judge_overall") is not None]

    # Domain distribution
    domain_counts = Counter(r["business_category"] for r in merged)
    included_domain_counts = Counter(r["business_category"] for r in splits["included"])

    # Aspect stats
    all_aspects = sum(r["num_aspects"] for r in merged)
    avg_aspects = all_aspects / total if total > 0 else 0

    # Polarity distribution
    polarity_counts = Counter()
    for r in merged:
        for asp in r.get("aspects", []):
            polarity_counts[asp.get("polarity", "unknown")] += 1

    # Category distribution
    category_counts = Counter()
    for r in merged:
        for asp in r.get("aspects", []):
            category_counts[asp.get("category", "unknown")] += 1

    # Judge score stats
    judge_avg = {}
    if judged:
        for dim in ["judge_completeness", "judge_accuracy", "judge_sentiment",
                     "judge_relevance", "judge_overall"]:
            vals = [r[dim] for r in judged if r.get(dim) is not None]
            judge_avg[dim] = round(sum(vals) / len(vals), 2) if vals else None

    stats = {
        "total_annotations": total,
        "total_judged": len(judged),
        "quality_tiers": {
            "included": len(splits["included"]),
            "flagged": len(splits["flagged"]),
            "excluded": len(splits["excluded"]),
            "unjudged": len(splits["unjudged"]),
        },
        "total_aspects": all_aspects,
        "avg_aspects_per_review": round(avg_aspects, 2),
        "domains": len(domain_counts),
        "domain_distribution": dict(domain_counts.most_common()),
        "included_domain_distribution": dict(included_domain_counts.most_common()),
        "polarity_distribution": dict(polarity_counts),
        "top_categories": dict(category_counts.most_common(20)),
        "judge_averages": judge_avg,
        "annotation_source": "qwen2.5-7b-finetuned",
        "assembly_date": datetime.now().isoformat(),
    }
    return stats


def save_final_dataset(
    merged: List[Dict],
    splits: Dict[str, List[Dict]],
    stats: Dict,
    output_dir: Path,
):
    """Save the final dataset in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full dataset (all annotations + scores)
    full_path = output_dir / "uzbek_multi_domain_absa_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info(f"Full dataset: {full_path} ({len(merged)} reviews)")

    # 2. Included subset (judge score >= 3.5 + unjudged)
    silver = splits["included"] + splits["unjudged"]
    silver_path = output_dir / "uzbek_multi_domain_absa_silver.json"
    with open(silver_path, "w", encoding="utf-8") as f:
        json.dump(silver, f, indent=2, ensure_ascii=False)
    logger.info(f"Silver dataset: {silver_path} ({len(silver)} reviews)")

    # 3. Judge-approved only (quality >= 3.5)
    approved_path = output_dir / "uzbek_multi_domain_absa_approved.json"
    with open(approved_path, "w", encoding="utf-8") as f:
        json.dump(splits["included"], f, indent=2, ensure_ascii=False)
    logger.info(f"Approved dataset: {approved_path} ({len(splits['included'])} reviews)")

    # 4. Flagged for review
    if splits["flagged"]:
        flagged_path = output_dir / "reviews_flagged_for_review.json"
        with open(flagged_path, "w", encoding="utf-8") as f:
            json.dump(splits["flagged"], f, indent=2, ensure_ascii=False)
        logger.info(f"Flagged reviews: {flagged_path} ({len(splits['flagged'])} reviews)")

    # 5. Dataset stats
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Dataset stats: {stats_path}")

    # 6. JSONL format (for HuggingFace datasets)
    jsonl_path = output_dir / "uzbek_multi_domain_absa.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in silver:
            # Clean record for HuggingFace
            hf_record = {
                "review_id": rec["review_id"],
                "text": rec["text"],
                "business_name": rec["business_name"],
                "business_category": rec["business_category"],
                "user_rating": rec["user_rating"],
                "aspects": rec["aspects"],
                "annotation_source": rec["annotation_source"],
                "judge_overall": rec.get("judge_overall"),
                "human_verified": rec.get("human_verified", False),
            }
            f.write(json.dumps(hf_record, ensure_ascii=False) + "\n")
    logger.info(f"JSONL dataset: {jsonl_path} ({len(silver)} reviews)")

    return {
        "full": str(full_path),
        "silver": str(silver_path),
        "approved": str(approved_path),
        "jsonl": str(jsonl_path),
        "stats": str(stats_path),
    }


def print_summary(stats: Dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("FINAL DATASET ASSEMBLY COMPLETE")
    print("=" * 60)
    print(f"\nTotal annotations: {stats['total_annotations']}")
    print(f"Total judged: {stats['total_judged']}")
    print(f"  - Included (≥3.5): {stats['quality_tiers']['included']}")
    print(f"  - Flagged (2.5-3.5): {stats['quality_tiers']['flagged']}")
    print(f"  - Excluded (<2.5): {stats['quality_tiers']['excluded']}")
    print(f"  - Unjudged: {stats['quality_tiers']['unjudged']}")
    print(f"\nTotal aspects: {stats['total_aspects']}")
    print(f"Avg aspects/review: {stats['avg_aspects_per_review']}")
    print(f"Domains: {stats['domains']}")

    if stats.get("judge_averages"):
        print("\nJudge score averages:")
        for dim, val in stats["judge_averages"].items():
            if val is not None:
                print(f"  {dim}: {val:.2f}")

    print(f"\nPolarity distribution: {stats['polarity_distribution']}")
    print(f"\nTop 10 domains:")
    for domain, count in list(stats["domain_distribution"].items())[:10]:
        print(f"  {domain}: {count}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Assemble final ABSA dataset")
    parser.add_argument(
        "--annotations", type=str,
        default="./data/annotated/reviews_annotated.json",
        help="Path to annotated reviews JSON",
    )
    parser.add_argument(
        "--judge-results", type=str, default=None,
        help="Path to judge results JSON (optional — can assemble without judge scores)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/final_dataset",
        help="Output directory",
    )
    parser.add_argument(
        "--include-threshold", type=float, default=3.5,
        help="Minimum judge score to include (default: 3.5)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    global INCLUDE_THRESHOLD
    INCLUDE_THRESHOLD = args.include_threshold

    logger.info("=" * 60)
    logger.info("UzABSA-LLM: Final Dataset Assembly")
    logger.info("=" * 60)

    # 1. Load annotations
    logger.info(f"Loading annotations: {args.annotations}")
    annotations = load_annotations(args.annotations)
    logger.info(f"Loaded {len(annotations)} annotations")

    # 2. Load judge results (optional)
    judge_index = {}
    if args.judge_results:
        logger.info(f"Loading judge results: {args.judge_results}")
        judge_index = load_judge_results(args.judge_results)
        logger.info(f"Loaded judge scores for {len(judge_index)} reviews")
    else:
        logger.info("No judge results provided — assembling without quality scores")

    # 3. Merge
    logger.info("Merging annotations with judge scores...")
    merged = merge_annotations_with_scores(annotations, judge_index)

    # 4. Quality filter
    logger.info("Applying quality filter...")
    splits = apply_quality_filter(merged)

    # 5. Stats
    stats = compute_dataset_stats(merged, splits)

    # 6. Save
    logger.info("Saving final dataset...")
    paths = save_final_dataset(merged, splits, stats, output_dir)

    # 7. Summary
    print_summary(stats)

    return paths


if __name__ == "__main__":
    main()
