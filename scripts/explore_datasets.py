#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Dataset Explorer
# =============================================================================
"""
Interactive script to explore raw and annotated datasets.

Usage:
    python scripts/explore_datasets.py
    python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv

Author: UzABSA Team
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_utils import (
    analyze_dataset_stats,
    clean_raw_reviews,
    inspect_annotated_dataset,
    load_annotated_absa_dataset,
    load_raw_reviews_csv,
    merge_raw_and_annotated,
)

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main exploration function."""
    parser = argparse.ArgumentParser(description="Explore UzABSA datasets")
    
    parser.add_argument(
        "--raw-file",
        type=str,
        default="./data/raw/reviews.csv",
        help="Path to raw reviews CSV file"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean raw reviews after loading"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze dataset statistics"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Try to merge raw and annotated datasets"
    )
    parser.add_argument(
        "--annotated-split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Which split to load from annotated dataset"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("UzABSA Dataset Explorer")
    print("=" * 70)
    
    # Load raw reviews
    print("\n[1/3] Loading Raw Reviews from sharh.commeta.uz")
    print("-" * 70)
    
    raw_file_path = Path(args.raw_file)
    if not raw_file_path.exists():
        logger.error(f"Raw reviews file not found: {args.raw_file}")
        sys.exit(1)
    
    raw_df = load_raw_reviews_csv(str(raw_file_path))
    
    # Show sample
    print(f"\nSample raw review:")
    print(f"  Object: {raw_df.iloc[0]['object_name']}")
    print(f"  Rating: {raw_df.iloc[0]['rating_value']}/5")
    print(f"  Text: {raw_df.iloc[0]['review_text'][:150]}...")
    
    # Clean if requested
    if args.clean:
        print("\nCleaning raw reviews...")
        raw_df = clean_raw_reviews(raw_df)
    
    # Analyze if requested
    if args.analyze:
        print("\nAnalyzing raw reviews statistics...")
        stats = analyze_dataset_stats(
            raw_df,
            text_field="review_text"
        )
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Load annotated dataset
    print("\n[2/3] Loading Annotated ABSA Dataset")
    print("-" * 70)
    
    try:
        annotated_dataset = load_annotated_absa_dataset(split=args.annotated_split)
        
        # Inspect
        print(f"\nInspecting annotated dataset ({args.annotated_split} split)...")
        info = inspect_annotated_dataset(annotated_dataset)
        
        print(f"\nSample annotated example:")
        if "text" in info['sample_example']:
            print(f"  Text: {info['sample_example']['text'][:100]}...")
        if "aspects" in info['sample_example']:
            print(f"  Aspects: {info['sample_example']['aspects']}")
        if "sentiments" in info['sample_example']:
            print(f"  Sentiments: {info['sample_example']['sentiments']}")
        
    except Exception as e:
        logger.error(f"Could not load annotated dataset: {e}")
        annotated_dataset = None
    
    # Merge datasets
    if args.merge and annotated_dataset:
        print("\n[3/3] Attempting to Merge Datasets")
        print("-" * 70)
        
        try:
            merged_df, matched_indices = merge_raw_and_annotated(
                raw_df,
                annotated_dataset,
                raw_text_column="review_text",
                annotated_text_column="text",
                match_type="fuzzy"
            )
            
            print(f"\nMerge Results:")
            print(f"  Raw reviews: {len(raw_df)}")
            print(f"  Annotated examples: {len(annotated_dataset)}")
            print(f"  Matched: {len(merged_df)}")
            print(f"  Match rate: {len(merged_df) / len(raw_df) * 100:.1f}%")
            
            if len(merged_df) > 0:
                print(f"\nSample merged entry:")
                sample = merged_df.iloc[0]
                print(f"  Object: {sample['object_name']}")
                print(f"  Raw text: {sample['review_text'][:100]}...")
                print(f"  Match score: {sample['match_score']:.2%}")
        
        except Exception as e:
            logger.error(f"Merge failed: {e}")
    
    print("\n" + "=" * 70)
    print("Dataset Exploration Complete!")
    print("=" * 70)
    
    # Summary
    print(f"\nSummary:")
    print(f"  Raw reviews loaded: {len(raw_df)}")
    if annotated_dataset:
        print(f"  Annotated examples: {len(annotated_dataset)}")
    print(f"\nNext steps:")
    print(f"  1. Run: python -m src.data_prep --output-dir ./data/processed")
    print(f"  2. Run: python scripts/train_unsloth.py --help")


if __name__ == "__main__":
    main()
