#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Dataset Utilities
# =============================================================================
"""
Utilities for loading and working with both raw and annotated datasets.

Supports:
1. Raw reviews from sharh.commeta.uz (CSV format)
2. Annotated ABSA dataset from Hugging Face
3. Combining and merging datasets
4. Data statistics and analysis

Author: UzABSA Team
License: MIT
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Raw Dataset Loading
# =============================================================================

def load_raw_reviews_csv(
    csv_path: str,
    text_column: str = "review_text",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load raw reviews from sharh.commeta.uz CSV file.

    Args:
        csv_path: Path to the CSV file.
        text_column: Name of the column containing review text.
        encoding: File encoding (default: utf-8).

    Returns:
        Pandas DataFrame with loaded reviews.

    Example:
        >>> df = load_raw_reviews_csv("./data/raw/reviews.csv")
        >>> print(f"Loaded {len(df)} reviews")
        >>> print(df.columns)
    """
    logger.info(f"Loading raw reviews from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        logger.info(f"Loaded {len(df)} reviews")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Sample shape: {df.shape}")
        
        # Check if text column exists
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise


def clean_raw_reviews(
    df: pd.DataFrame,
    text_column: str = "review_text",
    min_length: int = 10,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Clean raw reviews by removing null values, duplicates, and short texts.

    Args:
        df: Input DataFrame.
        text_column: Name of the text column.
        min_length: Minimum text length (characters).
        remove_duplicates: Whether to remove duplicate texts.

    Returns:
        Cleaned DataFrame.

    Example:
        >>> df = load_raw_reviews_csv("reviews.csv")
        >>> df_clean = clean_raw_reviews(df)
        >>> print(f"After cleaning: {len(df_clean)} reviews")
    """
    logger.info("Cleaning raw reviews...")
    
    initial_count = len(df)
    
    # Remove null texts
    df = df[df[text_column].notna()].copy()
    logger.info(f"  After removing nulls: {len(df)} reviews")
    
    # Remove empty strings
    df = df[df[text_column].str.strip() != ""]
    logger.info(f"  After removing empty texts: {len(df)} reviews")
    
    # Remove short texts
    df = df[df[text_column].str.len() >= min_length]
    logger.info(f"  After removing short texts (<{min_length} chars): {len(df)} reviews")
    
    # Remove duplicates
    if remove_duplicates:
        df = df.drop_duplicates(subset=[text_column])
        logger.info(f"  After removing duplicates: {len(df)} reviews")
    
    logger.info(f"Total cleaned: {initial_count} → {len(df)} reviews")
    
    return df


# =============================================================================
# Annotated Dataset Loading
# =============================================================================

def load_annotated_absa_dataset(
    dataset_id: str = "Sanatbek/aspect-based-sentiment-analysis-uzbek",
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Union[Dataset, DatasetDict]:
    """
    Load the annotated Uzbek ABSA dataset from Hugging Face.

    Args:
        dataset_id: Hugging Face dataset identifier.
        split: Specific split ('train', 'test', 'validation'). If None, loads all.
        cache_dir: Cache directory for downloaded datasets.

    Returns:
        HuggingFace Dataset or DatasetDict.

    Example:
        >>> dataset = load_annotated_absa_dataset()
        >>> print(dataset)
        >>> print(dataset["train"][0])
    """
    logger.info(f"Loading annotated ABSA dataset: {dataset_id}")
    
    try:
        dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir=cache_dir,
        )
        
        if isinstance(dataset, DatasetDict):
            total = sum(len(v) for v in dataset.values())
            logger.info(f"Loaded {total} examples across splits:")
            for split_name, split_data in dataset.items():
                logger.info(f"  {split_name}: {len(split_data)} examples")
        else:
            logger.info(f"Loaded {len(dataset)} examples")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load annotated dataset: {e}")
        raise


def inspect_annotated_dataset(dataset: Union[Dataset, DatasetDict]) -> Dict:
    """
    Inspect the structure and content of an annotated dataset.

    Args:
        dataset: HuggingFace Dataset or DatasetDict.

    Returns:
        Dictionary with inspection results.

    Example:
        >>> dataset = load_annotated_absa_dataset()
        >>> info = inspect_annotated_dataset(dataset)
        >>> print(info)
    """
    if isinstance(dataset, DatasetDict):
        sample_split = list(dataset.keys())[0]
        sample_data = dataset[sample_split]
    else:
        sample_data = dataset
    
    info = {
        "columns": sample_data.column_names,
        "num_rows": len(sample_data),
        "features": sample_data.features,
        "sample_example": sample_data[0] if len(sample_data) > 0 else None,
    }
    
    logger.info(f"Dataset columns: {info['columns']}")
    logger.info(f"Number of rows: {info['num_rows']}")
    if info['sample_example']:
        logger.info(f"Sample example keys: {list(info['sample_example'].keys())}")
        for key, value in info['sample_example'].items():
            val_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            logger.info(f"  {key}: {val_str}")
    
    return info


# =============================================================================
# Dataset Conversion and Format
# =============================================================================

def raw_reviews_to_hf_dataset(
    df: pd.DataFrame,
    text_column: str = "review_text",
) -> Dataset:
    """
    Convert a raw reviews DataFrame to a HuggingFace Dataset.

    Args:
        df: Pandas DataFrame with raw reviews.
        text_column: Name of the text column.

    Returns:
        HuggingFace Dataset.

    Example:
        >>> df = load_raw_reviews_csv("reviews.csv")
        >>> dataset = raw_reviews_to_hf_dataset(df)
    """
    logger.info("Converting raw reviews to HuggingFace Dataset...")
    
    # Select only the text column (or add metadata as needed)
    hf_dataset = Dataset.from_pandas(
        df[[text_column]].rename(columns={text_column: "text"}),
        preserve_index=False,
    )
    
    logger.info(f"Converted to HuggingFace Dataset with {len(hf_dataset)} examples")
    return hf_dataset


# =============================================================================
# Dataset Analysis and Statistics
# =============================================================================

def analyze_dataset_stats(
    dataset: Union[Dataset, pd.DataFrame],
    text_field: str = "text",
) -> Dict:
    """
    Analyze and compute statistics for a dataset.

    Args:
        dataset: HuggingFace Dataset or Pandas DataFrame.
        text_field: Name of the text field.

    Returns:
        Dictionary with statistics.

    Example:
        >>> df = load_raw_reviews_csv("reviews.csv")
        >>> stats = analyze_dataset_stats(df, text_field="review_text")
        >>> print(stats)
    """
    logger.info("Analyzing dataset statistics...")
    
    # Convert to list of texts
    if isinstance(dataset, Dataset):
        texts = dataset[text_field]
    elif isinstance(dataset, pd.DataFrame):
        texts = dataset[text_field].tolist()
    else:
        texts = list(dataset)
    
    # Compute statistics
    text_lengths = [len(t.split()) for t in texts]
    char_lengths = [len(t) for t in texts]
    
    stats = {
        "num_texts": len(texts),
        "avg_words": round(sum(text_lengths) / len(text_lengths), 2),
        "avg_chars": round(sum(char_lengths) / len(char_lengths), 2),
        "min_words": min(text_lengths),
        "max_words": max(text_lengths),
        "min_chars": min(char_lengths),
        "max_chars": max(char_lengths),
    }
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total texts: {stats['num_texts']}")
    logger.info(f"  Avg words per text: {stats['avg_words']}")
    logger.info(f"  Avg chars per text: {stats['avg_chars']}")
    logger.info(f"  Word range: {stats['min_words']}-{stats['max_words']}")
    logger.info(f"  Char range: {stats['min_chars']}-{stats['max_chars']}")
    
    return stats


# =============================================================================
# Dataset Merging and Filtering
# =============================================================================

def merge_raw_and_annotated(
    raw_df: pd.DataFrame,
    annotated_dataset: Dataset,
    raw_text_column: str = "review_text",
    annotated_text_column: str = "text",
    match_type: str = "exact",
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Merge raw reviews with annotated examples by matching text.

    Args:
        raw_df: DataFrame with raw reviews.
        annotated_dataset: Annotated dataset from Hugging Face.
        raw_text_column: Column name in raw DataFrame.
        annotated_text_column: Column name in annotated dataset.
        match_type: Matching strategy - "exact" or "fuzzy".

    Returns:
        Tuple of (merged DataFrame, list of matched indices).

    Example:
        >>> raw_df = load_raw_reviews_csv("reviews.csv")
        >>> annotated = load_annotated_absa_dataset()
        >>> merged_df, indices = merge_raw_and_annotated(raw_df, annotated["train"])
    """
    from difflib import SequenceMatcher
    
    logger.info("Merging raw and annotated datasets...")
    
    raw_texts = raw_df[raw_text_column].str.strip().tolist()
    annotated_texts = annotated_dataset[annotated_text_column]
    
    matched_indices = []
    merged_data = []
    
    def similarity_ratio(a: str, b: str) -> float:
        """Compute text similarity ratio."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    for raw_idx, raw_text in enumerate(raw_texts):
        best_match_idx = None
        best_score = 0.0
        
        for ann_idx, ann_text in enumerate(annotated_texts):
            if match_type == "exact":
                if raw_text == ann_text.strip():
                    best_match_idx = ann_idx
                    best_score = 1.0
                    break
            else:  # fuzzy
                score = similarity_ratio(raw_text, ann_text)
                if score > best_score:
                    best_score = score
                    best_match_idx = ann_idx
        
        if best_match_idx is not None and best_score > (0.95 if match_type == "fuzzy" else 0.99):
            matched_indices.append(best_match_idx)
            merged_data.append({
                **raw_df.iloc[raw_idx].to_dict(),
                "annotated_index": best_match_idx,
                "match_score": best_score,
            })
    
    merged_df = pd.DataFrame(merged_data)
    logger.info(f"Matched {len(merged_df)} raw reviews with annotated data")
    
    return merged_df, matched_indices


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI for dataset utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset utilities for UzABSA")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Load raw reviews
    load_raw = subparsers.add_parser("load-raw", help="Load raw reviews from CSV")
    load_raw.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    load_raw.add_argument("--clean", action="store_true", help="Clean reviews after loading")
    
    # Load annotated dataset
    load_ann = subparsers.add_parser("load-annotated", help="Load annotated ABSA dataset")
    load_ann.add_argument("--inspect", action="store_true", help="Inspect dataset structure")
    
    # Analyze
    analyze = subparsers.add_parser("analyze", help="Analyze dataset statistics")
    analyze.add_argument("--csv", type=str, help="Path to CSV file")
    analyze.add_argument("--text-column", type=str, default="review_text", help="Text column name")
    
    args = parser.parse_args()
    
    if args.command == "load-raw":
        df = load_raw_reviews_csv(args.csv)
        if args.clean:
            df = clean_raw_reviews(df)
        print(f"\nLoaded {len(df)} reviews")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst review:\n{df.iloc[0]['review_text'][:200]}...")
        
    elif args.command == "load-annotated":
        dataset = load_annotated_absa_dataset()
        if args.inspect:
            info = inspect_annotated_dataset(dataset)
        else:
            print(dataset)
        
    elif args.command == "analyze":
        if args.csv:
            df = load_raw_reviews_csv(args.csv)
            stats = analyze_dataset_stats(df, text_field=args.text_column)
        else:
            logger.error("Please specify --csv for analysis")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
