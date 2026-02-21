#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Complete Data Preparation Pipeline
# =============================================================================
"""
Complete end-to-end data preparation pipeline.

This script:
1. Loads raw reviews from sharh.commeta.uz
2. Loads the annotated ABSA dataset from Hugging Face
3. Converts to instruction-tuning format
4. Creates train/validation splits
5. Saves for training

Usage:
    python scripts/prepare_complete_dataset.py --output-dir ./data/processed

Author: UzABSA Team
License: MIT
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import DatasetDict
from src.data_prep import format_for_instruction_tuning, create_train_val_split
from src.dataset_utils import load_annotated_absa_dataset, load_raw_reviews_csv
from src.format_converter import analyze_converted_dataset, convert_dataset

# =============================================================================
# Configure Logging
# =============================================================================
# Create log file handler with explicit formatting and flushing
log_filename = f"data_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    force=True,
    handlers=[stream_handler, file_handler],
)
logger = logging.getLogger(__name__)


def main():
    """Main data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete data preparation pipeline for UzABSA-LLM"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default="./data/raw/reviews.csv",
        help="Path to raw reviews CSV (optional)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples to process (for testing)",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw reviews in the processed dataset",
    )
    parser.add_argument(
        "--use-english",
        action="store_true",
        help="Use English prompts instead of Uzbek",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("UzABSA-LLM Complete Data Preparation Pipeline")
    logger.info("=" * 70)
    
    # =========================================================================
    # Step 1: Load and Process Annotated Dataset
    # =========================================================================
    
    logger.info("\n[Step 1/4] Loading Annotated ABSA Dataset")
    logger.info("-" * 70)
    
    try:
        full_dataset = load_annotated_absa_dataset(split=None)
        logger.info(f"Loaded annotated dataset with splits: {list(full_dataset.keys())}")
        
        # Use train split or combine all
        if "train" in full_dataset:
            dataset = full_dataset["train"]
        else:
            # Combine all splits
            dataset = full_dataset[list(full_dataset.keys())[0]]
        
        # Limit examples if requested
        if args.max_examples:
            dataset = dataset.select(range(min(args.max_examples, len(dataset))))
            logger.info(f"Limited to {len(dataset)} examples")
        else:
            logger.info(f"Using full dataset: {len(dataset)} examples")
    
    except Exception as e:
        logger.error(f"Failed to load annotated dataset: {e}")
        sys.exit(1)
    
    # =========================================================================
    # Step 2: Convert to Instruction Format
    # =========================================================================
    
    logger.info("\n[Step 2/4] Converting to Instruction Format")
    logger.info("-" * 70)
    
    try:
        converted_list = convert_dataset(dataset, format_type="semeval")
        logger.info(f"Converted {len(converted_list)} examples")
        
        # Analyze the converted dataset
        stats = analyze_converted_dataset(converted_list)
        
        # Save conversion statistics
        stats_file = output_dir / "conversion_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved conversion statistics to {stats_file}")
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)
    
    # =========================================================================
    # Step 3: Format for LLM Training
    # =========================================================================
    
    logger.info("\n[Step 3/4] Formatting for LLM Training")
    logger.info("-" * 70)
    
    try:
        # Create a dataset from converted examples
        # Map the converted format to the data_prep format
        processed_examples = []
        for conv_example in converted_list:
            # Create instruction tuning format
            processed_examples.append({
                "text": conv_example.get("text", ""),
                "aspects": conv_example.get("aspects", []),
            })
        
        # Convert to HuggingFace dataset
        from datasets import Dataset
        
        formatted_dataset = Dataset.from_dict({
            "text": [e["text"] for e in processed_examples],
            "aspects": [json.dumps(e["aspects"], ensure_ascii=False) for e in processed_examples],
        })
        
        # Apply instruction formatting
        formatted_dataset = format_for_instruction_tuning(
            formatted_dataset,
            text_column="text",
            aspects_column="aspects",
            system_prompt=None,  # Will use default
            use_chatml=True,
        )
        
        logger.info(f"Formatted {len(formatted_dataset)} examples")
        
        # Show sample
        if len(formatted_dataset) > 0:
            sample = formatted_dataset[0]
            logger.info(f"\nSample formatted example:")
            logger.info(f"Length: {len(sample['text'])} characters")
            logger.info(f"First 200 chars: {sample['text'][:200]}...")
    
    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        sys.exit(1)
    
    # =========================================================================
    # Step 4: Create Train/Validation Split and Save
    # =========================================================================
    
    logger.info("\n[Step 4/4] Creating Splits and Saving")
    logger.info("-" * 70)
    
    try:
        # Create train/validation split
        split_dataset = create_train_val_split(
            formatted_dataset,
            val_size=args.val_size,
            seed=42,
        )
        
        logger.info(f"Train examples: {len(split_dataset['train'])}")
        logger.info(f"Validation examples: {len(split_dataset['validation'])}")
        
        # Save the datasets
        logger.info(f"\nSaving processed dataset to {output_dir}...")
        split_dataset.save_to_disk(str(output_dir))
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(formatted_dataset),
            "train_examples": len(split_dataset["train"]),
            "val_examples": len(split_dataset["validation"]),
            "val_size": args.val_size,
            "conversion_stats": stats,
            "source": "Sanatbek/aspect-based-sentiment-analysis-uzbek",
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    except Exception as e:
        logger.error(f"Save failed: {e}")
        sys.exit(1)
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Data Preparation Complete!")
    logger.info("=" * 70)
    
    logger.info(f"\nOutput Directory: {output_dir}")
    logger.info(f"Files Created:")
    logger.info(f"  - train/: Training dataset splits")
    logger.info(f"  - validation/: Validation dataset splits")
    logger.info(f"  - metadata.json: Dataset metadata and statistics")
    logger.info(f"  - conversion_stats.json: Format conversion statistics")
    
    logger.info(f"\nNext Steps:")
    logger.info(f"1. Review the prepared dataset:")
    logger.info(f"   python -c \"from datasets import load_from_disk; ds = load_from_disk('{output_dir}'); print(ds)\"")
    logger.info(f"\n2. Start training:")
    logger.info(f"   python scripts/train_unsloth.py --dataset {output_dir}")
    
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
