#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Evaluation Script
# =============================================================================
"""
Script to evaluate fine-tuned models on the Uzbek ABSA test set.

Usage:
    python scripts/evaluate.py --model ./outputs/my_run/merged_model --test-data ./data/processed

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

from datasets import load_from_disk
from src.inference import load_model
from src.evaluation import evaluate_model

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate ABSA model on test data"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model or LoRA adapters"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path (if using separate LoRA adapters)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--use-english",
        action="store_true",
        help="Use English prompts instead of Uzbek"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("UzABSA-LLM Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test data: {args.test_data}")
    
    # Load model
    logger.info("\nLoading model...")
    model, tokenizer = load_model(
        model_path=args.model_path if not args.base_model else args.base_model,
        adapter_path=args.model_path if args.base_model else None,
    )
    
    # Load test data
    logger.info("\nLoading test data...")
    test_dataset = load_from_disk(args.test_data)
    
    # Handle DatasetDict
    if hasattr(test_dataset, "keys"):
        if "test" in test_dataset:
            test_dataset = test_dataset["test"]
        elif "validation" in test_dataset:
            test_dataset = test_dataset["validation"]
        else:
            # Use first available split
            first_split = list(test_dataset.keys())[0]
            test_dataset = test_dataset[first_split]
            logger.warning(f"No test/validation split found, using '{first_split}'")
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(args.max_samples))
        logger.info(f"Limited to {args.max_samples} samples")
    
    logger.info(f"Test examples: {len(test_dataset)}")
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        use_uzbek=not args.use_english,
    )
    
    # Add metadata
    results["metadata"] = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(test_dataset),
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_results_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
