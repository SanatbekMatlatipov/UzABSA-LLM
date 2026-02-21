#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Data Preparation Module
# =============================================================================
"""
Data preparation module for Uzbek Aspect-Based Sentiment Analysis.

This module provides functions to:
1. Load the Uzbek ABSA dataset from Hugging Face
2. Format data into instruction-response pairs for LLM fine-tuning
3. Create train/validation splits
4. Handle data augmentation and preprocessing

Dataset: Sanatbek/aspect-based-sentiment-analysis-uzbek
Contains: 6,000 annotated Uzbek reviews with aspect terms, categories, and polarities

Author: UzABSA Team
License: MIT
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
# Constants and Configuration
# =============================================================================

# Hugging Face dataset identifier
DATASET_ID = "Sanatbek/aspect-based-sentiment-analysis-uzbek"

# Supported polarity labels (adjust based on actual dataset schema)
POLARITY_LABELS = ["positive", "negative", "neutral"]

# Default system prompt for instruction tuning
DEFAULT_SYSTEM_PROMPT = """Siz o'zbek tilida matnlardan aspektlarni va ularning hissiyotlarini aniqlash bo'yicha mutaxassissiz.

Berilgan matndan barcha aspekt terminlarini, ularning kategoriyalarini va hissiyot polaritesini (positive, negative, neutral) ajratib oling.

Javobni quyidagi Python dictionary formatida qaytaring:
{
    "aspects": [
        {
            "term": "aspekt termin",
            "category": "kategoriya",
            "polarity": "positive/negative/neutral"
        }
    ]
}"""

# English version of the system prompt (for multilingual models)
DEFAULT_SYSTEM_PROMPT_EN = """You are an expert in extracting aspects and their sentiments from Uzbek text.

From the given text, extract all aspect terms, their categories, and sentiment polarity (positive, negative, neutral).

Return the response in the following Python dictionary format:
{
    "aspects": [
        {
            "term": "aspect term",
            "category": "category",
            "polarity": "positive/negative/neutral"
        }
    ]
}"""


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_uzbek_absa_dataset(
    dataset_id: str = DATASET_ID,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Union[Dataset, DatasetDict]:
    """
    Load the Uzbek ABSA dataset from Hugging Face Hub.

    This function handles downloading and caching the dataset for efficient
    repeated access during development and training.

    Args:
        dataset_id: Hugging Face dataset identifier.
                   Default: "Sanatbek/aspect-based-sentiment-analysis-uzbek"
        split: Specific split to load ('train', 'validation', 'test').
               If None, returns all splits as a DatasetDict.
        cache_dir: Directory for caching downloaded datasets.
                  If None, uses Hugging Face default cache.
        token: Hugging Face API token for private datasets.

    Returns:
        Dataset or DatasetDict containing the loaded data.

    Raises:
        ValueError: If the dataset cannot be loaded.
        ConnectionError: If there's a network issue.

    Example:
        >>> dataset = load_uzbek_absa_dataset()
        >>> print(dataset)
        DatasetDict({
            train: Dataset({...})
            test: Dataset({...})
        })
    """
    logger.info(f"Loading dataset: {dataset_id}")
    
    try:
        dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir=cache_dir,
            token=token,
        )
        
        # Log dataset statistics
        if isinstance(dataset, DatasetDict):
            for split_name, split_data in dataset.items():
                logger.info(f"  {split_name}: {len(split_data)} examples")
        else:
            logger.info(f"  Loaded {len(dataset)} examples")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise ValueError(f"Could not load dataset '{dataset_id}': {e}")


# =============================================================================
# Formatting Functions for Instruction Tuning
# =============================================================================

def format_aspect_output(aspects: List[Dict[str, str]]) -> str:
    """
    Format aspect annotations into a structured string representation.

    Converts a list of aspect dictionaries into a clean, parseable string
    that can be used as the expected output for LLM training.

    Args:
        aspects: List of aspect dictionaries, each containing:
                - term: The aspect term from the text
                - category: The aspect category
                - polarity: Sentiment polarity (positive/negative/neutral)

    Returns:
        Formatted string representation of the aspects dictionary.

    Example:
        >>> aspects = [{"term": "narx", "category": "price", "polarity": "negative"}]
        >>> print(format_aspect_output(aspects))
        {
            "aspects": [
                {
                    "term": "narx",
                    "category": "price",
                    "polarity": "negative"
                }
            ]
        }
    """
    output_dict = {"aspects": aspects}
    # Use indent=4 for readable output, ensure_ascii=False for Uzbek characters
    return json.dumps(output_dict, indent=4, ensure_ascii=False)


def create_instruction_prompt(
    text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    use_uzbek: bool = True,
) -> str:
    """
    Create an instruction prompt for aspect extraction.

    Generates a formatted prompt that instructs the model to extract
    aspects from the given Uzbek text.

    Args:
        text: The input review/text to analyze.
        system_prompt: System-level instructions for the model.
        use_uzbek: If True, use Uzbek instruction; otherwise English.

    Returns:
        Formatted instruction prompt string.

    Example:
        >>> prompt = create_instruction_prompt("Bu telefon juda yaxshi!")
        >>> print(prompt[:50])
        Quyidagi o'zbek tilidagi matndan aspektlarni...
    """
    if use_uzbek:
        user_instruction = (
            f"Quyidagi o'zbek tilidagi matndan aspektlarni, "
            f"kategoriyalarni va hissiyot polaritesini aniqlang:\n\n"
            f"Matn: \"{text}\""
        )
    else:
        user_instruction = (
            f"Extract aspects, categories, and sentiment polarities "
            f"from the following Uzbek text:\n\n"
            f"Text: \"{text}\""
        )
    
    return user_instruction


def format_single_example(
    example: Dict[str, Any],
    text_column: str = "text",
    aspects_column: str = "aspects",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    use_chatml: bool = True,
) -> Dict[str, str]:
    """
    Format a single dataset example into instruction-response format.

    This function transforms a raw dataset entry into the format required
    for supervised fine-tuning (SFT) of instruction-following LLMs.

    Args:
        example: A single example from the dataset containing text and aspects.
        text_column: Name of the column containing the input text.
        aspects_column: Name of the column containing aspect annotations.
        system_prompt: System instructions for the model.
        use_chatml: If True, format using ChatML template.

    Returns:
        Dictionary with 'instruction', 'input', and 'output' keys,
        or 'text' key if using ChatML format.

    Example:
        >>> example = {
        ...     "text": "Bu restoran juda yaxshi",
        ...     "aspects": [{"term": "restoran", "category": "general", "polarity": "positive"}]
        ... }
        >>> formatted = format_single_example(example)
        >>> print(formatted.keys())
        dict_keys(['text'])
    """
    # Extract text and aspects from the example
    text = example.get(text_column, "")
    aspects = example.get(aspects_column, [])
    
    # Handle different aspect formats (string JSON vs list)
    if isinstance(aspects, str):
        try:
            aspects = json.loads(aspects)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse aspects: {aspects}")
            aspects = []
    
    # Create the instruction prompt
    instruction = create_instruction_prompt(text)
    
    # Format the expected output
    output = format_aspect_output(aspects)
    
    if use_chatml:
        # Format as ChatML for models like Qwen, Llama-3, etc.
        formatted_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        return {"text": formatted_text}
    else:
        # Return as separate fields for flexible formatting
        return {
            "instruction": instruction,
            "input": text,
            "output": output,
            "system": system_prompt,
        }


def format_for_instruction_tuning(
    dataset: Union[Dataset, DatasetDict],
    text_column: str = "text",
    aspects_column: str = "aspects",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    use_chatml: bool = True,
    num_proc: int = 4,
    remove_original_columns: bool = True,
) -> Union[Dataset, DatasetDict]:
    """
    Format the entire dataset for instruction tuning.

    Applies the formatting function to all examples in the dataset,
    converting raw annotations into instruction-response pairs.

    Args:
        dataset: The dataset to format (Dataset or DatasetDict).
        text_column: Name of the column containing input text.
        aspects_column: Name of the column containing aspect annotations.
        system_prompt: System-level instructions for the model.
        use_chatml: If True, use ChatML conversation format.
        num_proc: Number of processes for parallel processing.
        remove_original_columns: If True, remove original columns after formatting.

    Returns:
        Formatted dataset ready for SFT training.

    Example:
        >>> dataset = load_uzbek_absa_dataset()
        >>> formatted = format_for_instruction_tuning(dataset)
        >>> print(formatted["train"][0].keys())
        dict_keys(['text'])
    """
    logger.info("Formatting dataset for instruction tuning...")
    
    def format_fn(example: Dict[str, Any]) -> Dict[str, str]:
        """Wrapper function for dataset.map()"""
        return format_single_example(
            example=example,
            text_column=text_column,
            aspects_column=aspects_column,
            system_prompt=system_prompt,
            use_chatml=use_chatml,
        )
    
    # Get columns to remove (if requested)
    if isinstance(dataset, DatasetDict):
        columns_to_remove = list(dataset[list(dataset.keys())[0]].column_names)
    else:
        columns_to_remove = list(dataset.column_names)
    
    # Apply formatting
    formatted_dataset = dataset.map(
        format_fn,
        num_proc=num_proc,
        remove_columns=columns_to_remove if remove_original_columns else None,
        desc="Formatting examples",
    )
    
    logger.info("Dataset formatting complete!")
    return formatted_dataset


# =============================================================================
# Train/Validation Split Functions
# =============================================================================

def create_train_val_split(
    dataset: Dataset,
    val_size: float = 0.1,
    seed: int = 42,
    stratify_column: Optional[str] = None,
) -> DatasetDict:
    """
    Create train and validation splits from a dataset.

    Useful when the original dataset doesn't have a validation split,
    or when you want to create a custom split ratio.

    Args:
        dataset: The dataset to split.
        val_size: Fraction of data to use for validation (0.0 to 1.0).
        seed: Random seed for reproducibility.
        stratify_column: Column to use for stratified splitting (optional).

    Returns:
        DatasetDict with 'train' and 'validation' splits.

    Example:
        >>> dataset = load_uzbek_absa_dataset(split="train")
        >>> splits = create_train_val_split(dataset, val_size=0.15)
        >>> print(splits)
        DatasetDict({
            train: Dataset({...})
            validation: Dataset({...})
        })
    """
    logger.info(f"Creating train/val split with val_size={val_size}")
    
    # Use train_test_split from datasets
    split_dataset = dataset.train_test_split(
        test_size=val_size,
        seed=seed,
        shuffle=True,
    )
    
    # Rename 'test' to 'validation' for clarity
    return DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"],
    })


# =============================================================================
# Utility Functions
# =============================================================================

def inspect_dataset_schema(dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
    """
    Inspect and return the schema of the dataset.

    Useful for understanding the structure of the loaded dataset
    and identifying the correct column names.

    Args:
        dataset: The dataset to inspect.

    Returns:
        Dictionary containing schema information and sample data.
    """
    if isinstance(dataset, DatasetDict):
        sample_split = list(dataset.keys())[0]
        sample_dataset = dataset[sample_split]
    else:
        sample_dataset = dataset
    
    schema_info = {
        "columns": sample_dataset.column_names,
        "features": str(sample_dataset.features),
        "num_rows": len(sample_dataset),
        "sample": sample_dataset[0] if len(sample_dataset) > 0 else None,
    }
    
    logger.info(f"Dataset columns: {schema_info['columns']}")
    logger.info(f"Number of rows: {schema_info['num_rows']}")
    
    return schema_info


def validate_formatted_example(example: Dict[str, str]) -> bool:
    """
    Validate that a formatted example has the correct structure.

    Args:
        example: A formatted example dictionary.

    Returns:
        True if valid, False otherwise.
    """
    if "text" in example:
        # ChatML format
        text = example["text"]
        required_markers = ["<|im_start|>", "<|im_end|>", "assistant"]
        return all(marker in text for marker in required_markers)
    else:
        # Instruction format
        required_keys = ["instruction", "output"]
        return all(key in example for key in required_keys)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """
    Main entry point for data preparation script.

    This function demonstrates the typical workflow for preparing
    the Uzbek ABSA dataset for LLM fine-tuning.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare Uzbek ABSA dataset for LLM fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Directory to save processed dataset"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use-english-prompt",
        action="store_true",
        help="Use English prompts instead of Uzbek"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info("=" * 60)
    logger.info("Starting Uzbek ABSA Data Preparation")
    logger.info("=" * 60)
    
    try:
        dataset = load_uzbek_absa_dataset()
        
        # Inspect schema
        schema = inspect_dataset_schema(dataset)
        logger.info(f"Dataset schema: {schema}")
        
        # Select system prompt
        system_prompt = (
            DEFAULT_SYSTEM_PROMPT_EN if args.use_english_prompt 
            else DEFAULT_SYSTEM_PROMPT
        )
        
        # Format for instruction tuning
        formatted_dataset = format_for_instruction_tuning(
            dataset=dataset,
            system_prompt=system_prompt,
        )
        
        # Create validation split if not present
        if "validation" not in formatted_dataset:
            logger.info("Creating validation split...")
            formatted_dataset = create_train_val_split(
                formatted_dataset["train"],
                val_size=args.val_size,
                seed=args.seed,
            )
        
        # Save processed dataset
        logger.info(f"Saving processed dataset to {args.output_dir}")
        formatted_dataset.save_to_disk(args.output_dir)
        
        logger.info("=" * 60)
        logger.info("Data preparation complete!")
        logger.info(f"Train examples: {len(formatted_dataset['train'])}")
        logger.info(f"Validation examples: {len(formatted_dataset['validation'])}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()
