#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: ABSA Format Converter
# =============================================================================
"""
Convert annotations from various formats to instruction-tuning format.

Supported formats:
1. Hugging Face SemEVAL 2014 format (Sanatbek dataset)
2. Custom JSON format
3. CSV with aspect annotations

Author: UzABSA Team
License: MIT
"""

import json
import logging
from typing import Any, Dict, List, Optional

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Format Conversion Functions
# =============================================================================

def convert_semeval_to_instruction_format(
    example: Dict[str, Any],
) -> Dict[str, str]:
    """
    Convert SemEVAL 2014 ABSA format to instruction-tuning format.

    Handles the Hugging Face format from Sanatbek/aspect-based-sentiment-analysis-uzbek:
    {
        "sentence_id": "...",
        "text": "...",
        "aspect_terms": [
            {"term": "...", "polarity": "positive/negative/neutral", ...}
        ],
        "aspect_categories": [
            {"category": "...", "polarity": "positive/negative/neutral"}
        ]
    }

    Args:
        example: Raw example from HuggingFace dataset.

    Returns:
        Dictionary with "aspects" key containing structured output.

    Example:
        >>> example = {
        ...     "text": "Juda yaxshi ovqat va kayfiyat",
        ...     "aspect_terms": [{"term": "ovqat", "polarity": "positive"}],
        ...     "aspect_categories": [{"category": "ovqat", "polarity": "positive"}]
        ... }
        >>> output = convert_semeval_to_instruction_format(example)
        >>> print(output)
    """
    text = example.get("text", "")
    
    aspects = []
    
    # Process aspect terms
    aspect_terms = example.get("aspect_terms", [])
    for term_obj in aspect_terms:
        aspect = {
            "term": term_obj.get("term", ""),
            "polarity": term_obj.get("polarity", "neutral").lower(),
        }
        aspects.append(aspect)
    
    # Process aspect categories (if no terms or to add categories)
    aspect_categories = example.get("aspect_categories", [])
    for cat_obj in aspect_categories:
        # Check if we already have this aspect from terms
        category = cat_obj.get("category", "")
        polarity = cat_obj.get("polarity", "neutral").lower()
        
        # Create a category-level aspect entry
        aspect = {
            "category": category,
            "polarity": polarity,
        }
        aspects.append(aspect)
    
    output = {
        "aspects": aspects,
        "text": text,
    }
    
    return output


def convert_to_structured_dict(
    aspect_terms: List[Dict[str, Any]],
    aspect_categories: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    """
    Convert aspect terms and categories to a structured dictionary.

    Combines aspect-term level and aspect-category level annotations
    into a unified format.

    Args:
        aspect_terms: List of aspect term objects.
        aspect_categories: List of aspect category objects.

    Returns:
        Structured dictionary with aspects list.
    """
    aspects = []
    
    # Add aspect terms
    for term_obj in aspect_terms:
        aspect = {
            "term": term_obj.get("term", ""),
            "polarity": term_obj.get("polarity", "neutral").lower(),
        }
        if "category" in term_obj:
            aspect["category"] = term_obj["category"]
        aspects.append(aspect)
    
    # Add aspect categories
    for cat_obj in aspect_categories:
        aspect = {
            "category": cat_obj.get("category", ""),
            "polarity": cat_obj.get("polarity", "neutral").lower(),
        }
        aspects.append(aspect)
    
    return {"aspects": aspects}


def validate_aspect_structure(aspect: Dict[str, str]) -> bool:
    """
    Validate that an aspect has the required fields.

    Args:
        aspect: Aspect dictionary to validate.

    Returns:
        True if valid, False otherwise.
    """
    required_fields = ["polarity"]
    optional_fields = ["term", "category"]
    
    # Check required fields
    if not all(field in aspect for field in required_fields):
        return False
    
    # Check that at least one of optional fields exists
    if not any(field in aspect for field in optional_fields):
        return False
    
    # Validate polarity
    valid_polarities = ["positive", "negative", "neutral"]
    if aspect.get("polarity", "").lower() not in valid_polarities:
        return False
    
    return True


# =============================================================================
# Batch Conversion
# =============================================================================

def convert_dataset(
    dataset,
    format_type: str = "semeval",
) -> list:
    """
    Convert an entire dataset to instruction-tuning format.

    Args:
        dataset: HuggingFace Dataset to convert.
        format_type: Format of the source dataset ("semeval", "custom", "csv").

    Returns:
        List of converted examples.

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("Sanatbek/aspect-based-sentiment-analysis-uzbek")
        >>> converted = convert_dataset(dataset["train"], format_type="semeval")
        >>> print(f"Converted {len(converted)} examples")
    """
    logger.info(f"Converting {len(dataset)} examples from {format_type} format...")
    
    converted_examples = []
    invalid_count = 0
    
    for idx, example in enumerate(dataset):
        try:
            if format_type == "semeval":
                converted = convert_semeval_to_instruction_format(example)
            elif format_type == "custom":
                converted = example  # Assume already correct format
            else:
                raise ValueError(f"Unknown format: {format_type}")
            
            # Validate aspects
            aspects = converted.get("aspects", [])
            valid_aspects = [a for a in aspects if validate_aspect_structure(a)]
            
            if valid_aspects:
                converted["aspects"] = valid_aspects
                converted_examples.append(converted)
            else:
                invalid_count += 1
        
        except Exception as e:
            logger.debug(f"Failed to convert example {idx}: {e}")
            invalid_count += 1
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(dataset)} examples")
    
    logger.info(f"Converted {len(converted_examples)} valid examples ({invalid_count} invalid)")
    
    return converted_examples


# =============================================================================
# Output Formatting
# =============================================================================

def format_aspect_output_json(aspects: List[Dict[str, str]]) -> str:
    """
    Format aspects as JSON for model output.

    Args:
        aspects: List of aspect dictionaries.

    Returns:
        JSON string representation.

    Example:
        >>> aspects = [{"term": "narx", "polarity": "negative"}]
        >>> output = format_aspect_output_json(aspects)
        >>> print(output)
    """
    output_dict = {"aspects": aspects}
    return json.dumps(output_dict, indent=4, ensure_ascii=False)


def format_aspect_output_python(aspects: List[Dict[str, str]]) -> str:
    """
    Format aspects as Python dict representation for model output.

    Args:
        aspects: List of aspect dictionaries.

    Returns:
        Python dict string representation.

    Example:
        >>> aspects = [{"term": "narx", "polarity": "negative"}]
        >>> output = format_aspect_output_python(aspects)
        >>> print(output)
    """
    output_dict = {"aspects": aspects}
    return repr(output_dict)


# =============================================================================
# Debugging and Analysis
# =============================================================================

def analyze_converted_dataset(converted_examples: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the converted dataset for quality and distribution.

    Args:
        converted_examples: List of converted examples.

    Returns:
        Dictionary with statistics.

    Example:
        >>> converted = convert_dataset(dataset)
        >>> stats = analyze_converted_dataset(converted)
        >>> print(stats)
    """
    logger.info("Analyzing converted dataset...")
    
    total_aspects = 0
    aspects_per_text = []
    polarity_counts = {"positive": 0, "negative": 0, "neutral": 0}
    text_lengths = []
    
    for example in converted_examples:
        aspects = example.get("aspects", [])
        text = example.get("text", "")
        
        total_aspects += len(aspects)
        aspects_per_text.append(len(aspects))
        text_lengths.append(len(text.split()))
        
        for aspect in aspects:
            polarity = aspect.get("polarity", "neutral").lower()
            if polarity in polarity_counts:
                polarity_counts[polarity] += 1
    
    avg_aspects = total_aspects / len(converted_examples) if converted_examples else 0
    avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    
    stats = {
        "total_examples": len(converted_examples),
        "total_aspects": total_aspects,
        "avg_aspects_per_text": round(avg_aspects, 2),
        "avg_text_length_words": round(avg_text_length, 2),
        "polarity_distribution": polarity_counts,
        "examples_with_aspects": sum(1 for n in aspects_per_text if n > 0),
    }
    
    logger.info(f"Statistics:")
    logger.info(f"  Total examples: {stats['total_examples']}")
    logger.info(f"  Total aspects: {stats['total_aspects']}")
    logger.info(f"  Avg aspects per text: {stats['avg_aspects_per_text']}")
    logger.info(f"  Avg text length: {stats['avg_text_length_words']} words")
    logger.info(f"  Polarity distribution: {stats['polarity_distribution']}")
    logger.info(f"  Examples with aspects: {stats['examples_with_aspects']}")
    
    return stats


if __name__ == "__main__":
    # Test conversion
    test_example = {
        "sentence_id": "2771#1",
        "text": "Juda yaxshi ovqat va kayfiyat",
        "aspect_terms": [
            {"term": "ovqat", "polarity": "positive"},
            {"term": "kayfiyat", "polarity": "positive"},
        ],
        "aspect_categories": [
            {"category": "food", "polarity": "positive"},
        ],
    }
    
    print("Testing conversion:")
    print(json.dumps(test_example, indent=2, ensure_ascii=False))
    print("\n--- Converted ---\n")
    
    converted = convert_semeval_to_instruction_format(test_example)
    print(json.dumps(converted, indent=2, ensure_ascii=False))
    
    output = format_aspect_output_json(converted["aspects"])
    print(f"\n--- Model Output ---\n{output}")
