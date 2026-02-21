# =============================================================================
# UzABSA-LLM Source Package
# =============================================================================
"""
Uzbek Aspect-Based Sentiment Analysis with Large Language Models.

This package contains modules for:
- Data preparation and formatting for LLM fine-tuning
- Model training utilities
- Evaluation metrics and benchmarking
- Inference pipelines
"""

__version__ = "0.1.0"
__author__ = "UzABSA Team"

from .data_prep import (
    load_uzbek_absa_dataset,
    format_for_instruction_tuning,
    create_train_val_split,
)

# Lazy imports for optional dependencies
def __getattr__(name):
    """Lazy import for inference and evaluation modules."""
    if name == "load_model":
        from .inference import load_model
        return load_model
    elif name == "extract_aspects":
        from .inference import extract_aspects
        return extract_aspects
    elif name == "evaluate_model":
        from .evaluation import evaluate_model
        return evaluate_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Data preparation
    "load_uzbek_absa_dataset",
    "format_for_instruction_tuning",
    "create_train_val_split",
    # Inference (lazy loaded)
    "load_model",
    "extract_aspects",
    # Evaluation (lazy loaded)
    "evaluate_model",
]
