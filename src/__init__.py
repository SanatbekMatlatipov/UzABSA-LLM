# =============================================================================
# UzABSA-LLM Source Package
# =============================================================================
"""
Uzbek Aspect-Based Sentiment Analysis with Large Language Models.

This package contains modules for:
- Data preparation and formatting for LLM fine-tuning
- Dataset loading and utilities (raw + annotated)
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

from .dataset_utils import (
    load_raw_reviews_csv,
    load_annotated_absa_dataset,
    clean_raw_reviews,
    analyze_dataset_stats,
    merge_raw_and_annotated,
)

from .format_converter import (
    convert_semeval_to_instruction_format,
    convert_dataset,
    analyze_converted_dataset,
)

from .gpu_config import (
    get_gpu_info,
    print_gpu_status,
    recommend_training_config,
    get_batch_size_recommendations,
    estimate_model_memory,
)

from .training_metrics import (
    TrainingMetricsCallback,
    save_experiment_summary,
    plot_model_comparison,
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
    # Dataset utilities
    "load_raw_reviews_csv",
    "load_annotated_absa_dataset",
    "clean_raw_reviews",
    "analyze_dataset_stats",
    "merge_raw_and_annotated",
    # Format conversion
    "convert_semeval_to_instruction_format",
    "convert_dataset",
    "analyze_converted_dataset",
    # GPU config
    "get_gpu_info",
    "print_gpu_status",
    "recommend_training_config",
    "get_batch_size_recommendations",
    "estimate_model_memory",
    # Training metrics
    "TrainingMetricsCallback",
    "save_experiment_summary",
    "plot_model_comparison",
    # Inference (lazy loaded)
    "load_model",
    "extract_aspects",
    # Evaluation (lazy loaded)
    "evaluate_model",
]
