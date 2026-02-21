#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Unsloth Training Script
# =============================================================================
"""
Fine-tuning script for Uzbek Aspect-Based Sentiment Analysis using Unsloth.

This script provides a complete pipeline for fine-tuning large language models
on the Uzbek ABSA dataset using QLoRA (Quantized Low-Rank Adaptation) with
the Unsloth library for optimized training.

Supported Models:
- DeepSeek (unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit)
- Qwen 2.5 (unsloth/Qwen2.5-7B-Instruct-bnb-4bit)
- Llama 3 (unsloth/llama-3-8b-Instruct-bnb-4bit)

Features:
- 4-bit quantization for memory efficiency
- LoRA adapters for parameter-efficient fine-tuning
- Gradient checkpointing for reduced VRAM usage
- WandB integration for experiment tracking

Author: UzABSA Team
License: MIT
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import load_from_disk

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gpu_config import (
    get_gpu_info,
    print_gpu_status,
    recommend_training_config,
)
from src.training_metrics import (
    TrainingMetricsCallback,
    save_experiment_summary,
)

# =============================================================================
# Configure Logging
# =============================================================================
# Create log file handler with explicit formatting
log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


# =============================================================================
# Model Configuration
# =============================================================================

# Supported 4-bit quantized models for Unsloth
SUPPORTED_MODELS = {
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen2.5-14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "qwen2.5-32b": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    "llama3-8b": "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "deepseek-7b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
    "deepseek-14b": "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "gemma2-9b": "unsloth/gemma-2-9b-it-bnb-4bit",
}

# Default LoRA target modules for transformer models
DEFAULT_LORA_TARGETS = [
    "q_proj",      # Query projection
    "k_proj",      # Key projection
    "v_proj",      # Value projection
    "o_proj",      # Output projection
    "gate_proj",   # Gate projection (for SwiGLU/GeGLU)
    "up_proj",     # Up projection
    "down_proj",   # Down projection
]


# =============================================================================
# Training Configuration Dataclass
# =============================================================================

class TrainingConfig:
    """
    Configuration class for training parameters.
    
    This class encapsulates all hyperparameters and settings needed
    for fine-tuning, making it easy to modify and experiment.
    """
    
    def __init__(
        self,
        # Model settings
        model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        
        # LoRA settings
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        use_gradient_checkpointing: str = "unsloth",
        
        # Training settings
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        max_steps: int = 1000,
        num_train_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
        
        # Optimization settings
        optim: str = "adamw_8bit",
        fp16: bool = False,
        bf16: bool = True,
        
        # Logging and saving
        logging_steps: int = 10,
        save_steps: int = 100,
        save_total_limit: int = 3,
        output_dir: str = "./outputs",
        
        # Misc
        seed: int = 42,
        report_to: str = "wandb",
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or DEFAULT_LORA_TARGETS
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        
        self.optim = optim
        self.fp16 = fp16
        self.bf16 = bf16
        
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.output_dir = output_dir
        
        self.seed = seed
        self.report_to = report_to
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return vars(self)
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"TrainingConfig({self.to_dict()})"


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_model_and_tokenizer(config: TrainingConfig):
    """
    Load the model and tokenizer using Unsloth's optimized loading.
    
    Unsloth provides 2x faster loading and training compared to standard
    Hugging Face implementations through kernel optimizations.
    
    Args:
        config: TrainingConfig object with model settings.
        
    Returns:
        Tuple of (model, tokenizer) ready for fine-tuning.
        
    Example:
        >>> config = TrainingConfig(model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
        >>> model, tokenizer = load_model_and_tokenizer(config)
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            "Unsloth not installed. Install with:\n"
            "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        )
        raise
    
    logger.info(f"Loading model: {config.model_name}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"4-bit quantization: {config.load_in_4bit}")
    
    # Load model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=config.load_in_4bit,
    )
    
    logger.info("Model loaded successfully!")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def apply_lora_adapters(model, config: TrainingConfig):
    """
    Apply LoRA adapters to the model for parameter-efficient fine-tuning.
    
    LoRA (Low-Rank Adaptation) reduces the number of trainable parameters
    by factorizing weight updates into low-rank matrices.
    
    Args:
        model: The loaded language model.
        config: TrainingConfig with LoRA settings.
        
    Returns:
        Model with LoRA adapters applied.
        
    Note:
        Typical LoRA configurations:
        - r=8, alpha=16 for lightweight fine-tuning
        - r=16, alpha=32 for balanced performance (default)
        - r=32, alpha=64 for maximum expressiveness
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth not installed")
    
    logger.info("Applying LoRA adapters...")
    logger.info(f"  LoRA rank (r): {config.lora_r}")
    logger.info(f"  LoRA alpha: {config.lora_alpha}")
    logger.info(f"  LoRA dropout: {config.lora_dropout}")
    logger.info(f"  Target modules: {config.lora_target_modules}")
    
    # Apply LoRA using Unsloth's optimized implementation
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",  # Options: "none", "all", "lora_only"
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
        use_rslora=False,  # Rank-stabilized LoRA (experimental)
        loftq_config=None,  # LoftQ quantization (experimental)
    )
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model


# =============================================================================
# Dataset Loading and Preparation
# =============================================================================

def load_training_dataset(
    dataset_path: str,
    tokenizer,
    max_seq_length: int = 2048,
):
    """
    Load and prepare the training dataset.
    
    Args:
        dataset_path: Path to the processed dataset (from data_prep.py).
        tokenizer: The tokenizer for the model.
        max_seq_length: Maximum sequence length for truncation.
        
    Returns:
        Prepared dataset ready for training.
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Check if path is a HuggingFace dataset or local path
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        # Assume it's a HuggingFace dataset ID
        from datasets import load_dataset
        dataset = load_dataset(dataset_path)
    
    logger.info(f"Dataset loaded: {dataset}")
    
    return dataset


# =============================================================================
# Training Setup
# =============================================================================

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: TrainingConfig,
):
    """
    Create and configure the SFTTrainer for supervised fine-tuning.
    
    Args:
        model: Model with LoRA adapters.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset (optional).
        config: TrainingConfig with training settings.
        
    Returns:
        Configured SFTTrainer ready for training.
    """
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    logger.info("Creating SFTTrainer...")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure training arguments
    training_args = TrainingArguments(
        # Output and logging
        output_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=["tensorboard"] if config.report_to == "none" else [config.report_to, "tensorboard"],
        
        # Training hyperparameters
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        
        # Optimization
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        
        # Misc
        seed=config.seed,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.save_steps if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        
        # Memory optimization
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    
    # Create metrics callback for research-grade logging
    metrics_callback = TrainingMetricsCallback(output_dir=str(output_dir))
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",  # Column containing formatted text
        max_seq_length=config.max_seq_length,
        packing=False,  # Set to True for efficient packing of short sequences
        callbacks=[metrics_callback],
    )
    
    logger.info("SFTTrainer created successfully!")
    return trainer, metrics_callback


# =============================================================================
# Model Saving Functions
# =============================================================================

def save_model(
    model,
    tokenizer,
    output_dir: str,
    save_method: str = "lora",
):
    """
    Save the fine-tuned model.
    
    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer.
        output_dir: Directory to save the model.
        save_method: How to save - "lora" (adapters only), "merged" (full model),
                    or "gguf" (for llama.cpp).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to: {output_path}")
    logger.info(f"Save method: {save_method}")
    
    if save_method == "lora":
        # Save only LoRA adapters (smallest size)
        model.save_pretrained(str(output_path / "lora_adapters"))
        tokenizer.save_pretrained(str(output_path / "lora_adapters"))
        logger.info("LoRA adapters saved!")
        
    elif save_method == "merged":
        # Merge LoRA weights and save full model
        try:
            from unsloth import FastLanguageModel
            model.save_pretrained_merged(
                str(output_path / "merged_model"),
                tokenizer,
                save_method="merged_16bit",  # Options: merged_16bit, merged_4bit
            )
            logger.info("Merged model saved!")
        except Exception as e:
            logger.warning(f"Could not save merged model: {e}")
            # Fallback to standard save
            model.save_pretrained(str(output_path / "model"))
            tokenizer.save_pretrained(str(output_path / "model"))
            
    elif save_method == "gguf":
        # Save as GGUF for llama.cpp inference
        try:
            from unsloth import FastLanguageModel
            model.save_pretrained_gguf(
                str(output_path / "model_gguf"),
                tokenizer,
                quantization_method="q4_k_m",  # Options: q4_k_m, q8_0, f16
            )
            logger.info("GGUF model saved!")
        except Exception as e:
            logger.warning(f"Could not save GGUF model: {e}")
    
    else:
        raise ValueError(f"Unknown save method: {save_method}")


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    config: TrainingConfig,
    dataset_path: str,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Main training function that orchestrates the entire fine-tuning process.
    
    Args:
        config: TrainingConfig with all settings.
        dataset_path: Path to the prepared dataset.
        resume_from_checkpoint: Path to checkpoint to resume from.
    """
    logger.info("=" * 70)
    logger.info("UzABSA-LLM: Starting Fine-tuning with Unsloth")
    logger.info("=" * 70)
    logger.info(f"Config: {config}")
    
    # Step 1: Load model and tokenizer
    logger.info("\n[Step 1/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Step 2: Apply LoRA adapters
    logger.info("\n[Step 2/5] Applying LoRA adapters...")
    model = apply_lora_adapters(model, config)
    
    # Step 3: Load dataset
    logger.info("\n[Step 3/5] Loading training dataset...")
    dataset = load_training_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    
    # Get train and eval splits
    train_dataset = dataset.get("train", dataset)
    eval_dataset = dataset.get("validation", None)
    
    logger.info(f"Train examples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval examples: {len(eval_dataset)}")
    
    # Step 4: Create trainer
    logger.info("\n[Step 4/5] Creating trainer...")
    trainer, metrics_callback = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    
    # Print GPU memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        logger.info(f"GPU: {gpu_stats.name}")
        logger.info(f"VRAM: {gpu_stats.total_memory / 1024**3:.1f} GB")
        logger.info(f"Allocated: {allocated:.1f} GB")
        logger.info(f"Reserved: {reserved:.1f} GB")
    
    # Step 5: Train!
    logger.info("\n[Step 5/5] Starting training...")
    logger.info("-" * 70)
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info("-" * 70)
    logger.info("Training complete!")
    logger.info(f"Train loss: {train_result.training_loss:.4f}")
    logger.info(f"Total steps: {train_result.global_step}")
    
    # Log training curves summary
    curves_summary = metrics_callback.get_summary()
    if curves_summary:
        logger.info("\nTraining Curves Summary:")
        logger.info(f"  Initial loss: {curves_summary.get('initial_loss', 'N/A')}")
        logger.info(f"  Final loss:   {curves_summary.get('final_loss', 'N/A')}")
        logger.info(f"  Min loss:     {curves_summary.get('min_loss', 'N/A')}")
        logger.info(f"  Loss reduction: {curves_summary.get('loss_reduction_pct', 'N/A')}%")
        if 'best_eval_loss' in curves_summary:
            logger.info(f"  Best eval loss: {curves_summary['best_eval_loss']} (step {curves_summary['best_eval_step']})")
        if 'total_training_time_min' in curves_summary:
            logger.info(f"  Total time: {curves_summary['total_training_time_min']} min")
    
    # Save the model
    logger.info("\nSaving model...")
    save_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=config.output_dir,
        save_method="lora",
    )
    
    # Also save merged model for easy inference
    save_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=config.output_dir,
        save_method="merged",
    )
    
    # Save experiment summary JSON (for paper reproducibility)
    dataset_info = {
        "path": dataset_path,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset else 0,
    }
    save_experiment_summary(
        output_dir=config.output_dir,
        config=config,
        train_result=train_result,
        metrics_callback=metrics_callback,
        dataset_info=dataset_info,
    )
    
    # Generate GPU memory plot
    metrics_callback.plot_gpu_memory()
    
    logger.info("=" * 70)
    logger.info("Fine-tuning complete!")
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info(f"Outputs in {config.output_dir}:")
    logger.info(f"  lora_adapters/     — LoRA adapter weights")
    logger.info(f"  merged_model/      — Full merged model (for inference)")
    logger.info(f"  training_history.json  — Per-step metrics")
    logger.info(f"  training_history.csv   — Metrics as CSV")
    logger.info(f"  training_curves.png    — Loss curves plot")
    logger.info(f"  lr_schedule.png        — Learning rate schedule")
    logger.info(f"  experiment_summary.json — Full experiment config + results")
    logger.info("=" * 70)
    
    return trainer


# =============================================================================
# CLI Interface
# =============================================================================

def recommend_batch_size_for_gpu():
    """Recommend batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 1
    
    total_memory_gb = sum(
        torch.cuda.get_device_properties(i).total_memory / 1024**3
        for i in range(torch.cuda.device_count())
    )
    
    # RTX A6000: 46GB → batch_size 4-8
    # RTX A100: 40GB → batch_size 4-6
    # RTX 4090: 24GB → batch_size 2-4
    # RTX 3090: 24GB → batch_size 1-2
    
    if total_memory_gb >= 90:  # 2+ RTX A6000
        return 8
    elif total_memory_gb >= 40:  # 1 RTX A6000 or 1 A100
        return 4
    elif total_memory_gb >= 24:  # RTX 4090
        return 2
    else:
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs for Uzbek ABSA using Unsloth"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-7b",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to fine-tune (use shorthand names)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom model path (overrides --model)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    # GPU arguments
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "cuda:0", "cuda:1", "cpu"],
        help="Device to use for training (auto for multi-GPU)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Specific GPU ID to use (0, 1, 2, 3, etc)",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training with DistributedDataParallel",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/processed",
        help="Path to processed dataset",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    
    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps (-1 for epochs)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (used if max-steps=-1)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (for WandB)",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    return parser.parse_args()


def check_gpu_availability():
    """Check available GPUs and their memory."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s) available:")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        return num_gpus
    else:
        logger.warning("No GPUs found. Training will be slow on CPU.")
        return 0


def main():
    """Main entry point for the training script."""
    args = parse_args()
    
    # =========================================================================
    # GPU Configuration and Optimization
    # =========================================================================
    
    logger.info("=" * 80)
    logger.info("GPU Configuration and Setup")
    logger.info("=" * 80)
    
    # Check GPU availability
    num_gpus = check_gpu_availability()
    
    # Get detailed GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print_gpu_status()
    
    # Get GPU-specific training recommendations
    if gpu_info:
        gpu_recommendations = recommend_training_config(gpu_info)
        logger.info("\nRecommended Training Configuration:")
        for key, value in gpu_recommendations.items():
            if key != "note":
                logger.info(f"  {key}: {value}")
        logger.info(f"  Note: {gpu_recommendations['note']}")
        
        # Override batch size if not explicitly set and using defaults
        if args.batch_size == 2:  # Default value
            recommended_batch = gpu_recommendations.get("batch_size", 2)
            logger.info(f"  Auto-adjusting batch_size from 2 to {recommended_batch}")
            args.batch_size = recommended_batch
    
    # Handle GPU selection arguments
    if args.gpu_id is not None:
        if args.gpu_id >= torch.cuda.device_count():
            logger.warning(f"GPU {args.gpu_id} not found (only {torch.cuda.device_count()} GPUs available)")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            logger.info(f"Using GPU {args.gpu_id}")
            args.device_map = f"cuda:0"  # Map to first available after CUDA_VISIBLE_DEVICES
    
    if args.multi_gpu:
        logger.info("Multi-GPU training mode enabled")
        # Will be handled by DistributedDataParallel in trainer
    
    logger.info("=" * 80)
    
    # Resolve model name
    if args.model_path:
        model_name = args.model_path
    else:
        model_name = SUPPORTED_MODELS.get(args.model, args.model)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"uzabsa_{args.model}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    
    # Create configuration
    config = TrainingConfig(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.epochs,
        output_dir=str(output_dir),
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
    )
    
    # Set WandB project name
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "uzabsa-llm"
        os.environ["WANDB_NAME"] = run_name
    
    # Run training
    train(
        config=config,
        dataset_path=args.dataset,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
