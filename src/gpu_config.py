#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: GPU Configuration Module
# =============================================================================
"""
GPU configuration and management utilities.

Supports:
- GPU detection and memory reporting
- Batch size recommendations per GPU type
- Multi-GPU training setup
- Memory optimization

Tested on:
- RTX A6000 (46GB) - RECOMMENDED
- RTX A100 (40GB, 80GB)
- RTX 4090 (24GB)
- RTX 3090 (24GB)

Author: UzABSA Team
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# GPU Information and Detection
# =============================================================================

def get_gpu_info() -> Dict[int, Dict[str, any]]:
    """
    Get detailed information about all available GPUs.

    Returns:
        Dictionary mapping GPU ID to GPU properties.

    Example:
        >>> gpu_info = get_gpu_info()
        >>> for gpu_id, info in gpu_info.items():
        ...     print(f"GPU {gpu_id}: {info['name']} ({info['memory_gb']:.1f} GB)")
    """
    gpu_info = {}
    
    if not torch.cuda.is_available():
        logger.warning("No CUDA-capable GPUs found.")
        return gpu_info
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        
        gpu_info[i] = {
            "name": props.name,
            "memory_gb": memory_gb,
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
            "max_threads_per_block": props.max_threads_per_block,
        }
    
    return gpu_info


def print_gpu_status():
    """Print formatted GPU status information."""
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        logger.info("No GPUs available. CPU training will be slow.")
        return
    
    logger.info("=" * 70)
    logger.info("GPU Status")
    logger.info("=" * 70)
    
    total_memory = 0
    for gpu_id, info in gpu_info.items():
        logger.info(f"\nGPU {gpu_id}: {info['name']}")
        logger.info(f"  Memory: {info['memory_gb']:.1f} GB")
        logger.info(f"  SMs: {info['sm_count']}")
        logger.info(f"  Max threads/block: {info['max_threads_per_block']}")
        total_memory += info['memory_gb']
    
    logger.info(f"\nTotal GPU Memory: {total_memory:.1f} GB")
    logger.info("=" * 70)


# =============================================================================
# Batch Size Recommendations
# =============================================================================

GPU_BATCH_SIZE_RECOMMENDATIONS = {
    # GPU Model: (recommended_batch_size, max_batch_size, grad_accum)
    "NVIDIA RTX A6000": (8, 16, 2),      # 46GB - BEST
    "NVIDIA A100": (6, 12, 2),            # 40GB or 80GB
    "NVIDIA RTX 6000 Ada": (8, 16, 2),   # 48GB
    "NVIDIA RTX 4090": (2, 4, 4),         # 24GB
    "NVIDIA RTX 3090": (1, 2, 8),         # 24GB
    "NVIDIA RTX A5000": (4, 8, 2),        # 24GB
    "NVIDIA L40": (6, 12, 2),             # 48GB
}


def get_batch_size_recommendations(gpu_name: str) -> Tuple[int, int, int]:
    """
    Get batch size recommendations for a specific GPU.

    Args:
        gpu_name: Name of the GPU model.

    Returns:
        Tuple of (recommended_batch_size, max_batch_size, recommended_grad_accum).

    Example:
        >>> batch_size, max_batch, grad_accum = get_batch_size_recommendations("RTX A6000")
        >>> print(f"Use batch_size={batch_size}, grad_accum={grad_accum}")
    """
    for gpu_model, recommendations in GPU_BATCH_SIZE_RECOMMENDATIONS.items():
        if gpu_model.lower() in gpu_name.lower():
            return recommendations
    
    # Default recommendations if GPU not found
    logger.warning(f"Unknown GPU model: {gpu_name}. Using conservative defaults.")
    return (1, 2, 4)


def recommend_training_config(gpu_info: Dict[int, Dict]) -> Dict:
    """
    Recommend training configuration based on available GPU(s).

    Args:
        gpu_info: GPU information from get_gpu_info().

    Returns:
        Dictionary with recommended training parameters.

    Example:
        >>> gpu_info = get_gpu_info()
        >>> config = recommend_training_config(gpu_info)
        >>> print(f"Batch size: {config['batch_size']}")
        >>> print(f"Learning rate: {config['learning_rate']}")
    """
    if not gpu_info:
        logger.warning("No GPUs available. Recommending CPU training config.")
        return {
            "batch_size": 1,
            "grad_accum": 8,
            "learning_rate": 2e-4,
            "device_map": "cpu",
            "note": "CPU training - will be slow",
        }
    
    total_memory_gb = sum(info["memory_gb"] for info in gpu_info.values())
    num_gpus = len(gpu_info)
    
    # Get primary GPU info
    primary_gpu = gpu_info[0]
    gpu_name = primary_gpu["name"]
    
    batch_size, max_batch, grad_accum = get_batch_size_recommendations(gpu_name)
    
    # Adjust for multiple GPUs
    if num_gpus >= 2:
        batch_size = min(batch_size * 2, max_batch)
        grad_accum = max(1, grad_accum // 2)
        device_map = "auto"
        note = f"Multi-GPU training ({num_gpus} GPUs)"
    else:
        device_map = "cuda:0"
        note = f"Single GPU training (GPU 0: {gpu_name})"
    
    config = {
        "num_gpus": num_gpus,
        "total_memory_gb": total_memory_gb,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_batch_size": max_batch,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "device_map": device_map,
        "fp16": False,
        "bf16": True,
        "note": note,
    }
    
    return config


# =============================================================================
# Memory Optimization
# =============================================================================

def estimate_model_memory(
    model_size_billion: float,
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization_bits: int = 4,
) -> Dict[str, float]:
    """
    Estimate memory requirements for model training.

    Args:
        model_size_billion: Model size in billions of parameters.
        sequence_length: Maximum sequence length.
        batch_size: Batch size.
        quantization_bits: Quantization bits (4 for QLoRA, 16 for full).

    Returns:
        Dictionary with memory estimates in GB.

    Example:
        >>> mem = estimate_model_memory(7.0, sequence_length=2048, batch_size=4)
        >>> print(f"Estimated memory: {mem['total']:.1f} GB")
    """
    # Rough estimates (can vary with implementation)
    # Model weights: params * bytes_per_param
    # With 4-bit quantization: params * 0.5 bytes
    # With 16-bit: params * 2 bytes
    
    bytes_per_param = 0.5 if quantization_bits == 4 else 2.0
    
    # Model weights
    model_memory = (model_size_billion * 1e9 * bytes_per_param) / (1024**3)
    
    # Optimizer state (AdamW 8-bit: ~0.5x model size)
    optimizer_memory = model_memory * 0.5
    
    # Activations and gradients (rough estimate)
    # For a 7B model, hidden_dim is typically ~4096
    # activation_memory = batch_size * seq_length * hidden_dim * 4 bytes / 1GB
    # Approximate hidden_dim: 4096 for all models (works well for most LLMs)
    hidden_dim = 4096
    activation_memory = (batch_size * sequence_length * hidden_dim * 4) / (1024**3)
    
    total = model_memory + optimizer_memory + activation_memory
    
    return {
        "model_weights_gb": round(model_memory, 2),
        "optimizer_state_gb": round(optimizer_memory, 2),
        "activations_gb": round(activation_memory, 2),
        "total_gb": round(total, 2),
        "note": f"{quantization_bits}-bit quantization, batch_size={batch_size}",
    }


# =============================================================================
# Multi-GPU Setup
# =============================================================================

def setup_distributed_training():
    """Setup distributed training environment."""
    try:
        import torch.distributed as dist
        
        if not dist.is_available():
            logger.warning("Distributed training not available.")
            return False
        
        if not dist.is_initialized():
            dist.init_process_group("nccl")
            logger.info("Distributed training initialized.")
            return True
        
        return True
    
    except Exception as e:
        logger.warning(f"Could not initialize distributed training: {e}")
        return False


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Display GPU information and recommendations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU configuration utility")
    parser.add_argument("--check", action="store_true", help="Check GPU status")
    parser.add_argument("--recommend", action="store_true", help="Get training recommendations")
    parser.add_argument("--estimate", type=float, help="Estimate memory for model size (billions)")
    
    args = parser.parse_args()
    
    if args.check:
        print_gpu_status()
    
    if args.recommend:
        gpu_info = get_gpu_info()
        config = recommend_training_config(gpu_info)
        
        print("\n" + "=" * 70)
        print("Recommended Training Configuration")
        print("=" * 70)
        for key, value in config.items():
            print(f"{key:25s}: {value}")
        print("=" * 70)
    
    if args.estimate:
        mem = estimate_model_memory(args.estimate)
        print(f"\nMemory Estimation for {args.estimate}B parameter model:")
        for key, value in mem.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
