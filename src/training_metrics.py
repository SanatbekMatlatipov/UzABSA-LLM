#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Training Metrics & Experiment Tracking
# =============================================================================
"""
Research-grade training metrics, callbacks, and experiment tracking.

Provides:
1. CustomMetricsCallback — logs loss, LR, GPU memory, grad norm per step
2. Training history export to JSON/CSV
3. Automatic training curves plotting (loss, LR schedule, eval metrics)
4. Experiment summary JSON (config + final results + system info)

Designed to produce paper-quality data for academic publications.

Author: UzABSA Team
License: MIT
"""

import csv
import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Metrics Callback
# =============================================================================

class TrainingMetricsCallback(TrainerCallback):
    """
    Custom callback that records per-step training metrics for research papers.

    Captures:
    - Training loss (every logging_steps)
    - Learning rate schedule
    - Gradient norm
    - GPU memory usage (allocated / reserved)
    - Evaluation loss and metrics
    - Wall-clock time per step
    - Epoch progress

    All metrics are stored and can be exported as JSON/CSV for plotting.
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to save metrics files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-step records
        self.train_logs: List[Dict[str, Any]] = []
        self.eval_logs: List[Dict[str, Any]] = []

        # Timing
        self.train_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None

        # Running stats
        self.best_eval_loss: float = float("inf")
        self.best_eval_step: int = 0

        logger.info(f"TrainingMetricsCallback initialized → {self.output_dir}")

    # -------------------------------------------------------------------------
    # Lifecycle hooks
    # -------------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        """Record training start time."""
        self.train_start_time = time.time()
        logger.info("Training metrics collection started.")

    def on_step_begin(self, args, state, control, **kwargs):
        """Record step start time."""
        self.step_start_time = time.time()

    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        """
        Called every logging_steps. Capture all available metrics
        in the `logs` dict provided by Trainer.
        """
        if logs is None:
            return

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else None,
            "timestamp": datetime.now().isoformat(),
            "wall_time_s": round(time.time() - self.train_start_time, 2)
            if self.train_start_time
            else None,
        }

        # --- Training metrics ---
        if "loss" in logs:
            record["train_loss"] = round(logs["loss"], 6)
        if "learning_rate" in logs:
            record["learning_rate"] = logs["learning_rate"]
        if "grad_norm" in logs:
            record["grad_norm"] = round(float(logs["grad_norm"]), 6)

        # --- GPU memory ---
        if torch.cuda.is_available():
            record["gpu_mem_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1024**3, 3
            )
            record["gpu_mem_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            )

        # --- Evaluation metrics (when eval happens at a logging step) ---
        if "eval_loss" in logs:
            record["eval_loss"] = round(logs["eval_loss"], 6)
            if logs["eval_loss"] < self.best_eval_loss:
                self.best_eval_loss = logs["eval_loss"]
                self.best_eval_step = state.global_step

        # Capture any additional logged metrics (custom ones)
        for key, value in logs.items():
            if key not in ("loss", "learning_rate", "grad_norm", "eval_loss", "epoch"):
                try:
                    record[key] = round(float(value), 6)
                except (TypeError, ValueError):
                    record[key] = value

        self.train_logs.append(record)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after each evaluation. Store eval metrics separately."""
        if metrics is None:
            return

        eval_record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else None,
            "timestamp": datetime.now().isoformat(),
            "wall_time_s": round(time.time() - self.train_start_time, 2)
            if self.train_start_time
            else None,
        }

        for key, value in metrics.items():
            try:
                eval_record[key] = round(float(value), 6)
            except (TypeError, ValueError):
                eval_record[key] = value

        self.eval_logs.append(eval_record)

        # Log best eval loss
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_eval_step = state.global_step
            logger.info(
                f"New best eval loss: {eval_loss:.6f} at step {state.global_step}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Export all collected metrics when training finishes."""
        total_time = (
            time.time() - self.train_start_time if self.train_start_time else 0
        )
        logger.info(f"Training finished in {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"Total logged steps: {len(self.train_logs)}")
        logger.info(f"Total eval records: {len(self.eval_logs)}")

        # Auto-export
        self.export_json()
        self.export_csv()

        if HAS_MATPLOTLIB:
            self.plot_training_curves()
            self.plot_lr_schedule()
        else:
            logger.warning(
                "matplotlib not installed — skipping training curve plots. "
                "Install with: pip install matplotlib"
            )

    # -------------------------------------------------------------------------
    # Export methods
    # -------------------------------------------------------------------------

    def export_json(self, filename: str = "training_history.json"):
        """Export full training history as JSON."""
        path = self.output_dir / filename
        data = {
            "train_logs": self.train_logs,
            "eval_logs": self.eval_logs,
            "best_eval_loss": self.best_eval_loss
            if self.best_eval_loss < float("inf")
            else None,
            "best_eval_step": self.best_eval_step,
            "total_steps": len(self.train_logs),
            "exported_at": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Training history JSON saved to: {path}")
        return path

    def export_csv(self, filename: str = "training_history.csv"):
        """Export training metrics as CSV for easy plotting in Excel/R/pandas."""
        path = self.output_dir / filename
        if not self.train_logs:
            logger.warning("No training logs to export.")
            return None

        # Gather all unique keys across all records
        all_keys = []
        for record in self.train_logs:
            for key in record:
                if key not in all_keys:
                    all_keys.append(key)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.train_logs)

        logger.info(f"Training history CSV saved to: {path}")
        return path

    # -------------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------------

    def plot_training_curves(self, filename: str = "training_curves.png"):
        """
        Generate publication-quality training loss & eval loss curves.

        Saves a high-DPI PNG suitable for research papers.
        """
        if not HAS_MATPLOTLIB:
            return None

        path = self.output_dir / filename

        # Extract loss data
        steps = [r["step"] for r in self.train_logs if "train_loss" in r]
        losses = [r["train_loss"] for r in self.train_logs if "train_loss" in r]

        eval_steps = [r["step"] for r in self.eval_logs if "eval_loss" in r]
        eval_losses = [r["eval_loss"] for r in self.eval_logs if "eval_loss" in r]

        if not steps:
            logger.warning("No loss data to plot.")
            return None

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        # Training loss
        ax1.plot(steps, losses, label="Train Loss", color="#2196F3", linewidth=1.5, alpha=0.8)

        # Smoothed training loss (EMA)
        if len(losses) > 10:
            smoothed = _exponential_moving_average(losses, alpha=0.1)
            ax1.plot(steps, smoothed, label="Train Loss (smoothed)", color="#1565C0",
                     linewidth=2, linestyle="-")

        # Eval loss
        if eval_steps:
            ax1.plot(eval_steps, eval_losses, label="Eval Loss", color="#F44336",
                     linewidth=2, marker="o", markersize=4)

        ax1.set_xlabel("Training Steps", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training & Evaluation Loss", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)

        plt.tight_layout()
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training curves plot saved to: {path}")
        return path

    def plot_lr_schedule(self, filename: str = "lr_schedule.png"):
        """Plot learning rate schedule over training steps."""
        if not HAS_MATPLOTLIB:
            return None

        path = self.output_dir / filename

        steps = [r["step"] for r in self.train_logs if "learning_rate" in r]
        lrs = [r["learning_rate"] for r in self.train_logs if "learning_rate" in r]

        if not steps:
            logger.warning("No learning rate data to plot.")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(steps, lrs, color="#4CAF50", linewidth=2)
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule (Cosine)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))

        plt.tight_layout()
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"LR schedule plot saved to: {path}")
        return path

    def plot_gpu_memory(self, filename: str = "gpu_memory.png"):
        """Plot GPU memory usage over training."""
        if not HAS_MATPLOTLIB:
            return None

        path = self.output_dir / filename

        steps = [r["step"] for r in self.train_logs if "gpu_mem_allocated_gb" in r]
        allocated = [r["gpu_mem_allocated_gb"] for r in self.train_logs if "gpu_mem_allocated_gb" in r]
        reserved = [r["gpu_mem_reserved_gb"] for r in self.train_logs if "gpu_mem_reserved_gb" in r]

        if not steps:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(steps, allocated, label="Allocated", color="#FF9800", linewidth=1.5)
        ax.plot(steps, reserved, label="Reserved", color="#F44336", linewidth=1.5, alpha=0.6)
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("GPU Memory (GB)", fontsize=12)
        ax.set_title("GPU Memory Usage", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"GPU memory plot saved to: {path}")
        return path

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of collected training metrics."""
        if not self.train_logs:
            return {}

        losses = [r["train_loss"] for r in self.train_logs if "train_loss" in r]
        lrs = [r["learning_rate"] for r in self.train_logs if "learning_rate" in r]

        summary = {
            "total_recorded_steps": len(self.train_logs),
            "total_eval_records": len(self.eval_logs),
        }

        if losses:
            summary["initial_loss"] = losses[0]
            summary["final_loss"] = losses[-1]
            summary["min_loss"] = min(losses)
            summary["max_loss"] = max(losses)
            summary["loss_reduction"] = round(losses[0] - losses[-1], 6)
            summary["loss_reduction_pct"] = round(
                (losses[0] - losses[-1]) / losses[0] * 100, 2
            ) if losses[0] > 0 else 0

        if lrs:
            summary["peak_lr"] = max(lrs)
            summary["final_lr"] = lrs[-1]

        if self.best_eval_loss < float("inf"):
            summary["best_eval_loss"] = self.best_eval_loss
            summary["best_eval_step"] = self.best_eval_step

        if self.train_start_time:
            total_time = time.time() - self.train_start_time
            summary["total_training_time_s"] = round(total_time, 1)
            summary["total_training_time_min"] = round(total_time / 60, 1)

        return summary


# =============================================================================
# Experiment Summary
# =============================================================================

def save_experiment_summary(
    output_dir: str,
    config: Any,
    train_result: Any,
    metrics_callback: Optional[TrainingMetricsCallback] = None,
    model_name: str = "",
    dataset_info: Optional[Dict] = None,
    filename: str = "experiment_summary.json",
) -> str:
    """
    Save a comprehensive experiment summary as JSON.

    This produces a single file containing everything needed
    to reproduce the experiment and report results in a paper:
    - Full hyperparameter configuration
    - System information (GPU, CUDA, Python, PyTorch versions)
    - Training results (final loss, steps, runtime)
    - Training curves summary
    - Dataset information

    Args:
        output_dir: Directory to save the summary.
        config: TrainingConfig object.
        train_result: Result from trainer.train().
        metrics_callback: The TrainingMetricsCallback instance.
        model_name: Name/path of the model.
        dataset_info: Optional dict with dataset statistics.
        filename: Output filename.

    Returns:
        Path to the saved summary file.
    """
    output_path = Path(output_dir) / filename

    summary = {
        "experiment": {
            "name": Path(output_dir).name,
            "timestamp": datetime.now().isoformat(),
            "framework": "Unsloth + HuggingFace TRL",
        },
        "model": {
            "name": model_name or getattr(config, "model_name", "unknown"),
            "max_seq_length": getattr(config, "max_seq_length", None),
            "quantization": "4-bit NF4" if getattr(config, "load_in_4bit", True) else "none",
        },
        "lora_config": {
            "r": getattr(config, "lora_r", None),
            "alpha": getattr(config, "lora_alpha", None),
            "dropout": getattr(config, "lora_dropout", None),
            "target_modules": getattr(config, "lora_target_modules", None),
        },
        "training_config": {
            "learning_rate": getattr(config, "learning_rate", None),
            "batch_size": getattr(config, "per_device_train_batch_size", None),
            "gradient_accumulation_steps": getattr(config, "gradient_accumulation_steps", None),
            "effective_batch_size": (
                getattr(config, "per_device_train_batch_size", 1)
                * getattr(config, "gradient_accumulation_steps", 1)
            ),
            "max_steps": getattr(config, "max_steps", None),
            "num_epochs": getattr(config, "num_train_epochs", None),
            "warmup_ratio": getattr(config, "warmup_ratio", None),
            "weight_decay": getattr(config, "weight_decay", None),
            "lr_scheduler": getattr(config, "lr_scheduler_type", None),
            "optimizer": getattr(config, "optim", None),
            "fp16": getattr(config, "fp16", None),
            "bf16": getattr(config, "bf16", None),
            "seed": getattr(config, "seed", None),
        },
        "system": _get_system_info(),
    }

    # Training results from HuggingFace trainer
    if train_result is not None:
        summary["training_results"] = {
            "final_train_loss": round(train_result.training_loss, 6),
            "total_steps": train_result.global_step,
            "total_flos": getattr(train_result, "total_flos", None),
        }
        # metrics dict from train_result
        if hasattr(train_result, "metrics") and train_result.metrics:
            summary["training_results"]["metrics"] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in train_result.metrics.items()
            }

    # Metrics callback summary
    if metrics_callback:
        summary["training_curves_summary"] = metrics_callback.get_summary()

    # Dataset info
    if dataset_info:
        summary["dataset"] = dataset_info

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Experiment summary saved to: {output_path}")
    return str(output_path)


# =============================================================================
# Helpers
# =============================================================================

def _get_system_info() -> Dict[str, Any]:
    """Collect system information for reproducibility."""
    info = {
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "id": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / 1024**3, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import trl
        info["trl_version"] = trl.__version__
    except ImportError:
        pass

    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        pass

    return info


def _exponential_moving_average(values: List[float], alpha: float = 0.1) -> List[float]:
    """Compute EMA for smoothing noisy loss curves."""
    smoothed = []
    current = values[0]
    for v in values:
        current = alpha * v + (1 - alpha) * current
        smoothed.append(current)
    return smoothed


# =============================================================================
# Plot comparison across models (for paper tables/figures)
# =============================================================================

def plot_model_comparison(
    experiment_dirs: List[str],
    output_path: str = "model_comparison.png",
):
    """
    Generate a comparison plot of training curves across multiple models.

    Useful for paper figures showing loss convergence of different LLMs.

    Args:
        experiment_dirs: List of output directories, each containing
                        training_history.json.
        output_path: Path for the output plot.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed — cannot generate comparison plot.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5"]

    for idx, exp_dir in enumerate(experiment_dirs):
        history_path = Path(exp_dir) / "training_history.json"
        if not history_path.exists():
            logger.warning(f"No training_history.json in {exp_dir}")
            continue

        with open(history_path, "r") as f:
            data = json.load(f)

        logs = data.get("train_logs", [])
        steps = [r["step"] for r in logs if "train_loss" in r]
        losses = [r["train_loss"] for r in logs if "train_loss" in r]

        if not steps:
            continue

        # Smooth
        smoothed = _exponential_moving_average(losses, alpha=0.1)
        label = Path(exp_dir).name
        color = colors[idx % len(colors)]
        ax.plot(steps, smoothed, label=label, color=color, linewidth=2)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax.set_title("Model Comparison — Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Model comparison plot saved to: {output_path}")
    return output_path
