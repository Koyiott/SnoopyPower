#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for L1 vs L2 cache line classification pipeline.

This centralized config makes it easy to switch between models and 
experimental settings for publication-quality results.

Author: Eliott Quéré
Project: SnoopyPower
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Data paths and preprocessing settings."""
    # Data paths
    base_path: Path = Path("../firmware/traces")
    l1_csv: Path = field(default_factory=lambda: Path("../firmware/traces/l1_traces.csv"))
    l2_csv: Path = field(default_factory=lambda: Path("../firmware/traces/l2_traces.csv"))
    
    # Sampling parameters
    fs: float = 250e6  # Sampling rate (Hz)
    
    # Temporal cropping (microseconds)
    t0_us: float = 3.0  # Start time
    t1_us: float = 4.0  # End time
    
    # Trace filtering
    drop_exact_len: int = 8191  # Drop traces of exactly this length
    max_trace_len: int = 3000   # Drop traces >= this length
    
    # Data limits (0 = use all)
    max_traces_per_class: int = 0  # Limit for quick experiments
    
    # Class labels
    class_names: List[str] = field(default_factory=lambda: ["L1", "L2"])
    
    def __post_init__(self):
        """Convert paths and compute derived values."""
        self.l1_csv = Path(self.l1_csv)
        self.l2_csv = Path(self.l2_csv)
        self.base_path = Path(self.base_path)
        
        # Compute sample indices from temporal window
        self.start_samp = int(round(self.t0_us * 1e-6 * self.fs))
        self.end_samp = int(round(self.t1_us * 1e-6 * self.fs))
        self.n_features = self.end_samp - self.start_samp
        
        # Index files (will be created if missing)
        self.l1_idx = self.l1_csv.with_suffix(self.l1_csv.suffix + ".idx.npy")
        self.l2_idx = self.l2_csv.with_suffix(self.l2_csv.suffix + ".idx.npy")
        self.l1_meta = self.l1_csv.with_suffix(self.l1_csv.suffix + ".meta.json")
        self.l2_meta = self.l2_csv.with_suffix(self.l2_csv.suffix + ".meta.json")


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Cross-validation
    k_folds: int = 5
    test_size: float = 0.15      # Held-out test set
    val_size: float = 0.15       # Validation set (from remaining train)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Model selection
    model_type: str = "qda"  # Options: "qda", "lda", "cnn"
    
    # Classical models (QDA/LDA)
    covariance_shrinkage: float = 1e-3  # Ridge regularization
    
    # CNN-specific
    batch_size: int = 512
    num_workers: int = 4
    max_epochs: int = 25
    patience: int = 5
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    
    # CNN architecture
    base_channels: int = 48
    dropout: float = 0.25
    use_hpf: bool = True         # High-pass filter preprocessing
    hpf_window: int = 33         # Window for running mean subtraction
    
    # Training efficiency
    use_sampled_epochs: bool = True  # Sample fixed number per epoch (faster)
    samples_per_epoch: int = 200_000
    val_samples: int = 50_000
    
    # Compute
    use_gpu: bool = True
    device: str = "cuda"  # Auto-detected in training scripts
    use_amp: bool = True  # Automatic mixed precision
    
    # I/O
    io_batch_size: int = 2048  # For streaming data readers
    

@dataclass
class OutputConfig:
    """Output paths and file naming."""
    # Base output directory
    output_dir: Path = Path("results")
    
    # Model checkpoints
    models_dir: Path = field(default_factory=lambda: Path("results/models"))
    
    # Figures and plots
    figures_dir: Path = field(default_factory=lambda: Path("results/figures"))
    
    # Logs and metrics
    logs_dir: Path = field(default_factory=lambda: Path("results/logs"))
    
    # File naming
    experiment_name: str = "cache_line_classification"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Plot settings
    figure_dpi: int = 300
    figure_format: List[str] = field(default_factory=lambda: ["pdf", "png"])
    
    def __post_init__(self):
        """Create output directories."""
        for dir_path in [self.output_dir, self.models_dir, 
                        self.figures_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Settings for model evaluation and cache line experiments."""
    # Cache line evaluation
    n_cache_lines: int = 32      # Number of cache lines to evaluate
    n_repetitions: int = 10      # Repeat evaluation this many times
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "balanced_accuracy", "f1_score", 
        "precision", "recall", "auc_roc", "auc_pr"
    ])
    
    # Confidence intervals
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Plotting
    plot_roc: bool = True
    plot_precision_recall: bool = True
    plot_confusion_matrix: bool = True
    plot_score_distributions: bool = True
    plot_per_cache_line: bool = True


# ============================================================================
# Global configuration instances
# ============================================================================

def get_config(model_type: str = "qda") -> tuple:
    """
    Get complete configuration for specified model type.
    
    Args:
        model_type: One of "qda", "lda", "cnn"
    
    Returns:
        Tuple of (DataConfig, TrainingConfig, OutputConfig, EvaluationConfig)
    """
    data_cfg = DataConfig()
    train_cfg = TrainingConfig(model_type=model_type)
    output_cfg = OutputConfig()
    eval_cfg = EvaluationConfig()
    
    # Adjust output paths based on model type
    output_cfg.models_dir = output_cfg.output_dir / "models" / model_type
    output_cfg.figures_dir = output_cfg.output_dir / "figures" / model_type
    output_cfg.logs_dir = output_cfg.output_dir / "logs" / model_type
    
    # Create directories
    for dir_path in [output_cfg.models_dir, output_cfg.figures_dir, 
                    output_cfg.logs_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return data_cfg, train_cfg, output_cfg, eval_cfg


def print_config(data_cfg, train_cfg, output_cfg, eval_cfg):
    """Print configuration summary."""
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    
    print("\n[Data]")
    print(f"  L1 CSV: {data_cfg.l1_csv}")
    print(f"  L2 CSV: {data_cfg.l2_csv}")
    print(f"  Sampling rate: {data_cfg.fs/1e6:.1f} MHz")
    print(f"  Time window: [{data_cfg.t0_us:.2f}, {data_cfg.t1_us:.2f}] µs")
    print(f"  Sample window: [{data_cfg.start_samp}, {data_cfg.end_samp}]")
    print(f"  Features: {data_cfg.n_features}")
    
    print("\n[Training]")
    print(f"  Model type: {train_cfg.model_type.upper()}")
    print(f"  K-folds: {train_cfg.k_folds}")
    print(f"  Test/Val split: {train_cfg.test_size:.1%} / {train_cfg.val_size:.1%}")
    print(f"  Random seed: {train_cfg.seed}")
    
    if train_cfg.model_type in ["qda", "lda"]:
        print(f"  Covariance shrinkage: {train_cfg.covariance_shrinkage}")
    else:
        print(f"  Batch size: {train_cfg.batch_size}")
        print(f"  Learning rate: {train_cfg.learning_rate}")
        print(f"  Max epochs: {train_cfg.max_epochs}")
        print(f"  Base channels: {train_cfg.base_channels}")
    
    print("\n[Output]")
    print(f"  Results directory: {output_cfg.output_dir}")
    print(f"  Models: {output_cfg.models_dir}")
    print(f"  Figures: {output_cfg.figures_dir}")
    
    print("\n[Evaluation]")
    print(f"  Cache lines: {eval_cfg.n_cache_lines}")
    print(f"  Repetitions: {eval_cfg.n_repetitions}")
    print(f"  Metrics: {', '.join(eval_cfg.metrics)}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    data_cfg, train_cfg, output_cfg, eval_cfg = get_config(model_type="qda")
    print_config(data_cfg, train_cfg, output_cfg, eval_cfg)
