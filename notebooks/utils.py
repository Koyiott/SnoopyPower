#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for cache line classification pipeline.

Includes:
- Data indexing and loading
- Metric computation
- GPU/CPU utilities
- Statistical helpers

Author: ML Pipeline
Updated: 2026-02
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# Optional GPU support
GPU = False
xp = np
try:
    import cupy as cp
    GPU = True
    xp = cp
    # Memory pool setup
    _pool = cp.cuda.MemoryPool()
    free0, total0 = cp.cuda.Device(0).mem_info
    _pool.set_limit(size=int(0.85 * total0))
    cp.cuda.set_allocator(_pool.malloc)
    print(f"[GPU] CuPy {cp.__version__} — Memory limit: {0.85*total0/1e9:.1f} GB")
except Exception as e:
    GPU = False
    xp = np


# ============================================================================
# GPU Utilities
# ============================================================================

def gpu_available() -> bool:
    """Check if GPU is available."""
    return GPU


def get_memory_info() -> str:
    """Get GPU memory info string."""
    if not GPU:
        return "[CPU mode]"
    free_b, total_b = cp.cuda.Device(0).mem_info
    used_b = total_b - free_b
    return f"[GPU] {used_b/1e9:.2f} GB / {total_b/1e9:.2f} GB used"


def free_gpu_memory(hard: bool = False):
    """Free GPU memory pools."""
    if not GPU:
        return
    try:
        cp.cuda.runtime.deviceSynchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    if hard:
        import gc
        gc.collect()


def to_numpy(x):
    """Convert CuPy or PyTorch tensor to NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    if GPU and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    # PyTorch tensor
    if hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        return x.cpu().numpy()
    return np.asarray(x)


# ============================================================================
# File Indexing for Fast Random Access
# ============================================================================

def build_trace_index(
    csv_path: Path,
    drop_exact_len: int = 8191,
    max_len: int = 3000,
    comment_prefix: bytes = b"#",
    force_rebuild: bool = False
) -> Tuple[Path, Dict]:
    """
    Build byte-offset index for CSV file for random access.
    
    Creates:
        - <csv>.idx.npy: int64 array of byte offsets
        - <csv>.meta.json: metadata (counts, lengths)
    
    Args:
        csv_path: Path to CSV file
        drop_exact_len: Drop traces of exactly this length
        max_len: Drop traces >= this length
        comment_prefix: Lines starting with this are ignored
        force_rebuild: Rebuild even if index exists
    
    Returns:
        (idx_path, metadata_dict)
    """
    csv_path = Path(csv_path)
    idx_path = csv_path.with_suffix(csv_path.suffix + ".idx.npy")
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    
    # Return existing if available and not forcing rebuild
    if not force_rebuild and idx_path.exists() and meta_path.exists():
        with meta_path.open("r") as f:
            meta = json.load(f)
        return idx_path, meta
    
    print(f"[Index] Building index for {csv_path.name}...")
    t0 = time.perf_counter()
    
    offsets = []
    n_drop_exact = 0
    n_drop_long = 0
    min_len = None
    max_len_seen = 0
    
    with csv_path.open("rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            
            s = line.strip()
            if not s or s.startswith(comment_prefix):
                continue
            
            # Count commas to get length (fast)
            L = s.count(b",") + 1
            
            # Filter traces
            if L == drop_exact_len:
                n_drop_exact += 1
                continue
            if L >= max_len:
                n_drop_long += 1
                continue
            
            # Keep this trace
            offsets.append(off)
            max_len_seen = max(max_len_seen, L)
            if min_len is None:
                min_len = L
            else:
                min_len = min(min_len, L)
    
    # Save index
    offsets = np.asarray(offsets, dtype=np.int64)
    np.save(idx_path, offsets)
    
    # Save metadata
    meta = {
        "file": str(csv_path),
        "n_traces": int(len(offsets)),
        "dropped_exact": int(n_drop_exact),
        "dropped_long": int(n_drop_long),
        "min_length": int(min_len if min_len is not None else 0),
        "max_length": int(max_len_seen),
        "index_time_sec": time.perf_counter() - t0
    }
    
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Kept: {len(offsets):,} traces")
    print(f"  Dropped: {n_drop_exact:,} (exact={drop_exact_len}) + "
          f"{n_drop_long:,} (>={max_len})")
    print(f"  Length range: [{min_len}, {max_len_seen}]")
    print(f"  Time: {meta['index_time_sec']:.2f}s")
    
    return idx_path, meta


# ============================================================================
# Trace Reader with Random Access
# ============================================================================

class IndexedTraceReader:
    """
    Fast random-access reader for large CSV trace files.
    Uses byte-offset index for O(1) access to any trace.
    """
    
    def __init__(self, csv_path: Path, idx_path: Path, meta_path: Optional[Path] = None):
        self.csv_path = Path(csv_path)
        self.idx_path = Path(idx_path)
        
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.idx_path.is_file():
            raise FileNotFoundError(f"Index not found: {self.idx_path}")
        
        # Load byte offsets (memory-mapped for efficiency)
        self.offsets = np.load(self.idx_path, mmap_mode="r")
        self.n_traces = len(self.offsets)
        
        # Load metadata if available
        self.meta = None
        if meta_path and Path(meta_path).is_file():
            try:
                with Path(meta_path).open("r") as f:
                    self.meta = json.load(f)
            except Exception:
                pass
        
        # File handle (opened on demand per process/thread)
        self._fh = None
    
    def __len__(self) -> int:
        return self.n_traces
    
    def _get_fh(self):
        """Get file handle (lazy initialization per worker)."""
        if self._fh is None or self._fh.closed:
            self._fh = self.csv_path.open("rb", buffering=16*1024*1024)
        return self._fh
    
    def read_trace(self, idx: int, start_samp: int = 0, end_samp: Optional[int] = None) -> np.ndarray:
        """
        Read a single trace and optionally crop.
        
        Args:
            idx: Trace index
            start_samp: Start sample (inclusive)
            end_samp: End sample (exclusive), None = read all
        
        Returns:
            1D float32 array
        """
        fh = self._get_fh()
        offset = int(self.offsets[idx])
        fh.seek(offset)
        line = fh.readline()
        
        # Parse line
        if end_samp is not None:
            # Only parse up to end_samp for efficiency
            s = line[:end_samp * 15].decode("ascii", errors="ignore")
            trace = np.fromstring(s, sep=",", dtype=np.float32, count=end_samp)
        else:
            s = line.decode("ascii", errors="ignore")
            trace = np.fromstring(s, sep=",", dtype=np.float32)
        
        # Crop if requested
        if start_samp > 0 or end_samp is not None:
            end = end_samp if end_samp is not None else len(trace)
            trace = trace[start_samp:end]
        
        return trace
    
    def iter_batches(
        self,
        batch_size: int,
        start_samp: int = 0,
        end_samp: Optional[int] = None,
        max_traces: Optional[int] = None,
        indices: Optional[np.ndarray] = None
    ):
        """
        Iterate over traces in batches.
        
        Args:
            batch_size: Number of traces per batch
            start_samp: Start sample for cropping
            end_samp: End sample for cropping
            max_traces: Limit number of traces (None = all)
            indices: Specific indices to read (None = sequential)
        
        Yields:
            (batch_indices, batch_data) where batch_data is (B, D) float32
        """
        if indices is not None:
            trace_indices = indices
        else:
            n = self.n_traces if max_traces is None else min(self.n_traces, max_traces)
            trace_indices = np.arange(n, dtype=np.int64)
        
        n_total = len(trace_indices)
        d = (end_samp - start_samp) if end_samp is not None else None
        
        fh = self._get_fh()
        
        for i in range(0, n_total, batch_size):
            batch_idx = trace_indices[i:i + batch_size]
            batch_size_actual = len(batch_idx)
            
            if d is None:
                # Read first trace to determine dimension
                first = self.read_trace(batch_idx[0], start_samp, end_samp)
                d = len(first)
            
            X = np.empty((batch_size_actual, d), dtype=np.float32)
            
            # Read all traces in batch
            for j in range(batch_size_actual):
                idx = int(batch_idx[j])
                offset = int(self.offsets[idx])
                fh.seek(offset)
                line = fh.readline()
                
                # Decode only what we need (more efficient)
                # Estimate: ~12 chars per number including comma
                max_chars = end_samp * 15 if end_samp else 100000
                s = line[:max_chars].decode("ascii", errors="ignore")
                
                # Parse only up to end_samp
                vals = np.fromstring(s, sep=",", dtype=np.float32, count=end_samp if end_samp else -1)
                
                # If we didn't get enough values, try full line
                if end_samp and vals.size < end_samp:
                    s = line.decode("ascii", errors="ignore")
                    vals = np.fromstring(s, sep=",", dtype=np.float32, count=end_samp)
                
                # Crop to [start_samp:end_samp]
                if end_samp is not None and start_samp == 0:
                    X[j, :] = vals[:d]
                elif end_samp is not None:
                    X[j, :] = vals[start_samp:end_samp]
                else:
                    X[j, :] = vals[start_samp:]
            
            yield batch_idx, X
    
    def close(self):
        """Close file handle."""
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
    
    def __del__(self):
        self.close()


# ============================================================================
# Metrics and Evaluation
# ============================================================================

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (0/1)
        y_pred: Predicted labels (0/1)
        y_score: Prediction scores for AUC computation (optional)
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    if y_score is not None:
        # ROC AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metrics["auc_roc"] = auc(fpr, tpr)
        except Exception:
            metrics["auc_roc"] = float("nan")
        
        # PR AUC
        try:
            metrics["auc_pr"] = average_precision_score(y_true, y_score)
        except Exception:
            metrics["auc_pr"] = float("nan")
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty-print metrics."""
    if prefix:
        print(f"\n{prefix}")
    for name, value in metrics.items():
        if np.isnan(value):
            print(f"  {name:20s}: N/A")
        else:
            print(f"  {name:20s}: {value:.4f}")


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                             normalize: Optional[str] = None) -> np.ndarray:
    """
    Compute confusion matrix with optional normalization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        normalize: None, 'true', 'pred', or 'all'
    
    Returns:
        Confusion matrix (2x2)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    return cm


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                metric_fn, n_bootstrap: int = 1000, 
                confidence: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        metric_fn: Function that takes (y_true, y_score) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95%)
        seed: Random seed
    
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    # Point estimate on full data
    point = metric_fn(y_true, y_score)
    
    # Bootstrap
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_score[idx])
            if np.isfinite(score):
                scores.append(score)
        except Exception:
            pass
    
    if len(scores) == 0:
        return point, point, point
    
    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    
    return point, lower, upper


# ============================================================================
# Statistical Utilities
# ============================================================================

def assign_stratified_folds(n: int, k: int, seed: int = 42) -> np.ndarray:
    """
    Assign balanced fold IDs for stratified k-fold CV.
    
    Args:
        n: Number of samples
        k: Number of folds
        seed: Random seed
    
    Returns:
        Array of fold IDs (0 to k-1) with shape (n,)
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.uint8)
    
    # Round-robin assignment for balance
    for j, idx in enumerate(perm):
        fold_id[idx] = j % k
    
    return fold_id


def ridge_shrinkage_cov(cov: np.ndarray, shrinkage: float = 1e-3) -> np.ndarray:
    """
    Apply ridge-like shrinkage to covariance matrix.
    
    Shrinks toward scaled identity: (1-λ)Σ + λ(tr(Σ)/d)I
    
    Args:
        cov: Covariance matrix (d, d)
        shrinkage: Shrinkage parameter λ ∈ [0, 1]
    
    Returns:
        Shrunk covariance matrix
    """
    d = cov.shape[0]
    tr = float(np.trace(cov))
    
    if not np.isfinite(tr) or tr <= 0:
        tr = 1.0
    
    scale = tr / d
    return (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(d, dtype=cov.dtype)


def stable_log_det(A: np.ndarray) -> float:
    """
    Compute log-determinant with numerical stability.
    
    Args:
        A: Square matrix
    
    Returns:
        log(det(A))
    """
    sign, logdet = np.linalg.slogdet(A)
    
    if sign <= 0 or not np.isfinite(logdet):
        # Add small jitter
        eps = 1e-6 * np.trace(A) / A.shape[0]
        sign, logdet = np.linalg.slogdet(A + eps * np.eye(A.shape[0], dtype=A.dtype))
    
    return logdet


# ============================================================================
# Timer Utility
# ============================================================================

class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.verbose:
            msg = f"[Timer]"
            if self.name:
                msg += f" {self.name}:"
            msg += f" {self.elapsed:.2f}s"
            print(msg)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Utilities module loaded successfully!")
    print(f"GPU available: {gpu_available()}")
    if gpu_available():
        print(get_memory_info())