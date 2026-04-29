#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete training pipeline for QDA/LDA cache line classification.

Features:
- Stratified K-fold cross-validation
- Streaming data loading (memory-efficient)
- GPU acceleration (optional)
- Comprehensive metrics and visualization
- Model export for deployment

Publication-ready code for top-tier conferences.

Author: ML Pipeline
Updated: 2026-02
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from scipy import signal as sp_signal

from config import get_config, print_config
from utils import (
    build_trace_index, IndexedTraceReader, Timer,
    assign_stratified_folds, ridge_shrinkage_cov, stable_log_det,
    compute_all_metrics, print_metrics, gpu_available, 
    get_memory_info, free_gpu_memory, to_numpy
)

try:
    import cupy as cp
    GPU = gpu_available()
    xp = cp if GPU else np
except ImportError:
    GPU = False
    xp = np


# ============================================================================
# High-Pass Filter — removes TDC DC-offset drift for cross-session robustness
# ============================================================================

_hpf_sos_cache = {}

def _get_hpf_sos(cutoff_hz, fs, order=4):
    """Cached SOS coefficient computation (once per config)."""
    key = (cutoff_hz, fs, order)
    if key not in _hpf_sos_cache:
        wn = cutoff_hz / (fs / 2.0)
        _hpf_sos_cache[key] = sp_signal.butter(order, wn,
                                                btype='high', output='sos')
    return _hpf_sos_cache[key]


def highpass_filter_batch(X_batch, cutoff_hz, fs, order=4):
    """
    Zero-phase Butterworth HPF on every row of a batch.

    The TDC baseline (DC component) drifts between sessions due to
    recalibration, temperature, and supply voltage.  The discriminative
    signal — the AC transient from cache events — is preserved.

    Uses SOS form + sosfiltfilt for numerical stability and zero phase.
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        return X_batch
    sos = _get_hpf_sos(cutoff_hz, fs, order)
    out = np.empty_like(X_batch, dtype=np.float32)
    for i in range(X_batch.shape[0]):
        out[i] = sp_signal.sosfiltfilt(sos, X_batch[i]).astype(np.float32)
    return out


def znorm_batch(X_batch):
    """
    Per-trace z-score normalization: x_i = (x_i - mean) / std.

    Removes BOTH additive DC drift AND multiplicative gain drift caused
    by TDC recalibration between L1/L2 collection runs.  HPF alone only
    removes additive drift; if the TDC sensitivity differs by e.g. 5%
    between runs, every sample is scaled by 1.05× — znorm fixes this.

    Applied AFTER HPF (HPF removes DC first, then znorm normalizes gain).
    """
    std = np.std(X_batch, axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    mean = np.mean(X_batch, axis=1, keepdims=True)
    return ((X_batch - mean) / std).astype(np.float32)


# ============================================================================
# Session Normalization (removes inter-session thermal drift)
# ============================================================================

def compute_session_stats(
    reader: 'IndexedTraceReader',
    indices: np.ndarray,
    start_samp: int,
    end_samp: int,
    session_size: int,
    batch_size: int = 2048,
    hpf_cutoff_hz: float = None,
    fs: float = 250e6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First pass: compute per-session mean and std.

    Each session is a contiguous block of `session_size` traces in the CSV,
    corresponding to one invocation of ``myapp sca``.  Traces within a
    session share the same thermal state; sessions differ.

    Returns:
        session_means: (n_sessions, d)
        session_stds:  (n_sessions, d)
    """
    d = end_samp - start_samp
    n_traces = len(indices)
    n_sessions = (n_traces + session_size - 1) // session_size

    sums  = np.zeros((n_sessions, d), dtype=np.float64)
    sumsq = np.zeros((n_sessions, d), dtype=np.float64)
    cnts  = np.zeros(n_sessions, dtype=np.int64)

    for batch_idx, X_batch in reader.iter_batches(
        batch_size=batch_size,
        start_samp=start_samp,
        end_samp=end_samp,
        indices=indices
    ):
        valid = np.isfinite(X_batch).all(axis=1)
        if not valid.all():
            X_batch = X_batch[valid]
            batch_idx = batch_idx[valid]
            if len(batch_idx) == 0:
                continue

        if hpf_cutoff_hz is not None and hpf_cutoff_hz > 0:
            X_batch = highpass_filter_batch(X_batch, hpf_cutoff_hz, fs)

        X_f64 = X_batch.astype(np.float64)
        sess_ids = batch_idx // session_size

        for s in np.unique(sess_ids):
            mask = sess_ids == s
            sums[s]  += X_f64[mask].sum(axis=0)
            sumsq[s] += (X_f64[mask] ** 2).sum(axis=0)
            cnts[s]  += mask.sum()

    means = np.zeros_like(sums)
    stds  = np.ones_like(sums)
    for s in range(n_sessions):
        if cnts[s] > 1:
            means[s] = sums[s] / cnts[s]
            var_s = sumsq[s] / cnts[s] - means[s] ** 2
            stds[s] = np.sqrt(np.maximum(var_s, 1e-10))

    return means, stds


def session_normalize_batch(X_batch, batch_idx, sess_means, sess_stds, session_size):
    """
    Normalize each trace by its session's mean and std (in-place-safe).

    After this step every session has zero mean and unit std per feature,
    removing the thermal baseline that differs between sessions.
    """
    X_out = X_batch.copy()
    sess_ids = batch_idx // session_size
    for s in np.unique(sess_ids):
        mask = sess_ids == s
        X_out[mask] = ((X_out[mask] - sess_means[s]) / sess_stds[s]).astype(X_batch.dtype)
    return X_out


# ============================================================================
# PCA Projection (Analytical — from sufficient statistics, no extra data pass)
# ============================================================================

def compute_pca_projection(mean0, cov0, n0, mean1, cov1, n1, n_components):
    """
    Compute PCA projection from sufficient statistics.
    
    Uses StandardScaler (global mean + std) followed by PCA on the
    pooled covariance — exactly matching sklearn's Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=K))
    ]).
    
    Computed analytically from already-accumulated statistics (no extra data pass).
    
    Returns:
        pca_mean:   (d,) global mean for centering
        pca_std:    (d,) global std for scaling
        pca_V:      (d, K) projection matrix
        pca_ev:     (K,) explained variance ratios
    """
    N = n0 + n1
    
    # Global mean (weighted by class counts)
    global_mean = (n0 * mean0 + n1 * mean1) / N
    
    # Pooled covariance including between-class scatter
    cov_pooled = (n0 * cov0 + n1 * cov1) / N
    d0 = mean0 - global_mean
    d1 = mean1 - global_mean
    cov_pooled += (n0 * np.outer(d0, d0) + n1 * np.outer(d1, d1)) / N
    
    # StandardScaler: std = sqrt(diag(Σ_pooled))
    global_std = np.sqrt(np.maximum(np.diag(cov_pooled), 0.0))
    global_std = np.where(global_std < 1e-10, 1.0, global_std)
    
    # Scale the pooled covariance: Σ_scaled = D^{-1} Σ D^{-1}
    inv_std = 1.0 / global_std
    cov_scaled = cov_pooled * np.outer(inv_std, inv_std)
    
    # Eigendecompose (descending)
    eigvals, eigvecs = np.linalg.eigh(cov_scaled)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    K = min(n_components, len(eigvals))
    V = eigvecs[:, :K]
    
    # Explained variance ratios
    total_var = max(eigvals.sum(), 1e-12)
    ev_ratio = eigvals[:K] / total_var
    
    return global_mean, global_std, V.astype(np.float64), ev_ratio


def project_to_pca(X, pca_mean, pca_std, pca_V):
    """Project data to PCA space: X_pca = ((X - mean) / std) @ V"""
    X_centered = (X - pca_mean[None, :]) / pca_std[None, :]
    return (X_centered @ pca_V).astype(np.float32)


def project_stats_to_pca(mean, cov, pca_mean, pca_std, pca_V):
    """
    Analytically project class mean and covariance to PCA space.
    
    μ_pca = V^T · D^{-1} · (μ - μ_global)
    Σ_pca = V^T · D^{-1} · Σ · D^{-1} · V
    """
    inv_std = 1.0 / pca_std
    mean_scaled = (mean - pca_mean) * inv_std
    mean_pca = pca_V.T @ mean_scaled
    cov_scaled = cov * np.outer(inv_std, inv_std)
    cov_pca = pca_V.T @ cov_scaled @ pca_V
    return mean_pca, cov_pca


# ============================================================================
# Streaming Statistics Accumulation
# ============================================================================

def accumulate_statistics(
    reader: IndexedTraceReader,
    indices: np.ndarray,
    fold_ids: np.ndarray,
    k_folds: int,
    start_samp: int,
    end_samp: int,
    batch_size: int = 2048,
    use_gpu: bool = False,
    hpf_cutoff_hz: float = None,
    fs: float = 250e6,
    use_znorm: bool = False,
    session_stats: Tuple = None,
    session_size: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accumulate sufficient statistics per fold in one pass.
    
    Preprocessing order:  HPF → session-norm → znorm → accumulate
    """
    d = end_samp - start_samp
    
    # Initialize accumulators
    counts = np.zeros(k_folds, dtype=np.int64)
    sums = np.zeros((k_folds, d), dtype=np.float64)
    sum_sqs = np.zeros((k_folds, d, d), dtype=np.float64)
    
    n_processed = 0
    n_skipped = 0
    
    with Timer(f"Accumulating statistics for {len(indices):,} traces", verbose=False):
        for batch_idx, X_batch in reader.iter_batches(
            batch_size=batch_size,
            start_samp=start_samp,
            end_samp=end_samp,
            indices=indices
        ):
            # Check for NaN/Inf values and filter them out
            valid_mask = np.isfinite(X_batch).all(axis=1)
            if not valid_mask.all():
                n_skipped += (~valid_mask).sum()
                X_batch = X_batch[valid_mask]
                batch_idx = batch_idx[valid_mask]
                if len(batch_idx) == 0:
                    continue
            
            # ---- High-pass filter (removes TDC DC drift) ----
            if hpf_cutoff_hz is not None and hpf_cutoff_hz > 0:
                X_batch = highpass_filter_batch(X_batch, hpf_cutoff_hz, fs)
            
            # ---- Session normalization (removes inter-session thermal drift) ----
            if session_stats is not None and session_size is not None:
                X_batch = session_normalize_batch(
                    X_batch, batch_idx,
                    session_stats[0], session_stats[1], session_size
                )
            
            # ---- Z-score normalization (removes TDC gain drift) ----
            if use_znorm:
                X_batch = znorm_batch(X_batch)
            
            # Get fold IDs for this batch
            folds_batch = fold_ids[batch_idx]
            
            # Process each fold
            for f in range(k_folds):
                mask = (folds_batch == f)
                if not mask.any():
                    continue
                
                X_fold = X_batch[mask].astype(np.float64)
                n_fold = X_fold.shape[0]
                
                if use_gpu and GPU:
                    # GPU computation
                    Xg = cp.asarray(X_fold)
                    counts[f] += n_fold
                    sums[f] += cp.asnumpy(cp.sum(Xg, axis=0))
                    sum_sqs[f] += cp.asnumpy(Xg.T @ Xg)
                    del Xg
                else:
                    # CPU computation
                    counts[f] += n_fold
                    sums[f] += np.sum(X_fold, axis=0)
                    sum_sqs[f] += X_fold.T @ X_fold
            
            n_processed += len(batch_idx)
            if n_processed % 50000 == 0:
                print(f"  Processed: {n_processed:,} / {len(indices):,}", end="\r")
        
        if use_gpu and GPU:
            free_gpu_memory()
    
    if n_skipped > 0:
        print(f"  Warning: Skipped {n_skipped:,} traces with NaN/Inf values")
    print(f"  Processed: {n_processed:,} / {len(indices):,} ✓")
    return counts, sums, sum_sqs


def statistics_to_mean_cov(count: int, sum_vec: np.ndarray, 
                          sum_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sufficient statistics to mean and covariance.
    
    Args:
        count: Number of samples
        sum_vec: Sum of samples (d,)
        sum_sq: Sum of outer products (d, d)
    
    Returns:
        (mean, covariance)
    """
    if count == 0:
        d = len(sum_vec)
        return np.zeros(d), np.eye(d)
    
    # Check for NaN/Inf in inputs
    if not np.all(np.isfinite(sum_vec)):
        raise ValueError(f"sum_vec contains non-finite values: {np.sum(~np.isfinite(sum_vec))} bad values")
    
    if not np.all(np.isfinite(sum_sq)):
        raise ValueError(f"sum_sq contains non-finite values: {np.sum(~np.isfinite(sum_sq))} bad values")
    
    mean = sum_vec / count
    
    # Check mean
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"Computed mean contains non-finite values")
    
    # Cov = E[xx^T] - E[x]E[x]^T
    cov = (sum_sq / count) - np.outer(mean, mean)
    
    # Check covariance
    if not np.all(np.isfinite(cov)):
        print(f"Warning: Covariance contains {np.sum(~np.isfinite(cov))} non-finite values")
        # Replace non-finite with zeros on off-diagonal, small value on diagonal
        cov = np.where(np.isfinite(cov), cov, 0.0)
        np.fill_diagonal(cov, np.where(np.isfinite(np.diag(cov)), np.diag(cov), 1.0))
    
    # Ensure symmetric
    cov = (cov + cov.T) / 2
    
    return mean, cov


# ============================================================================
# Model Building
# ============================================================================

def build_qda_model(mean0: np.ndarray, cov0: np.ndarray,
                   mean1: np.ndarray, cov1: np.ndarray,
                   prior0: float, prior1: float,
                   shrinkage: float = 1e-3) -> Dict:
    """
    Build QDA model parameters.
    
    Args:
        mean0, cov0: Class 0 mean and covariance
        mean1, cov1: Class 1 mean and covariance
        prior0, prior1: Class priors
        shrinkage: Covariance shrinkage parameter
    
    Returns:
        Dictionary with model parameters
    """
    # Apply shrinkage
    cov0_reg = ridge_shrinkage_cov(cov0, shrinkage)
    cov1_reg = ridge_shrinkage_cov(cov1, shrinkage)
    
    # Compute inverses and log-determinants
    inv0 = np.linalg.inv(cov0_reg)
    inv1 = np.linalg.inv(cov1_reg)
    logdet0 = stable_log_det(cov0_reg)
    logdet1 = stable_log_det(cov1_reg)
    
    # Log prior ratio
    logprior_ratio = math.log(max(prior1, 1e-12) / max(prior0, 1e-12))
    
    return {
        "mean0": mean0,
        "mean1": mean1,
        "inv0": inv0,
        "inv1": inv1,
        "logdet0": logdet0,
        "logdet1": logdet1,
        "logprior_ratio": logprior_ratio,
        "prior0": prior0,
        "prior1": prior1,
        "type": "qda"
    }


def build_lda_model(mean0: np.ndarray, cov0: np.ndarray,
                   mean1: np.ndarray, cov1: np.ndarray,
                   prior0: float, prior1: float, 
                   n0: int, n1: int,
                   shrinkage: float = 1e-3) -> Dict:
    """
    Build LDA model parameters (pooled covariance).
    
    Args:
        mean0, cov0: Class 0 mean and covariance
        mean1, cov1: Class 1 mean and covariance
        prior0, prior1: Class priors
        n0, n1: Sample counts for pooling
        shrinkage: Covariance shrinkage parameter
    
    Returns:
        Dictionary with model parameters
    """
    # Pooled covariance (weighted by counts)
    cov_pooled = (n0 * cov0 + n1 * cov1) / (n0 + n1)
    cov_pooled = ridge_shrinkage_cov(cov_pooled, shrinkage)
    
    # Since covariance is same for both classes, log-det cancels
    inv_pooled = np.linalg.inv(cov_pooled)
    logdet = stable_log_det(cov_pooled)
    
    # Log prior ratio
    logprior_ratio = math.log(max(prior1, 1e-12) / max(prior0, 1e-12))
    
    return {
        "mean0": mean0,
        "mean1": mean1,
        "inv_pooled": inv_pooled,
        "logdet": logdet,
        "logprior_ratio": logprior_ratio,
        "prior0": prior0,
        "prior1": prior1,
        "type": "lda"
    }


# ============================================================================
# Scoring Functions
# ============================================================================

def score_qda(X: np.ndarray, model: Dict, use_gpu: bool = False) -> np.ndarray:
    """
    Compute QDA discriminant scores.
    
    Score = δ1(x) - δ0(x) where δk(x) = -0.5 log|Σk| - 0.5 (x-μk)^T Σk^-1 (x-μk) + log πk
    Positive score => class 1, negative => class 0
    
    Args:
        X: Data matrix (n, d)
        model: QDA model dictionary
        use_gpu: Use GPU acceleration
    
    Returns:
        Scores array (n,)
    """
    if use_gpu and GPU:
        Xg = cp.asarray(X, dtype=cp.float32)
        m0g = cp.asarray(model["mean0"], dtype=cp.float32)
        m1g = cp.asarray(model["mean1"], dtype=cp.float32)
        inv0g = cp.asarray(model["inv0"], dtype=cp.float32)
        inv1g = cp.asarray(model["inv1"], dtype=cp.float32)
        
        Z0 = Xg - m0g[None, :]
        Z1 = Xg - m1g[None, :]
        
        q0 = cp.sum((Z0 @ inv0g) * Z0, axis=1)
        q1 = cp.sum((Z1 @ inv1g) * Z1, axis=1)
        
        delta0 = -0.5 * model["logdet0"] - 0.5 * q0
        delta1 = -0.5 * model["logdet1"] - 0.5 * q1
        
        scores = (delta1 - delta0) + model["logprior_ratio"]
        return cp.asnumpy(scores).astype(np.float32)
    else:
        Z0 = X - model["mean0"][None, :]
        Z1 = X - model["mean1"][None, :]
        
        q0 = np.sum((Z0 @ model["inv0"]) * Z0, axis=1)
        q1 = np.sum((Z1 @ model["inv1"]) * Z1, axis=1)
        
        delta0 = -0.5 * model["logdet0"] - 0.5 * q0
        delta1 = -0.5 * model["logdet1"] - 0.5 * q1
        
        scores = (delta1 - delta0) + model["logprior_ratio"]
        return scores.astype(np.float32)


def score_lda(X: np.ndarray, model: Dict, use_gpu: bool = False) -> np.ndarray:
    """
    Compute LDA discriminant scores.
    
    With pooled covariance, simplifies to linear decision boundary.
    
    Args:
        X: Data matrix (n, d)
        model: LDA model dictionary
        use_gpu: Use GPU acceleration
    
    Returns:
        Scores array (n,)
    """
    if use_gpu and GPU:
        Xg = cp.asarray(X, dtype=cp.float32)
        m0g = cp.asarray(model["mean0"], dtype=cp.float32)
        m1g = cp.asarray(model["mean1"], dtype=cp.float32)
        invg = cp.asarray(model["inv_pooled"], dtype=cp.float32)
        
        # Linear discriminant
        w = invg @ (m1g - m0g)
        c = -0.5 * (m1g @ invg @ m1g - m0g @ invg @ m0g) + model["logprior_ratio"]
        
        scores = Xg @ w + c
        return cp.asnumpy(scores).astype(np.float32)
    else:
        w = model["inv_pooled"] @ (model["mean1"] - model["mean0"])
        c = -0.5 * (model["mean1"] @ model["inv_pooled"] @ model["mean1"] - 
                    model["mean0"] @ model["inv_pooled"] @ model["mean0"]) + model["logprior_ratio"]
        
        scores = X @ w + c
        return scores.astype(np.float32)


# ============================================================================
# Cross-Validation
# ============================================================================

def cross_validate(
    reader0: IndexedTraceReader,
    reader1: IndexedTraceReader,
    indices0: np.ndarray,
    indices1: np.ndarray,
    fold_ids0: np.ndarray,
    fold_ids1: np.ndarray,
    k_folds: int,
    data_cfg,
    train_cfg,
    use_gpu: bool = False
) -> Tuple[Dict, Dict]:
    """
    Perform K-fold cross-validation for both QDA and LDA.
    
    Returns:
        (qda_results, lda_results) where each is a dict with:
            - fold_scores: List of score arrays per fold
            - fold_labels: List of label arrays per fold
            - fold_metrics: List of metric dicts per fold
            - models: List of model dicts per fold
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION")
    print("="*80)
    
    # ---- Session normalization: first pass to compute per-session stats ----
    _session_size = getattr(train_cfg, 'session_size', None)
    _hpf = getattr(train_cfg, 'hpf_cutoff_hz', None)
    sess_stats0 = sess_stats1 = None
    
    if _session_size and _session_size > 0:
        print(f"\n[0/4] Computing per-session statistics (session_size={_session_size:,})...")
        print(f"  L1: {len(indices0):,} traces → {(len(indices0) + _session_size - 1) // _session_size} sessions")
        sess_stats0 = compute_session_stats(
            reader0, indices0, data_cfg.start_samp, data_cfg.end_samp,
            _session_size, batch_size=train_cfg.io_batch_size,
            hpf_cutoff_hz=_hpf, fs=data_cfg.fs
        )
        print(f"  L2: {len(indices1):,} traces → {(len(indices1) + _session_size - 1) // _session_size} sessions")
        sess_stats1 = compute_session_stats(
            reader1, indices1, data_cfg.start_samp, data_cfg.end_samp,
            _session_size, batch_size=train_cfg.io_batch_size,
            hpf_cutoff_hz=_hpf, fs=data_cfg.fs
        )
        print(f"  ✓ Session stats computed")
    
    # Accumulate statistics per fold for each class
    print("\n[1/4] Accumulating statistics for class 0 (L1)...")
    cnt0, sum0, sumsq0 = accumulate_statistics(
        reader0, indices0, fold_ids0, k_folds,
        data_cfg.start_samp, data_cfg.end_samp,
        batch_size=train_cfg.io_batch_size,
        use_gpu=use_gpu,
        hpf_cutoff_hz=_hpf,
        fs=data_cfg.fs,
        use_znorm=getattr(train_cfg, 'use_znorm', False),
        session_stats=sess_stats0,
        session_size=_session_size
    )
    
    print("\n[2/4] Accumulating statistics for class 1 (L2)...")
    cnt1, sum1, sumsq1 = accumulate_statistics(
        reader1, indices1, fold_ids1, k_folds,
        data_cfg.start_samp, data_cfg.end_samp,
        batch_size=train_cfg.io_batch_size,
        use_gpu=use_gpu,
        hpf_cutoff_hz=_hpf,
        fs=data_cfg.fs,
        use_znorm=getattr(train_cfg, 'use_znorm', False),
        session_stats=sess_stats1,
        session_size=_session_size
    )
    
    # Build models per fold
    n_pca = getattr(train_cfg, 'n_pca', 0)
    if n_pca and n_pca > 0:
        print(f"\n[3/4] Building models with PCA({n_pca}) per fold...")
    else:
        print("\n[3/4] Building QDA and LDA models for each fold...")
    qda_models = []
    lda_models = []
    pca_projections = []  # store per-fold PCA for eval
    
    for f in range(k_folds):
        # For fold f, train on all OTHER folds
        train_folds = [i for i in range(k_folds) if i != f]
        
        # Sum statistics over training folds
        n0_train = sum(cnt0[i] for i in train_folds)
        n1_train = sum(cnt1[i] for i in train_folds)
        
        sum0_train = sum(sum0[i] for i in train_folds)
        sum1_train = sum(sum1[i] for i in train_folds)
        
        sumsq0_train = sum(sumsq0[i] for i in train_folds)
        sumsq1_train = sum(sumsq1[i] for i in train_folds)
        
        # Compute mean and covariance
        mean0, cov0 = statistics_to_mean_cov(n0_train, sum0_train, sumsq0_train)
        mean1, cov1 = statistics_to_mean_cov(n1_train, sum1_train, sumsq1_train)
        
        # --- PCA dimensionality reduction (analytical, no extra data pass) ---
        if n_pca and n_pca > 0:
            pca_mean, pca_std, pca_V, pca_ev = compute_pca_projection(
                mean0, cov0, n0_train, mean1, cov1, n1_train, n_pca
            )
            if f == 0:
                cum_var = np.cumsum(pca_ev)
                print(f"  PCA: {pca_V.shape[1]} components, "
                      f"explained variance = {cum_var[-1]:.4f}")
            
            # Project class stats to PCA space
            mean0, cov0 = project_stats_to_pca(mean0, cov0, pca_mean, pca_std, pca_V)
            mean1, cov1 = project_stats_to_pca(mean1, cov1, pca_mean, pca_std, pca_V)
            
            pca_projections.append((pca_mean, pca_std, pca_V))
        else:
            pca_projections.append(None)
        
        # Priors
        prior0 = n0_train / (n0_train + n1_train)
        prior1 = n1_train / (n0_train + n1_train)
        
        # Build models (on PCA-projected features if PCA enabled)
        qda_model = build_qda_model(
            mean0, cov0, mean1, cov1, prior0, prior1,
            shrinkage=train_cfg.covariance_shrinkage
        )
        lda_model = build_lda_model(
            mean0, cov0, mean1, cov1, prior0, prior1, n0_train, n1_train,
            shrinkage=train_cfg.covariance_shrinkage
        )
        
        qda_models.append(qda_model)
        lda_models.append(lda_model)
    
    d_eff = pca_V.shape[1] if (n_pca and n_pca > 0) else data_cfg.n_features
    print(f"  Built {k_folds} QDA and LDA models ({d_eff}D features)")
    
    # Evaluate on test folds
    print("\n[4/4] Evaluating models on test folds...")
    
    qda_fold_scores = []
    qda_fold_labels = []
    lda_fold_scores = []
    lda_fold_labels = []
    
    for f in range(k_folds):
        # Test indices for fold f
        test_idx0 = indices0[fold_ids0 == f]
        test_idx1 = indices1[fold_ids1 == f]
        
        scores_qda = []
        scores_lda = []
        labels = []
        
        # HPF params for eval (must match training preprocessing)
        _hpf = getattr(train_cfg, 'hpf_cutoff_hz', None)
        _fs  = data_cfg.fs
        _znorm = getattr(train_cfg, 'use_znorm', False)
        _pca = pca_projections[f]  # PCA projection for this fold (or None)
        
        # Score class 0 test samples
        for batch_idx, X_batch in reader0.iter_batches(
            batch_size=train_cfg.io_batch_size,
            start_samp=data_cfg.start_samp,
            end_samp=data_cfg.end_samp,
            indices=test_idx0
        ):
            if _hpf is not None and _hpf > 0:
                X_batch = highpass_filter_batch(X_batch, _hpf, _fs)
            if sess_stats0 is not None and _session_size:
                X_batch = session_normalize_batch(X_batch, batch_idx, sess_stats0[0], sess_stats0[1], _session_size)
            if _znorm:
                X_batch = znorm_batch(X_batch)
            if _pca is not None:
                X_batch = project_to_pca(X_batch, _pca[0], _pca[1], _pca[2])
            scores_qda.append(score_qda(X_batch, qda_models[f], use_gpu=use_gpu))
            scores_lda.append(score_lda(X_batch, lda_models[f], use_gpu=use_gpu))
            labels.append(np.zeros(len(batch_idx), dtype=np.int32))
        
        # Score class 1 test samples
        for batch_idx, X_batch in reader1.iter_batches(
            batch_size=train_cfg.io_batch_size,
            start_samp=data_cfg.start_samp,
            end_samp=data_cfg.end_samp,
            indices=test_idx1
        ):
            if _hpf is not None and _hpf > 0:
                X_batch = highpass_filter_batch(X_batch, _hpf, _fs)
            if sess_stats1 is not None and _session_size:
                X_batch = session_normalize_batch(X_batch, batch_idx, sess_stats1[0], sess_stats1[1], _session_size)
            if _znorm:
                X_batch = znorm_batch(X_batch)
            if _pca is not None:
                X_batch = project_to_pca(X_batch, _pca[0], _pca[1], _pca[2])
            scores_qda.append(score_qda(X_batch, qda_models[f], use_gpu=use_gpu))
            scores_lda.append(score_lda(X_batch, lda_models[f], use_gpu=use_gpu))
            labels.append(np.ones(len(batch_idx), dtype=np.int32))
        
        # Concatenate
        scores_qda = np.concatenate(scores_qda)
        scores_lda = np.concatenate(scores_lda)
        labels = np.concatenate(labels)
        
        qda_fold_scores.append(scores_qda)
        qda_fold_labels.append(labels)
        lda_fold_scores.append(scores_lda)
        lda_fold_labels.append(labels)
        
        # Compute metrics
        pred_qda = (scores_qda > 0).astype(np.int32)
        pred_lda = (scores_lda > 0).astype(np.int32)
        
        metrics_qda = compute_all_metrics(labels, pred_qda, scores_qda)
        metrics_lda = compute_all_metrics(labels, pred_lda, scores_lda)
        
        print(f"  Fold {f}: QDA AUC={metrics_qda['auc_roc']:.4f}, "
              f"LDA AUC={metrics_lda['auc_roc']:.4f}")
        
        if use_gpu and GPU:
            free_gpu_memory()
    
    # Aggregate results
    qda_results = {
        "fold_scores": qda_fold_scores,
        "fold_labels": qda_fold_labels,
        "models": qda_models
    }
    
    lda_results = {
        "fold_scores": lda_fold_scores,
        "fold_labels": lda_fold_labels,
        "models": lda_models
    }
    
    return qda_results, lda_results


# ============================================================================
# Model Export
# ============================================================================

def export_full_model(
    reader0: IndexedTraceReader,
    reader1: IndexedTraceReader,
    data_cfg,
    train_cfg,
    output_path: Path,
    model_type: str = "qda",
    max_traces: int = None
):
    """
    Train and export a model on the full dataset (no CV).
    """
    print(f"\n[Export] Training full-dataset {model_type.upper()} model...")
    
    with Timer("Full model training"):
        # Accumulate statistics for all data (or max_traces subset)
        n0_cap = min(max_traces, len(reader0)) if max_traces else len(reader0)
        n1_cap = min(max_traces, len(reader1)) if max_traces else len(reader1)
        all_idx0 = np.arange(n0_cap, dtype=np.int64)
        all_idx1 = np.arange(n1_cap, dtype=np.int64)
        
        # Single fold (all data in fold 0)
        fold_ids = np.zeros(max(len(all_idx0), len(all_idx1)), dtype=np.uint8)
        
        # Session normalization (if enabled)
        _session_size = getattr(train_cfg, 'session_size', None)
        _hpf = getattr(train_cfg, 'hpf_cutoff_hz', None)
        sess_stats0 = sess_stats1 = None
        
        if _session_size and _session_size > 0:
            sess_stats0 = compute_session_stats(
                reader0, all_idx0, data_cfg.start_samp, data_cfg.end_samp,
                _session_size, batch_size=train_cfg.io_batch_size,
                hpf_cutoff_hz=_hpf, fs=data_cfg.fs
            )
            sess_stats1 = compute_session_stats(
                reader1, all_idx1, data_cfg.start_samp, data_cfg.end_samp,
                _session_size, batch_size=train_cfg.io_batch_size,
                hpf_cutoff_hz=_hpf, fs=data_cfg.fs
            )
        
        cnt0, sum0, sumsq0 = accumulate_statistics(
            reader0, all_idx0, fold_ids[:len(all_idx0)], 1,
            data_cfg.start_samp, data_cfg.end_samp,
            batch_size=train_cfg.io_batch_size,
            hpf_cutoff_hz=_hpf,
            fs=data_cfg.fs,
            use_znorm=getattr(train_cfg, 'use_znorm', False),
            session_stats=sess_stats0,
            session_size=_session_size
        )
        
        cnt1, sum1, sumsq1 = accumulate_statistics(
            reader1, all_idx1, fold_ids[:len(all_idx1)], 1,
            data_cfg.start_samp, data_cfg.end_samp,
            batch_size=train_cfg.io_batch_size,
            hpf_cutoff_hz=_hpf,
            fs=data_cfg.fs,
            use_znorm=getattr(train_cfg, 'use_znorm', False),
            session_stats=sess_stats1,
            session_size=_session_size
        )
        
        # Get totals
        n0 = int(cnt0[0])
        n1 = int(cnt1[0])
        
        mean0, cov0 = statistics_to_mean_cov(n0, sum0[0], sumsq0[0])
        mean1, cov1 = statistics_to_mean_cov(n1, sum1[0], sumsq1[0])
        
        # --- PCA dimensionality reduction ---
        n_pca = getattr(train_cfg, 'n_pca', 0)
        pca_mean = pca_std = pca_V = None
        if n_pca and n_pca > 0:
            pca_mean, pca_std, pca_V, pca_ev = compute_pca_projection(
                mean0, cov0, n0, mean1, cov1, n1, n_pca
            )
            cum_var = np.cumsum(pca_ev)
            print(f"  PCA: {pca_V.shape[1]} components, "
                  f"explained variance = {cum_var[-1]:.4f}")
            
            mean0, cov0 = project_stats_to_pca(mean0, cov0, pca_mean, pca_std, pca_V)
            mean1, cov1 = project_stats_to_pca(mean1, cov1, pca_mean, pca_std, pca_V)
        
        prior0 = n0 / (n0 + n1)
        prior1 = n1 / (n0 + n1)
        
        # Build model
        if model_type.lower() == "qda":
            model = build_qda_model(
                mean0, cov0, mean1, cov1, prior0, prior1,
                shrinkage=train_cfg.covariance_shrinkage
            )
        else:
            model = build_lda_model(
                mean0, cov0, mean1, cov1, prior0, prior1, n0, n1,
                shrinkage=train_cfg.covariance_shrinkage
            )
        
        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model_type": model_type,
            "mean0": model["mean0"].astype(np.float64),
            "mean1": model["mean1"].astype(np.float64),
            "logprior_ratio": np.float64(model["logprior_ratio"]),
            "prior0": np.float64(prior0),
            "prior1": np.float64(prior1),
            "n_train0": np.int64(n0),
            "n_train1": np.int64(n1),
            # Preprocessing info
            "fs": np.float64(data_cfg.fs),
            "t0_us": np.float64(data_cfg.t0_us),
            "t1_us": np.float64(data_cfg.t1_us),
            "start_samp": np.int64(data_cfg.start_samp),
            "end_samp": np.int64(data_cfg.end_samp),
            "n_features": np.int64(data_cfg.n_features),
            "shrinkage": np.float64(train_cfg.covariance_shrinkage),
            "class_names": np.array(data_cfg.class_names),
            # Preprocessing — HPF cutoff saved so inference can auto-match
            "hpf_cutoff_hz": np.float64(getattr(train_cfg, 'hpf_cutoff_hz', 0.0) or 0.0),
            "use_znorm": np.bool_(getattr(train_cfg, 'use_znorm', False)),
            "session_norm": np.bool_(_session_size is not None and _session_size > 0),
            "n_pca": np.int64(n_pca if n_pca else 0),
        }
        
        # Save TDC delay if provided
        _tdc_delay = getattr(train_cfg, 'tdc_delay', None)
        if _tdc_delay:
            save_dict["tdc_delay"] = np.array(_tdc_delay, dtype='U64')
        
        # Save PCA projection if used
        if pca_V is not None:
            save_dict["pca_mean"] = pca_mean.astype(np.float64)
            save_dict["pca_std"] = pca_std.astype(np.float64)
            save_dict["pca_V"] = pca_V.astype(np.float64)
        
        if model_type.lower() == "qda":
            save_dict.update({
                "inv0": model["inv0"].astype(np.float64),
                "inv1": model["inv1"].astype(np.float64),
                "logdet0": np.float64(model["logdet0"]),
                "logdet1": np.float64(model["logdet1"]),
            })
        else:  # LDA
            save_dict.update({
                "inv_pooled": model["inv_pooled"].astype(np.float64),
                "logdet": np.float64(model["logdet"]),
            })
        
        np.savez(output_path, **save_dict)
        print(f"  ✓ Model saved to: {output_path}")


# ============================================================================
# Diagnostics — Signal Localization
# ============================================================================

def _run_diagnostics(reader0, reader1, data_cfg, train_cfg):
    """
    Plot mean traces per class and per-sample Welch t-statistic to
    identify where the L1/L2 discriminative signal lives.

    Three panels:
      1. Mean traces overlaid (L1 blue, L2 red) ± 1 std
      2. Difference (mean_L1 - mean_L2)
      3. Per-sample Welch t-statistic — spikes where L1 ≠ L2

    Also suggests an optimal --t0-us / --t1-us crop window.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "="*80)
    print("SIGNAL DIAGNOSTICS")
    print("="*80)

    fs = data_cfg.fs
    # Use full trace (up to 8 µs) regardless of current crop
    n_samp = reader0.meta.get('max_len', 2000)
    start_samp = 0
    end_samp = min(n_samp, round(8.0e-6 * fs))
    d = end_samp - start_samp

    hpf = getattr(train_cfg, 'hpf_cutoff_hz', None)
    use_zn = getattr(train_cfg, 'use_znorm', False)
    n_diag = min(50000, len(reader0), len(reader1))
    diag_idx = np.arange(n_diag, dtype=np.int64)

    print(f"\n[Diag] Using {n_diag:,} traces/class, {d} samples"
          f", HPF={'%.1f MHz'%(hpf/1e6) if hpf else 'off'}"
          f", znorm={'on' if use_zn else 'off'}")

    def _accum(reader, indices):
        s = np.zeros(d, dtype=np.float64)
        sq = np.zeros(d, dtype=np.float64)
        cnt = 0
        for _, X in reader.iter_batches(
            batch_size=2048, start_samp=start_samp,
            end_samp=end_samp, indices=indices
        ):
            valid = np.isfinite(X).all(axis=1)
            X = X[valid]
            if hpf and hpf > 0:
                X = highpass_filter_batch(X, hpf, fs)
            if use_zn:
                X = znorm_batch(X)
            X64 = X.astype(np.float64)
            s += X64.sum(axis=0)
            sq += (X64 ** 2).sum(axis=0)
            cnt += len(X64)
        mean = s / cnt
        var = (sq / cnt) - mean ** 2
        return mean, var, cnt

    print("  Computing L1 statistics...")
    mean0, var0, cnt0 = _accum(reader0, diag_idx)
    print("  Computing L2 statistics...")
    mean1, var1, cnt1 = _accum(reader1, diag_idx)

    std0 = np.sqrt(np.maximum(var0, 1e-12))
    std1 = np.sqrt(np.maximum(var1, 1e-12))

    # Welch's t-statistic per sample
    se = np.sqrt(var0 / cnt0 + var1 / cnt1 + 1e-20)
    t_stat = (mean0 - mean1) / se
    diff = mean0 - mean1
    t_us = np.arange(d) / (fs / 1e6)

    peak = np.argmax(np.abs(t_stat))
    print(f"\n  L1 traces: {cnt0:,}  |  L2 traces: {cnt1:,}")
    print(f"  Max |t|: {np.abs(t_stat[peak]):.1f} at sample {peak} ({t_us[peak]:.2f} µs)")
    print(f"  Max |diff|: {np.max(np.abs(diff)):.4f} at {t_us[np.argmax(np.abs(diff))]:.2f} µs")

    # Find discriminative region (|t| > 5)
    hot = np.abs(t_stat) > 5
    if hot.any():
        hs = np.argmax(hot)
        he = d - np.argmax(hot[::-1])
        margin = 0.3
        sug_t0 = max(0.0, t_us[hs] - margin)
        sug_t1 = t_us[min(he, d-1)] + margin
        print(f"\n  Signal region (|t|>5): [{t_us[hs]:.2f}, {t_us[min(he,d-1)]:.2f}] µs")
        print(f"  Suggested crop: --t0-us {sug_t0:.1f} --t1-us {sug_t1:.1f}")
    else:
        print("\n  ⚠ No samples with |t| > 5 — signal is very weak!")
        print("    Possible causes:")
        print("    1. TDC recalibrated between L1 and L2 runs (use run_experiments_locked.sh)")
        print("    2. Try --znorm to remove residual gain drift")
        print("    3. Signal may need amplification (double-tap ldrb in membench_core)")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax = axes[0]
    ax.plot(t_us, mean0, color='#3498db', alpha=0.8, lw=0.8, label='L1 (class 0)')
    ax.plot(t_us, mean1, color='#e74c3c', alpha=0.8, lw=0.8, label='L2 (class 1)')
    ax.fill_between(t_us, mean0-std0, mean0+std0, color='#3498db', alpha=0.1)
    ax.fill_between(t_us, mean1-std1, mean1+std1, color='#e74c3c', alpha=0.1)
    ax.set_ylabel('TDC Weight (mean ± std)')
    ax.set_title(f'Mean Trace per Class ({cnt0:,} L1 vs {cnt1:,} L2)', fontweight='bold')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_us, diff, color='purple', lw=1)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_ylabel('Difference (L1 − L2)')
    ax.set_title('Mean Trace Difference', fontweight='bold'); ax.grid(alpha=0.3)
    if hot.any():
        for i in range(len(hot)):
            if hot[i]:
                ax.axvspan(t_us[i], t_us[min(i+1, d-1)], alpha=0.15, color='red')

    ax = axes[2]
    ax.plot(t_us, t_stat, color='darkgreen', lw=1)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axhline(5, color='red', ls=':', alpha=0.5, label='|t| = 5')
    ax.axhline(-5, color='red', ls=':', alpha=0.5)
    ax.set_ylabel('Welch t-statistic')
    ax.set_xlabel('Time (µs)')
    ax.set_title('Per-Sample T-Statistic (L1 vs L2)', fontweight='bold')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    # Mark current crop window
    ct0 = data_cfg.t0_us
    ct1 = data_cfg.t1_us
    for a in axes:
        a.axvline(ct0, color='orange', ls='--', alpha=0.7)
        a.axvline(ct1, color='orange', ls='--', alpha=0.7)
    axes[0].axvline(ct0, color='orange', ls='--', alpha=0.7,
                    label=f'Crop [{ct0:.1f}, {ct1:.1f}] µs')
    axes[0].legend(loc='upper right')

    plt.suptitle('Signal Diagnostics: Where is the L1/L2 Difference?',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    fig_path = Path("signal_diagnostics.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {fig_path}")
    plt.close(fig)

    print("\n[Diagnose] Done. Use the plot to find the signal region,")
    print("           then retrain with the suggested --t0-us / --t1-us.")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train QDA/LDA models for cache line classification"
    )
    
    # Data arguments
    parser.add_argument("--l1-csv", type=str, required=True, help="Path to L1 traces CSV")
    parser.add_argument("--l2-csv", type=str, required=True, help="Path to L2 traces CSV")
    
    # Model selection
    parser.add_argument("--model", type=str, default="qda", 
                       choices=["qda", "lda", "both"],
                       help="Model type to train")
    
    # Training parameters
    parser.add_argument("--k-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Data parameters
    parser.add_argument("--fs", type=float, default=250e6, help="Sampling rate (Hz)")
    parser.add_argument("--t0-us", type=float, default=3.0, help="Start time (µs)")
    parser.add_argument("--t1-us", type=float, default=4.0, help="End time (µs)")
    parser.add_argument("--hpf", type=float, default=1.0,
                       help="High-pass filter cutoff in MHz. Removes TDC DC drift "
                            "for cross-session robustness. Set 0 to disable. (default: 1.0)")
    
    # Compute
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--batch-size", type=int, default=2048, help="IO batch size")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--export-models", action="store_true",
                       help="Export full-dataset models for deployment")
    
    # Preprocessing
    parser.add_argument("--znorm", action="store_true",
                       help="Per-trace z-score normalization. Removes TDC gain drift "
                            "between L1/L2 collection sessions (critical when TDC "
                            "recalibrates between pattern 1 and pattern 2 runs).")
    parser.add_argument("--pca", type=int, default=0,
                       help="PCA dimensionality reduction before QDA/LDA. "
                            "Reduces D features to K principal components, "
                            "cutting model parameters from D² to K². "
                            "WARNING: PCA captures noise variance, not cache signal. "
                            "Only use if you have a strong reason. "
                            "Set 0 to disable. (default: 0 = disabled)")
    parser.add_argument("--shrinkage", type=float, default=0.1, 
                       help="Covariance shrinkage (regularization). "
                            "Higher = more robust to noise. "
                            "0.1-0.3 recommended with PCA. (default: 0.1)")
    parser.add_argument("--session-norm", type=int, default=None, metavar="SIZE",
                       help="Per-session normalization. SIZE = traces per session "
                            "(= ITERS in run_experiments.sh, typically 5000). "
                            "Removes thermal drift between collection sessions by "
                            "normalizing each session to zero-mean, unit-std per feature. "
                            "CRITICAL for multi-session datasets. (default: disabled)")
    parser.add_argument("--max-traces", type=int, default=None,
                       help="Max traces PER CLASS. Keeps first N traces from each CSV. "
                            "Use to limit to a single session (e.g. --max-traces 5000). "
                            "(default: all traces)")
    parser.add_argument("--tdc-delay", type=str, default=None,
                       help="TDC calibration string (e.g. '0x70F10314,0x00003DD5'). "
                            "Saved into model .npz so inference can auto-read it. "
                            "Also saved to tdc_calibration.txt.")
    
    # Diagnostics
    parser.add_argument("--diagnose", action="store_true",
                       help="Plot mean trace per class and per-sample t-statistic "
                            "to find the discriminative region. Exits after plotting.")
    
    args = parser.parse_args()
    
    # Configuration
    data_cfg, train_cfg, output_cfg, eval_cfg = get_config(model_type=args.model)
    
    # Override from command line
    data_cfg.l1_csv = Path(args.l1_csv)
    data_cfg.l2_csv = Path(args.l2_csv)
    data_cfg.fs = args.fs
    data_cfg.t0_us = args.t0_us
    data_cfg.t1_us = args.t1_us
    data_cfg.__post_init__()  # Recompute derived values
    
    train_cfg.k_folds = args.k_folds
    train_cfg.covariance_shrinkage = args.shrinkage
    train_cfg.seed = args.seed
    train_cfg.model_type = args.model
    train_cfg.io_batch_size = args.batch_size
    train_cfg.use_gpu = not args.no_gpu and gpu_available()
    train_cfg.hpf_cutoff_hz = args.hpf * 1e6 if args.hpf and args.hpf > 0 else None
    train_cfg.use_znorm = args.znorm
    train_cfg.n_pca = args.pca if args.pca and args.pca > 0 else 0
    train_cfg.session_size = args.session_norm if args.session_norm and args.session_norm > 0 else None
    train_cfg.tdc_delay = args.tdc_delay
    
    output_cfg.output_dir = Path(args.output_dir)
    output_cfg.__post_init__()
    
    # Print configuration
    print_config(data_cfg, train_cfg, output_cfg, eval_cfg)
    hpf_hz = getattr(train_cfg, 'hpf_cutoff_hz', None)
    if hpf_hz:
        print(f"[Preprocessing] HPF: {hpf_hz/1e6:.1f} MHz Butterworth order-4 zero-phase")
    else:
        print(f"[Preprocessing] HPF: disabled (raw TDC weights)")
    if train_cfg.use_znorm:
        print(f"[Preprocessing] Z-norm: per-trace z-score normalization enabled")
    if train_cfg.n_pca and train_cfg.n_pca > 0:
        d = data_cfg.n_features
        k = train_cfg.n_pca
        print(f"[Preprocessing] PCA: {d}D → {k}D "
              f"(QDA params: {2*d*d:,} → {2*k*k:,})")
    if train_cfg.session_size:
        print(f"[Preprocessing] Session-norm: {train_cfg.session_size:,} traces/session")
    if train_cfg.tdc_delay:
        print(f"[TDC Lock] {train_cfg.tdc_delay}")
    print(get_memory_info())
    
    # Build indices if needed
    print("\n[Setup] Building trace indices...")
    idx1_path, meta1 = build_trace_index(
        data_cfg.l1_csv, 
        drop_exact_len=data_cfg.drop_exact_len,
        max_len=data_cfg.max_trace_len,
        force_rebuild=False
    )
    idx2_path, meta2 = build_trace_index(
        data_cfg.l2_csv,
        drop_exact_len=data_cfg.drop_exact_len,
        max_len=data_cfg.max_trace_len,
        force_rebuild=False
    )
    
    # Create readers
    reader0 = IndexedTraceReader(data_cfg.l1_csv, idx1_path, data_cfg.l1_meta)
    reader1 = IndexedTraceReader(data_cfg.l2_csv, idx2_path, data_cfg.l2_meta)
    
    print(f"\nClass 0 (L1): {len(reader0):,} traces")
    print(f"Class 1 (L2): {len(reader1):,} traces")
    
    # ---- Apply max-traces limit ----
    max_tr = args.max_traces
    if max_tr and max_tr > 0:
        n0_use = min(max_tr, len(reader0))
        n1_use = min(max_tr, len(reader1))
        print(f"\n[max-traces] Limiting to first {max_tr:,} traces per class")
        print(f"  L1: {len(reader0):,} → {n0_use:,}")
        print(f"  L2: {len(reader1):,} → {n1_use:,}")
    else:
        n0_use = len(reader0)
        n1_use = len(reader1)
    
    # Session norm safety check
    if train_cfg.session_size:
        n_sess0 = (n0_use + train_cfg.session_size - 1) // train_cfg.session_size
        n_sess1 = (n1_use + train_cfg.session_size - 1) // train_cfg.session_size
        if n_sess0 <= 1 and n_sess1 <= 1:
            print(f"\n⚠️  WARNING: --session-norm {train_cfg.session_size} with only "
                  f"1 session per class!")
            print(f"   Session-norm makes each class zero-mean → LDA signal destroyed.")
            print(f"   Either: (a) collect multiple sessions (REPS > 1), or")
            print(f"           (b) remove --session-norm for single-session data.")
            print(f"   Continuing anyway, but expect degraded LDA accuracy.\n")
    
    # ---- Run diagnostics if requested ----
    if args.diagnose:
        _run_diagnostics(reader0, reader1, data_cfg, train_cfg)
        return
    
    # Assign folds (stratified)
    print(f"\n[Setup] Assigning {train_cfg.k_folds}-fold stratified splits...")
    fold_ids0 = assign_stratified_folds(n0_use, train_cfg.k_folds, 
                                        seed=train_cfg.seed)
    fold_ids1 = assign_stratified_folds(n1_use, train_cfg.k_folds, 
                                        seed=train_cfg.seed + 1)
    
    indices0 = np.arange(n0_use, dtype=np.int64)
    indices1 = np.arange(n1_use, dtype=np.int64)
    
    # Cross-validation
    with Timer("Total training time"):
        qda_results, lda_results = cross_validate(
            reader0, reader1,
            indices0, indices1,
            fold_ids0, fold_ids1,
            train_cfg.k_folds,
            data_cfg, train_cfg,
            use_gpu=train_cfg.use_gpu
        )
    
    # Compute aggregate metrics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for model_name, results in [("QDA", qda_results), ("LDA", lda_results)]:
        print(f"\n{model_name}:")
        
        # Aggregate across folds
        all_scores = np.concatenate(results["fold_scores"])
        all_labels = np.concatenate(results["fold_labels"])
        all_preds = (all_scores > 0).astype(np.int32)
        
        metrics = compute_all_metrics(all_labels, all_preds, all_scores)
        print_metrics(metrics, prefix="  Overall:")
        
        # Per-fold statistics
        fold_metrics = []
        for f in range(train_cfg.k_folds):
            scores_f = results["fold_scores"][f]
            labels_f = results["fold_labels"][f]
            preds_f = (scores_f > 0).astype(np.int32)
            fold_metrics.append(compute_all_metrics(labels_f, preds_f, scores_f))
        
        print("\n  Per-fold statistics:")
        for metric_name in ["accuracy", "balanced_accuracy", "f1_score", "auc_roc"]:
            values = [m[metric_name] for m in fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"    {metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results
    results_path = output_cfg.logs_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "qda": {
                "scores": [s.tolist() for s in qda_results["fold_scores"]],
                "labels": [l.tolist() for l in qda_results["fold_labels"]],
            },
            "lda": {
                "scores": [s.tolist() for s in lda_results["fold_scores"]],
                "labels": [l.tolist() for l in lda_results["fold_labels"]],
            },
            "config": {
                "k_folds": train_cfg.k_folds,
                "shrinkage": train_cfg.covariance_shrinkage,
                "seed": train_cfg.seed,
            }
        }, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Export full models if requested
    if args.export_models:
        print("\n" + "="*80)
        print("EXPORTING FULL-DATASET MODELS")
        print("="*80)
        
        if args.model in ["qda", "both"]:
            qda_path = output_cfg.models_dir / "qda_full.npz"
            export_full_model(reader0, reader1, data_cfg, train_cfg, 
                            qda_path, model_type="qda", max_traces=max_tr)
        
        if args.model in ["lda", "both"]:
            lda_path = output_cfg.models_dir / "lda_full.npz"
            export_full_model(reader0, reader1, data_cfg, train_cfg, 
                            lda_path, model_type="lda", max_traces=max_tr)
        
        # Save TDC calibration to text file for easy reuse
        if args.tdc_delay:
            cal_path = output_cfg.output_dir / "tdc_calibration.txt"
            cal_path.write_text(args.tdc_delay)
            print(f"\n✓ TDC calibration saved to: {cal_path}")
            print(f"  (inference will auto-read this)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()