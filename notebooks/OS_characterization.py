#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlined Side-Channel Leakage Assessment Pipeline  (v4)
===========================================================

Clean, focused analysis:

  §1  Data loading & preprocessing (HPF, crop, valid-region mask)
  §2a Normality assessment (Shapiro-Wilk, D'Agostino, KS normality)
  §2b Welch's t-test (basic pairwise comparison)
  §2c KS-based POI identification (non-parametric, distribution-free)
  §2d POI sweep — accuracy vs #POIs for LDA & QDA
  §3  LDA/QDA classifiers on optimal POIs
  §4  Publication-quality figures (9 figures, Okabe-Ito palette, 300 dpi)

Author : Eliott Quéré
Project: SnoopyPower
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

import os
import gc
import time as _time
import numpy as np
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, List

from scipy import signal, stats
from scipy.stats import shapiro, normaltest, ks_2samp, kstest

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe  # noqa: E402
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "figure.constrained_layout.use": True,
    "mathtext.fontset": "cm",
})

# Okabe-Ito colourblind-safe
PAL = {
    "L1":   "#0072B2",
    "L2":   "#D55E00",
    "DRAM": "#009E73",
    "aux":  "#CC79A7",
    "grey": "#999999",
    "gold": "#E69F00",
}
PAIR_COLS = [PAL["L1"], PAL["DRAM"], PAL["aux"]]

# ── Experiment parameters ─────────────────────────────────────────────────
BASE = Path("../firmware/traces")
FILES = {
    "L1":   BASE / "l1_traces.csv",
    "L2":   BASE / "l2_traces.csv",
    "DRAM": BASE / "DRAM_traces.csv",
}
CLASS_NAMES = list(FILES.keys())

FS       = 250e6
CUTOFF   = 3e6
HPF_ORD  = 1

N_FOLDS   = 5
ALPHA_FAM = 0.01
SEED      = 42

# ── Trace windowing (@ 250 MHz: 1 µs = 250 samples) ──────────────────────
CROP_US       = 5.0
MAX_TRACE_US  = 5.5
CROP_SAMPLES  = int(CROP_US     * 1e-6 * FS)
MAX_TRACE_LEN = int(MAX_TRACE_US * 1e-6 * FS)

# ── POI sweep ────────────────────────────────────────────────────────────
POI_SWEEP_RANGE = None  # Set by CLI or defaults computed at runtime

# ── Memory ────────────────────────────────────────────────────────────────
USE_FLOAT32 = True
MAX_TRACES_PER_CLASS = 6_000
ENABLE_MEMORY_MONITORING = True

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1

def print_memory(label=""):
    if ENABLE_MEMORY_MONITORING:
        mem = get_memory_usage()
        if mem > 0:
            print(f"  [MEM] {label}: {mem:.1f} MB")

def force_gc():
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv_traces_ragged(path, drop_len=8191, max_len=MAX_TRACE_LEN,
                           max_traces=None, use_float32=True,
                           comment_prefix="#", crop_len=CROP_SAMPLES):
    dtype = np.float32 if use_float32 else np.float64

    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    print_memory(f"Before loading {path.name}")

    rows, lens = [], []
    n_drop_exact = 0
    n_drop_long  = 0
    n_loaded     = 0

    with path.open("r") as fh:
        for line in fh:
            if max_traces and n_loaded >= max_traces:
                break
            line = line.strip()
            if not line or line.startswith(comment_prefix):
                continue
            vals = np.fromstring(line, sep=",", dtype=dtype)
            if vals.size == 0:
                continue
            if vals.size == drop_len:
                n_drop_exact += 1
                continue
            if vals.size >= max_len:
                n_drop_long += 1
                continue
            if crop_len is not None and vals.size > crop_len:
                vals = vals[:crop_len]
            rows.append(vals)
            lens.append(vals.size)
            n_loaded += 1

    if not rows:
        raise ValueError(f"No valid traces after filtering in {path}")

    n        = len(rows)
    lens_arr = np.array(lens)
    Lmin, Lmax = int(lens_arr.min()), int(lens_arr.max())
    n_total_dropped = n_drop_exact + n_drop_long

    print(f"  {path.name}: kept N={n}  (dropped {n_total_dropped}: "
          f"{n_drop_exact} @ len={drop_len}, {n_drop_long} @ len>={max_len})  "
          f"min={Lmin}  max={Lmax}  "
          f"[cropped to <={crop_len} samples = {crop_len/FS*1e6:.1f} us]")

    X = np.zeros((n, Lmax), dtype=dtype)
    for i, r in enumerate(rows):
        X[i, :r.size] = r

    del rows, lens, lens_arr
    force_gc()
    print_memory(f"After loading {path.name}")
    return X, Lmin


def hpf_filtfilt(X, b, a):
    return signal.filtfilt(b, a, X, axis=1)


def preprocess_traces(files, fs, cutoff, order, drop_len=8191,
                      max_len=MAX_TRACE_LEN, max_traces=None,
                      use_float32=True, crop_len=CROP_SAMPLES):
    print(f"[1/5] Loading traces (max_traces={max_traces or 'unlimited'}, "
          f"crop={crop_len} samples = {crop_len/fs*1e6:.1f} us) ...")
    print_memory("Start preprocessing")

    raw, min_lens = {}, {}
    for name, fp in files.items():
        X, Lmin = load_csv_traces_ragged(
            fp, drop_len=drop_len, max_len=max_len,
            max_traces=max_traces, use_float32=use_float32,
            crop_len=crop_len
        )
        raw[name] = X
        min_lens[name] = Lmin

    global_max = max(X.shape[1] for X in raw.values())
    print(f"[2/5] Aligning to global max = {global_max} "
          f"({global_max/fs*1e6:.2f} us)")

    for k in list(raw.keys()):
        X = raw[k]
        if X.shape[1] < global_max:
            padded = np.zeros((X.shape[0], global_max), dtype=X.dtype)
            padded[:, :X.shape[1]] = X
            raw[k] = padded
            del X
            force_gc()

    valid_end = min(min_lens.values())
    valid_mask = np.zeros(global_max, dtype=bool)
    valid_mask[:valid_end] = True
    print(f"[3/5] Valid region: 0-{valid_end-1} "
          f"({valid_end}/{global_max} = {valid_end/global_max:.0%})  "
          f"[{valid_end/fs*1e6:.2f} us - {global_max/fs*1e6:.2f} us]")

    b, a = signal.butter(order, cutoff / (fs / 2), btype="high")
    print(f"[4/5] HPF: Butterworth order={order}, fc={cutoff/1e6:.1f} MHz")

    filtered = {}
    for k in list(raw.keys()):
        print(f"  Filtering {k}...")
        X = raw[k]
        filtered[k] = hpf_filtfilt(X, b, a)
        del raw[k], X
        force_gc()

    del raw
    force_gc()

    print("[5/5] Final shapes:")
    for k, X in filtered.items():
        print(f"  {k:4s}: {X.shape}  ({X.shape[1]/fs*1e6:.2f} us)")

    print_memory("End preprocessing")
    return filtered, valid_mask


def build_dataset(traces_f, class_names):
    X = np.vstack([traces_f[c] for c in class_names])
    y = np.concatenate([np.full(traces_f[c].shape[0], i, dtype=np.int32)
                        for i, c in enumerate(class_names)])
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a. Normality assessment ─────────────────────────────────────────────

def test_normality_per_sample(X, y, class_names, n_test=50, alpha=0.05):
    """Test normality with Shapiro-Wilk, D'Agostino, and KS (Lilliefors)."""
    rng = np.random.default_rng(SEED)
    cols = rng.choice(X.shape[1], min(n_test, X.shape[1]), replace=False)
    cols.sort()
    results = {}
    for ci, c in enumerate(class_names):
        Xi = X[y == ci]
        n_sub = min(Xi.shape[0], 5000)
        idx = (rng.choice(Xi.shape[0], n_sub, replace=False)
               if Xi.shape[0] > 5000 else np.arange(Xi.shape[0]))
        sw_rej = da_rej = ks_rej = 0
        for col in cols:
            xc = Xi[idx, col]
            # Shapiro-Wilk
            _, p = shapiro(xc)
            if p < alpha:
                sw_rej += 1
            # D'Agostino-Pearson
            if xc.shape[0] >= 20:
                _, p2 = normaltest(xc)
                if p2 < alpha:
                    da_rej += 1
            # KS test against fitted normal
            mu, sigma = xc.mean(), xc.std(ddof=1)
            if sigma > 0:
                _, p3 = kstest(xc, 'norm', args=(mu, sigma))
                if p3 < alpha:
                    ks_rej += 1
        n = len(cols)
        results[c] = {
            "n_tested": n,
            "shapiro_reject_frac": sw_rej / n,
            "dagostino_reject_frac": da_rej / n,
            "ks_reject_frac": ks_rej / n,
        }
    return results


# ── 2b. Welch's t-test ──────────────────────────────────────────────────

def welch_t_per_sample(x1, x2):
    return stats.ttest_ind(x1, x2, axis=0, equal_var=False)


# ── 2c. KS-based POI identification ──────────────────────────────────────

def ks_test_per_sample(x1, x2):
    """Two-sample KS test per time point (non-parametric POI identification)."""
    n = x1.shape[1]
    ks_stat = np.empty(n, dtype=np.float32)
    ks_pval = np.empty(n, dtype=np.float32)
    for t in range(n):
        s, p = ks_2samp(x1[:, t], x2[:, t])
        ks_stat[t] = s
        ks_pval[t] = p
    return ks_stat, ks_pval


def identify_pois_ks(traces_f, class_names, valid_mask):
    """Identify POIs using multiclass KS: max KS statistic over all pairs."""
    n_samples = next(iter(traces_f.values())).shape[1]
    ks_combined = np.zeros(n_samples, dtype=np.float32)
    pair_ks = {}

    for ci, cj in combinations(class_names, 2):
        ks_stat, ks_pval = ks_test_per_sample(traces_f[ci], traces_f[cj])
        pair_ks[f"{ci}_vs_{cj}"] = {"stat": ks_stat, "pval": ks_pval}
        ks_combined = np.maximum(ks_combined, ks_stat)

    # Mask invalid region
    ks_combined[~valid_mask] = 0.0

    # Rank POIs by descending KS statistic
    poi_order = np.argsort(-ks_combined)
    # Only keep valid POIs
    poi_order = poi_order[valid_mask[poi_order]]

    return ks_combined, pair_ks, poi_order


# ── 2d. POI sweep — accuracy vs #POIs ───────────────────────────────────

def poi_sweep(X, y, poi_order, class_names, n_folds=5, seed=42,
              poi_counts=None):
    """Sweep over different numbers of POIs and evaluate LDA & QDA accuracy."""
    n_classes = len(class_names)

    if poi_counts is None:
        max_pois = min(len(poi_order), X.shape[1])
        # Logarithmic sweep for comprehensive coverage
        raw = np.unique(np.concatenate([
            np.array([1, 2, 3, 5, 8]),
            np.geomspace(10, max(10, max_pois), num=20, dtype=int),
        ]))
        poi_counts = raw[raw <= max_pois]

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {"n_pois": [], "lda_acc": [], "qda_acc": []}

    for n_poi in poi_counts:
        sel = poi_order[:n_poi]
        Xsel = X[:, sel]
        scaler = StandardScaler()
        Xz = scaler.fit_transform(Xsel)

        # LDA
        n_comp = min(n_poi, n_classes - 1)
        if n_comp < 1:
            n_comp = 1
        lda = LDA(n_components=n_comp)
        yp_lda = cross_val_predict(lda, Xz, y, cv=cv)
        acc_lda = accuracy_score(y, yp_lda)

        # QDA
        n_pca = min(50, n_poi)
        qda = Pipeline([
            ("pca", PCA(n_components=n_pca, random_state=seed)),
            ("qda", QDA(reg_param=0.3)),
        ])
        yp_qda = cross_val_predict(qda, Xz, y, cv=cv)
        acc_qda = accuracy_score(y, yp_qda)

        results["n_pois"].append(int(n_poi))
        results["lda_acc"].append(acc_lda)
        results["qda_acc"].append(acc_qda)
        print(f"    POIs={n_poi:4d}:  LDA={acc_lda*100:.1f}%  QDA={acc_qda*100:.1f}%")

    # Find optimal
    lda_best_idx = int(np.argmax(results["lda_acc"]))
    qda_best_idx = int(np.argmax(results["qda_acc"]))
    results["lda_best_n"] = results["n_pois"][lda_best_idx]
    results["qda_best_n"] = results["n_pois"][qda_best_idx]
    results["lda_best_acc"] = results["lda_acc"][lda_best_idx]
    results["qda_best_acc"] = results["qda_acc"][qda_best_idx]

    print(f"\n  Best LDA: {results['lda_best_n']} POIs -> {results['lda_best_acc']*100:.2f}%")
    print(f"  Best QDA: {results['qda_best_n']} POIs -> {results['qda_best_acc']*100:.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIERS ON OPTIMAL POIs
# ═══════════════════════════════════════════════════════════════════════════════

def run_classifiers(X, y, class_names, poi_order, sweep_results,
                    n_folds=5, seed=42):
    """Run LDA & QDA on optimal POIs determined by sweep."""
    print_memory("Start classifiers")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {}
    n_classes = len(class_names)

    for clf_name, best_n_key in [("LDA", "lda_best_n"), ("QDA", "qda_best_n")]:
        n_poi = sweep_results[best_n_key]
        sel = poi_order[:n_poi]
        Xsel = X[:, sel]
        scaler = StandardScaler()
        Xz = scaler.fit_transform(Xsel)

        if clf_name == "LDA":
            n_comp = min(n_poi, n_classes - 1)
            clf = LDA(n_components=max(1, n_comp))
        else:
            n_pca = min(50, n_poi)
            clf = Pipeline([
                ("pca", PCA(n_components=n_pca, random_state=seed)),
                ("qda", QDA(reg_param=0.3)),
            ])

        yp  = cross_val_predict(clf, Xz, y, cv=cv)
        acc = accuracy_score(y, yp)
        cm  = confusion_matrix(y, yp)
        results[clf_name] = {
            "y_pred": yp, "accuracy": acc, "confusion_matrix": cm,
            "Xz": Xz, "n_pois": n_poi, "poi_indices": sel,
        }
        print(f"\n{'='*60}")
        print(f"  {clf_name} - {n_folds}-fold CV on {n_poi} POIs: {acc*100:.2f}%")
        print(f"{'='*60}")
        print(f"Confusion matrix:\n{cm}\n")
        print(classification_report(y, yp, target_names=class_names, digits=4))

    print_memory("End classifiers")
    return results


def pairwise_auc(X, y, class_names, poi_order, sweep_results,
                 n_folds=5, seed=42):
    """Pairwise AUC on LDA-optimal POIs."""
    n_poi = sweep_results["lda_best_n"]
    sel = poi_order[:n_poi]
    Xsel = X[:, sel]
    Xz = StandardScaler().fit_transform(Xsel)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    auc_results = {}
    for (i, ci), (j, cj) in combinations(enumerate(class_names), 2):
        m   = (y == i) | (y == j)
        lda = LDA(n_components=1)
        sc  = cross_val_predict(lda, Xz[m], (y[m] == j).astype(int),
                                cv=cv, method="decision_function")
        auc = roc_auc_score((y[m] == j).astype(int), sc)
        fpr, tpr, _ = roc_curve((y[m] == j).astype(int), sc)
        auc_results[f"{ci}_vs_{cj}"] = {"auc": auc, "fpr": fpr, "tpr": tpr,
                                          "labels": (ci, cj)}
        print(f"  {ci} vs {cj}: AUC = {auc:.4f}")
    return auc_results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    force_gc()
    print(f"  -> {name}")

def _t_us(n, fs):
    return np.arange(n) / fs * 1e6

def _shade_pad(ax, valid_mask, t):
    yl = ax.get_ylim()
    ax.fill_between(t, yl[0], yl[1], where=~valid_mask,
                     color="0.92", alpha=0.6, zorder=0)
    ax.set_ylim(yl)

def mad_scale(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=axis)
    mad[mad == 0] = np.finfo(x.dtype).eps
    return mad


# ── Fig 1: Median +/- MAD spread ────────────────────────────────────────

def fig01_median_spread(traces_f, class_names, fs, vm):
    n = next(iter(traces_f.values())).shape[1]
    t = _t_us(n, fs)
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)

    for c in class_names:
        med = np.median(traces_f[c], axis=0)
        axes[0].plot(t, med, color=PAL[c], lw=0.9, label=c)
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].set_title("(a) Median trace per class")
    axes[0].legend(frameon=True, fancybox=False, edgecolor="0.7")
    _shade_pad(axes[0], vm, t)

    for c in class_names:
        axes[1].plot(t, mad_scale(traces_f[c]), color=PAL[c], lw=0.8, label=c)
    axes[1].set_ylabel("MAD (sigma-equiv.)")
    axes[1].set_xlabel("Time (us)")
    axes[1].set_title("(b) Intra-class spread (MAD)")
    axes[1].legend(frameon=True, fancybox=False, edgecolor="0.7")
    axes[1].set_xlim(left=0, right=CROP_US)
    _shade_pad(axes[1], vm, t)
    _save(fig, "fig01_median_spread")


# ── Fig 2: Normality assessment ─────────────────────────────────────────

def fig02_normality(X, y, class_names, nr):
    rng = np.random.default_rng(SEED)
    rep = X.shape[1] // 2
    n_sub = min(5000, X.shape[0] // len(class_names))

    fig = plt.figure(figsize=(8, 6.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Row 0: Q-Q plots
    for ci, c in enumerate(class_names):
        ax = fig.add_subplot(gs[0, ci])
        Xi = X[y == ci]
        n_draw = min(n_sub, Xi.shape[0])
        idx = rng.choice(Xi.shape[0], n_draw, replace=False)
        xc  = Xi[idx, rep]
        osm, osr = stats.probplot(xc, dist="norm", fit=False)
        ax.scatter(osm, osr, s=3, color=PAL[c], alpha=0.5, rasterized=True)
        sl, ic = np.polyfit(osm, osr, 1)
        ax.plot(osm, sl * np.array(osm) + ic, "k--", lw=0.6)
        ax.set_title(f"{c} (sample #{rep})", fontsize=9)
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Observed quantiles", fontsize=8)

    # Row 1: rejection rate bar chart (3 tests)
    ax = fig.add_subplot(gs[1, :])
    cls = list(nr.keys())
    n_cls = len(cls)
    x_pos = np.arange(n_cls)
    bar_w = 0.22

    sw = [nr[c]["shapiro_reject_frac"] for c in cls]
    da = [nr[c]["dagostino_reject_frac"] for c in cls]
    ks = [nr[c]["ks_reject_frac"] for c in cls]

    bars1 = ax.bar(x_pos - bar_w, sw, bar_w,
                   color=[PAL[c] for c in cls], edgecolor="0.3", lw=0.5,
                   label="Shapiro-Wilk")
    bars2 = ax.bar(x_pos, da, bar_w,
                   color=[PAL[c] for c in cls], edgecolor="0.3", lw=0.5,
                   alpha=0.65, hatch="//", label="D'Agostino-Pearson")
    bars3 = ax.bar(x_pos + bar_w, ks, bar_w,
                   color=[PAL[c] for c in cls], edgecolor="0.3", lw=0.5,
                   alpha=0.45, hatch="\\\\", label="KS (vs. fitted normal)")

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold")

    ax.axhline(0.05, color="k", ls=":", lw=0.7, label="Expected alpha = 0.05")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cls, fontsize=10)
    ax.set_ylabel("Fraction rejecting H0 (non-Gaussian)")
    ax.set_title("Normality rejection rate (50 random time samples, N<=5000)",
                 fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, edgecolor="0.7", ncol=2)
    ax.set_ylim(0, 1.15)

    fig.suptitle("Distribution assessment: Gaussian or not?",
                 fontsize=12, y=1.02, fontweight="medium")
    _save(fig, "fig02_normality")


# ── Fig 3: Welch's t-test ──────────────────────────────────────────────

def fig03_welch(traces_f, class_names, fs, vm):
    pairs = list(combinations(class_names, 2))
    n   = next(iter(traces_f.values())).shape[1]
    t   = _t_us(n, fs)

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(7, 2.4 * n_pairs),
                             sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for pi, (ci, cj) in enumerate(pairs):
        ax = axes[pi]
        x1, x2 = traces_f[ci], traces_f[cj]
        ts, pv = welch_t_per_sample(x1, x2)

        # Significance at Bonferroni-corrected alpha
        m_v = int(vm.sum())
        alpha_bonf = ALPHA_FAM / m_v
        sig = np.zeros(n, dtype=bool)
        sig[vm] = pv[vm] < alpha_bonf

        ax.plot(t, ts, color=PAIR_COLS[pi], lw=0.55, alpha=0.85)
        ax.fill_between(t, ts, 0, where=sig,
                         color=PAIR_COLS[pi], alpha=0.18,
                         label=f"Significant ({sig.sum()} pts)")
        ax.axhline( 4.5, color="0.5", ls="--", lw=0.5, alpha=0.7,
                    label="+/-4.5 (traditional)")
        ax.axhline(-4.5, color="0.5", ls="--", lw=0.5, alpha=0.7)
        ax.set_ylabel("Welch t-statistic")
        ax.set_title(f"{ci} vs {cj}  ({sig.sum()}/{m_v} significant, "
                     f"Bonferroni alpha={alpha_bonf:.2e})", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=6.5, frameon=True, edgecolor="0.7")
        ax.set_xlim(left=0, right=CROP_US)
        _shade_pad(ax, vm, t)

    axes[-1].set_xlabel("Time (us)")
    fig.suptitle("Welch's t-test (pairwise, Bonferroni-corrected)",
                 fontsize=11, y=1.01)
    _save(fig, "fig03_welch_ttest")


# ── Fig 4: KS-based POI identification ─────────────────────────────────

def fig04_ks_pois(ks_combined, pair_ks, poi_order, class_names, fs, vm,
                  n_top=20):
    pairs = list(combinations(class_names, 2))
    n = ks_combined.shape[0]
    t = _t_us(n, fs)
    top_pois = poi_order[:n_top]

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs + 1, 1,
                             figsize=(7.5, 2.4 * (n_pairs + 1)),
                             sharex=True)

    # Per-pair KS statistic
    for pi, (ci, cj) in enumerate(pairs):
        ax = axes[pi]
        key = f"{ci}_vs_{cj}"
        ks_stat = pair_ks[key]["stat"]
        ks_pval = pair_ks[key]["pval"]

        ax.plot(t, ks_stat, color=PAIR_COLS[pi], lw=0.6, alpha=0.8)

        # Shade significant regions
        m_v = int(vm.sum())
        alpha_bonf = ALPHA_FAM / m_v
        sig = np.zeros(n, dtype=bool)
        sig[vm] = ks_pval[vm] < alpha_bonf
        ax.fill_between(t, ks_stat, 0, where=sig,
                         color=PAIR_COLS[pi], alpha=0.15)

        ax.set_ylabel("KS statistic")
        ax.set_title(f"{ci} vs {cj}  ({sig.sum()}/{m_v} significant)",
                     fontsize=9.5)
        ax.set_xlim(left=0, right=CROP_US)
        _shade_pad(ax, vm, t)

    # Bottom panel: combined KS with top POIs marked
    ax = axes[-1]
    ax.plot(t, ks_combined, color="0.3", lw=0.7, alpha=0.7,
            label="Max KS (across pairs)")
    ax.scatter(t[top_pois], ks_combined[top_pois], s=30, color="red",
               zorder=5, marker="v", edgecolors="darkred", linewidths=0.4,
               label=f"Top {n_top} POIs")

    # Label top 5 with time
    for _rank, idx in enumerate(top_pois[:5]):
        ax.annotate(f"{t[idx]:.2f} us",
                    xy=(t[idx], ks_combined[idx]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=6, ha="center", color="darkred",
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.set_ylabel("Max KS statistic")
    ax.set_xlabel("Time (us)")
    ax.set_title(f"Combined KS — Top {n_top} POIs identified", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7, frameon=True, edgecolor="0.7")
    ax.set_xlim(left=0, right=CROP_US)
    _shade_pad(ax, vm, t)

    fig.suptitle("KS-based POI identification (non-parametric)",
                 fontsize=11, y=1.01)
    _save(fig, "fig04_ks_poi")


# ── Fig 5: POI sweep optimization ──────────────────────────────────────

def fig05_poi_sweep(sweep_results):
    n_pois = sweep_results["n_pois"]
    lda_acc = np.array(sweep_results["lda_acc"]) * 100
    qda_acc = np.array(sweep_results["qda_acc"]) * 100

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Lines
    ax.plot(n_pois, lda_acc, "o-", color=PAL["L1"], lw=1.3, ms=5,
            label="LDA", zorder=3)
    ax.plot(n_pois, qda_acc, "s--", color=PAL["L2"], lw=1.3, ms=5,
            label="QDA", zorder=3)

    # Mark optimal points
    lda_best_i = int(np.argmax(lda_acc))
    qda_best_i = int(np.argmax(qda_acc))

    ax.scatter(n_pois[lda_best_i], lda_acc[lda_best_i],
               s=120, color=PAL["L1"], edgecolors="black", linewidths=1.2,
               zorder=5, marker="*")
    ax.annotate(f"Best: {n_pois[lda_best_i]} POIs\n{lda_acc[lda_best_i]:.1f}%",
                xy=(n_pois[lda_best_i], lda_acc[lda_best_i]),
                xytext=(15, -15), textcoords="offset points",
                fontsize=7.5, color=PAL["L1"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PAL["L1"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=PAL["L1"], lw=0.5, alpha=0.9))

    ax.scatter(n_pois[qda_best_i], qda_acc[qda_best_i],
               s=120, color=PAL["L2"], edgecolors="black", linewidths=1.2,
               zorder=5, marker="*")
    ax.annotate(f"Best: {n_pois[qda_best_i]} POIs\n{qda_acc[qda_best_i]:.1f}%",
                xy=(n_pois[qda_best_i], qda_acc[qda_best_i]),
                xytext=(15, 10), textcoords="offset points",
                fontsize=7.5, color=PAL["L2"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PAL["L2"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=PAL["L2"], lw=0.5, alpha=0.9))

    # Chance level
    n_classes = 3  # L1, L2, DRAM
    ax.axhline(100.0 / n_classes, color="0.5", ls=":", lw=0.7,
               label=f"Chance ({100.0/n_classes:.1f}%)")

    ax.set_xlabel("Number of POIs (KS-ranked)")
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("POI sweep: accuracy vs number of KS-ranked POIs")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(bottom=max(0, min(min(lda_acc), min(qda_acc)) - 5),
                top=min(100.5, max(max(lda_acc), max(qda_acc)) + 3))

    _save(fig, "fig05_poi_sweep")


# ── Fig 6: LDA/QDA confusion matrices ─────────────────────────────────

def fig06_classifiers(y, class_names, cr):
    fig = plt.figure(figsize=(10, 4.5))
    gs  = GridSpec(1, 5, figure=fig, width_ratios=[3, 0.2, 1.5, 0.2, 1.5],
                   wspace=0.1)

    # LDA 2D projection (using LDA-optimal POIs)
    ax = fig.add_subplot(gs[0, 0])
    lda_info = cr["LDA"]
    Xz = lda_info["Xz"]
    n_poi = lda_info["n_pois"]

    n_comp = min(2, len(class_names) - 1)
    Z = LDA(n_components=n_comp).fit_transform(Xz, y)

    for i, c in enumerate(class_names):
        m = y == i
        if Z.shape[1] >= 2:
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.25, color=PAL[c],
                       label=c, rasterized=True)
        else:
            ax.scatter(Z[m, 0], np.zeros(m.sum()), s=4, alpha=0.25,
                       color=PAL[c], label=c, rasterized=True)
    ax.set_xlabel("LD 1")
    ax.set_ylabel("LD 2" if Z.shape[1] >= 2 else "")
    ax.set_title(f"LDA projection ({n_poi} POIs)")
    ax.legend(markerscale=2.5, frameon=True, fancybox=False, edgecolor="0.7")

    # Confusion matrices
    for idx, (cn, gc) in enumerate([("LDA", 2), ("QDA", 4)]):
        if cn not in cr:
            continue
        ax  = fig.add_subplot(gs[0, gc])
        cm  = cr[cn]["confusion_matrix"]
        cmn = cm.astype(float) / cm.sum(1, keepdims=True)
        n_poi_c = cr[cn]["n_pois"]

        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        nc = len(class_names)
        for r in range(nc):
            for c_idx in range(nc):
                tc = "white" if cmn[r, c_idx] > 0.6 else "black"
                ax.text(c_idx, r, f"{cm[r, c_idx]}\n({cmn[r, c_idx]:.0%})",
                        ha="center", va="center", fontsize=7.5, color=tc)
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        if idx == 0:
            ax.set_ylabel("True")
        ax.set_title(f"{cn} ({cr[cn]['accuracy']*100:.1f}%, {n_poi_c} POIs)",
                     fontsize=9.5)

    fig.suptitle("Classification on KS-optimal POIs", fontsize=11, y=1.02)
    _save(fig, "fig06_lda_qda_cm")


# ── Fig 7: ROC curves ──────────────────────────────────────────────────

def fig07_roc(ar, sweep_results):
    n_poi = sweep_results["lda_best_n"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for pi, (pk, res) in enumerate(ar.items()):
        ci, cj = res["labels"]
        ax.plot(res["fpr"], res["tpr"], color=PAIR_COLS[pi], lw=1.2,
                label=f"{ci}-{cj} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Pairwise ROC (LDA, {n_poi} POIs)")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    _save(fig, "fig07_roc")


# ── Fig 8: Exemplar traces ────────────────────────────────────────────

def _select_exemplar(X, valid_mask):
    region = X[:, valid_mask]
    med = np.median(region, axis=0)
    dists = np.linalg.norm(region - med, axis=1)
    return int(np.argmin(dists))


def _mad_band(X):
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    return med, med - mad, med + mad


def _hex_to_rgba(hex_col, alpha):
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig08_exemplar_traces(traces_f, class_names, fs, vm):
    n_classes = len(class_names)
    n_rows = n_classes + 1
    n_samples = next(iter(traces_f.values())).shape[1]
    t = _t_us(n_samples, fs)

    exemplars = {}
    for c in class_names:
        idx = _select_exemplar(traces_f[c], vm)
        exemplars[c] = traces_f[c][idx]
        print(f"    {c}: exemplar trace #{idx}  "
              f"(of {traces_f[c].shape[0]}, dist-to-median rank 1)")

    # Static (matplotlib)
    row_h = 1.75
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, row_h * n_rows + 0.55),
                             sharex=True, gridspec_kw={"hspace": 0.10})

    ax0 = axes[0]
    for c in class_names:
        ax0.plot(t, exemplars[c], color=PAL[c], lw=0.85, label=c,
                 zorder=3, rasterized=True)
    ax0.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax0.set_title("(a)  All classes overlaid", fontsize=10, pad=6)
    ax0.legend(loc="upper right", frameon=True, fancybox=False,
               edgecolor="0.65", fontsize=7.5, ncol=n_classes,
               handlelength=1.6, columnspacing=1.0)
    _shade_pad(ax0, vm, t)

    for ri, c in enumerate(class_names, start=1):
        ax = axes[ri]
        col = PAL[c]
        med, lo, hi = _mad_band(traces_f[c])
        ax.fill_between(t, lo, hi, color=col, alpha=0.10, lw=0,
                         label="Median +/- MAD", zorder=1)
        ax.plot(t, med, color=col, lw=0.45, alpha=0.45, ls="--", zorder=2,
                rasterized=True)
        ax.plot(t, exemplars[c], color=col, lw=0.90, zorder=3,
                label=f"{c} exemplar", rasterized=True)
        letter = chr(ord('b') + ri - 1)
        ax.set_title(f"({letter})  {c}", fontsize=10, pad=6)
        ax.set_ylabel("Amplitude (a.u.)", fontsize=9)
        ax.legend(loc="upper right", frameon=True, fancybox=False,
                  edgecolor="0.65", fontsize=7, handlelength=1.6)
        _shade_pad(ax, vm, t)

    axes[-1].set_xlabel("Time (us)", fontsize=10)
    axes[-1].set_xlim(0, CROP_US)
    fig.suptitle("Exemplar traces (L1 / L2 / DRAM)",
                 fontsize=11.5, y=1.02, fontweight="medium")
    _save(fig, "fig08_exemplar_traces")

    # Interactive (Plotly)
    subplot_titles = ["All classes overlaid"] + list(class_names)
    pfig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                          vertical_spacing=0.035,
                          subplot_titles=subplot_titles)

    for c in class_names:
        pfig.add_trace(go.Scattergl(
            x=t, y=exemplars[c], mode="lines",
            name=c, line=dict(color=PAL[c], width=1.2),
            legendgroup=c,
        ), row=1, col=1)

    for ri, c in enumerate(class_names, start=2):
        col = PAL[c]
        med, lo, hi = _mad_band(traces_f[c])
        pfig.add_trace(go.Scattergl(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=_hex_to_rgba(col, 0.12),
            line=dict(width=0), name=f"{c} +/- MAD",
            legendgroup=f"{c}_band", showlegend=True, hoverinfo="skip",
        ), row=ri, col=1)
        pfig.add_trace(go.Scattergl(
            x=t, y=med, mode="lines", name=f"{c} median",
            line=dict(color=col, width=0.7, dash="dash"),
            legendgroup=f"{c}_med", showlegend=False, opacity=0.5,
        ), row=ri, col=1)
        pfig.add_trace(go.Scattergl(
            x=t, y=exemplars[c], mode="lines", name=f"{c} exemplar",
            line=dict(color=col, width=1.3),
            legendgroup=c, showlegend=False,
        ), row=ri, col=1)

    pfig.update_layout(
        height=220 * n_rows + 80, width=1050,
        title=dict(text="Exemplar traces (interactive)",
                   font=dict(size=15, family="serif")),
        font=dict(family="serif", size=11),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#bbb", borderwidth=1),
        template="plotly_white",
        margin=dict(l=65, r=25, t=90, b=55),
    )
    for ri in range(1, n_rows + 1):
        ya = f"yaxis{ri if ri > 1 else ''}"
        pfig.update_layout(**{ya: dict(title="Amplitude (a.u.)")})
    xa_last = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
    pfig.update_layout(**{xa_last: dict(title="Time (us)")})
    pfig.update_xaxes(range=[0, CROP_US])
    html_path = OUT_DIR / "fig08_exemplar_traces.html"
    pfig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  -> fig08_exemplar_traces.html  (interactive)")


# ── Fig 9: Alignment waterfall ────────────────────────────────────────

def fig09_alignment_waterfall(traces_f, fs, vm, focus_class="L1", n_show=8):
    X = traces_f[focus_class]
    n_total = X.shape[0]
    n_samples = X.shape[1]
    t = _t_us(n_samples, fs)
    col = PAL[focus_class]

    region = X[:, vm]
    med_region = np.median(region, axis=0)
    dists = np.linalg.norm(region - med_region, axis=1)
    order = np.argsort(dists)
    percentiles = np.linspace(0, len(order) - 1, n_show, dtype=int)
    selected = order[percentiles]

    print(f"    {focus_class}: showing {n_show} traces  "
          f"(d range {dists[selected[0]]:.2f} - {dists[selected[-1]]:.2f}, "
          f"out of {n_total})")

    med_full, lo_full, hi_full = _mad_band(X)
    all_vals = np.concatenate([X[idx, vm] for idx in selected])
    ymin = min(float(all_vals.min()), float(lo_full[vm].min())) * 1.08
    ymax = max(float(all_vals.max()), float(hi_full[vm].max())) * 1.08

    n_rows = n_show + 1

    # Static (matplotlib)
    row_h = 1.15
    fig = plt.figure(figsize=(7.2, 2.2 + row_h * n_show + 0.5))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.06,
                           height_ratios=[1.6] + [1.0] * n_show)

    ax0 = fig.add_subplot(gs[0])
    ax0.fill_between(t, lo_full, hi_full, color=col, alpha=0.12, lw=0,
                      label="Median +/- MAD")
    ax0.plot(t, med_full, color=col, lw=1.0, label="Median", zorder=3)
    for idx in selected:
        ax0.plot(t, X[idx], color=col, lw=0.25, alpha=0.30, zorder=2,
                 rasterized=True)
    ax0.set_ylabel("Amplitude\n(a.u.)", fontsize=8.5)
    ax0.set_title(f"(a)  {focus_class} - Median +/- MAD with individual "
                  f"traces overlaid", fontsize=9.5, pad=5)
    ax0.legend(loc="upper right", frameon=True, fancybox=False,
               edgecolor="0.65", fontsize=7, ncol=2, handlelength=1.4)
    ax0.set_ylim(ymin, ymax)
    ax0.set_xlim(0, CROP_US)
    ax0.tick_params(labelbottom=False)
    _shade_pad(ax0, vm, t)

    for ri, idx in enumerate(selected):
        ax = fig.add_subplot(gs[ri + 1], sharex=ax0)
        ax.plot(t, X[idx], color=col, lw=0.65, zorder=3, rasterized=True)
        ax.plot(t, med_full, color=col, lw=0.30, alpha=0.25, ls="--", zorder=1)
        ax.set_ylim(ymin, ymax)
        rank = np.searchsorted(np.sort(dists), dists[idx]) + 1
        pct = rank / len(dists) * 100
        ax.text(0.995, 0.88, f"d = {dists[idx]:.1f}  (P{pct:.0f})",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
                color="0.40", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="0.8", lw=0.4))
        ax.set_ylabel(f"#{ri+1}", fontsize=8, rotation=0, labelpad=14,
                       va="center")
        ax.tick_params(axis='y', labelsize=7)
        if ri < n_show - 1:
            ax.tick_params(labelbottom=False)
        _shade_pad(ax, vm, t)

    ax.set_xlabel("Time (us)", fontsize=9.5)
    fig.suptitle(f"Intra-class alignment - {focus_class}",
                 fontsize=11, y=1.015, fontweight="medium")
    _save(fig, "fig09_alignment_waterfall")

    # Interactive (Plotly)
    sub_titles = ([f"{focus_class} - Median +/- MAD + overlaid traces"] +
                  [f"Trace #{i+1}  (d={dists[selected[i]]:.1f})"
                   for i in range(n_show)])
    pfig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                          vertical_spacing=0.015,
                          subplot_titles=sub_titles)

    pfig.add_trace(go.Scattergl(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([hi_full, lo_full[::-1]]),
        fill="toself", fillcolor=_hex_to_rgba(col, 0.12),
        line=dict(width=0), name="Median +/- MAD",
        showlegend=True, hoverinfo="skip"), row=1, col=1)
    pfig.add_trace(go.Scattergl(
        x=t, y=med_full, mode="lines", name="Median",
        line=dict(color=col, width=1.5)), row=1, col=1)
    for i, idx in enumerate(selected):
        pfig.add_trace(go.Scattergl(
            x=t, y=X[idx], mode="lines",
            line=dict(color=col, width=0.5), opacity=0.35,
            name=f"Trace #{i+1}", showlegend=False), row=1, col=1)

    for ri, idx in enumerate(selected):
        r = ri + 2
        pfig.add_trace(go.Scattergl(
            x=t, y=med_full, mode="lines",
            line=dict(color=col, width=0.5, dash="dash"), opacity=0.3,
            name="Median ref", showlegend=False), row=r, col=1)
        pfig.add_trace(go.Scattergl(
            x=t, y=X[idx], mode="lines",
            line=dict(color=col, width=1.1),
            name=f"Trace #{ri+1}", showlegend=(ri == 0)), row=r, col=1)

    pfig.update_layout(
        height=180 * n_rows + 80, width=1050,
        title=dict(text=f"Intra-class alignment - {focus_class} (interactive)",
                   font=dict(size=14, family="serif")),
        font=dict(family="serif", size=10),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#bbb", borderwidth=1),
        template="plotly_white",
        margin=dict(l=55, r=20, t=80, b=45),
    )
    for ri in range(1, n_rows + 1):
        ya = f"yaxis{ri if ri > 1 else ''}"
        pfig.update_layout(**{ya: dict(range=[ymin, ymax])})
    pfig.update_xaxes(range=[0, CROP_US])
    html_path = OUT_DIR / "fig09_alignment_waterfall.html"
    pfig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  -> fig09_alignment_waterfall.html  (interactive)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    T0 = _time.perf_counter()
    print("=" * 72)
    print("  SCA Leakage Assessment v4 — Streamlined")
    print("=" * 72)
    print(f"  Float32:          {USE_FLOAT32}")
    print(f"  Max traces/class: {MAX_TRACES_PER_CLASS or 'unlimited'}")
    print(f"  Crop at:          {CROP_US} us  ({CROP_SAMPLES} samples)")
    print(f"  Drop if len >=:   {MAX_TRACE_US} us  ({MAX_TRACE_LEN} samples)")
    print("=" * 72)
    print_memory("Start")

    # ── S1 Data ──────────────────────────────────────────────────────────
    traces_f, vm = preprocess_traces(
        FILES, FS, CUTOFF, HPF_ORD,
        max_len=MAX_TRACE_LEN,
        max_traces=MAX_TRACES_PER_CLASS,
        use_float32=USE_FLOAT32,
        crop_len=CROP_SAMPLES,
    )
    X, y = build_dataset(traces_f, CLASS_NAMES)
    print_memory("After building dataset")

    ns = X.shape[1]; mv = int(vm.sum())
    print(f"\nDataset: X={X.shape}, dtype={X.dtype}, valid={mv}/{ns}")
    print(f"Time axis: 0 - {ns/FS*1e6:.2f} us  "
          f"(valid: 0 - {mv/FS*1e6:.2f} us)")
    print(f"Counts: { {c: int((y==i).sum()) for i,c in enumerate(CLASS_NAMES)} }\n")

    # ── S2a Normality ────────────────────────────────────────────────────
    _sec("S2a  Normality assessment (Shapiro-Wilk + D'Agostino + KS)")
    nr = test_normality_per_sample(X, y, CLASS_NAMES)
    for c, r in nr.items():
        print(f"  {c}: SW reject={r['shapiro_reject_frac']:.0%}  "
              f"DA reject={r['dagostino_reject_frac']:.0%}  "
              f"KS reject={r['ks_reject_frac']:.0%}")

    is_gaussian = all(
        r['shapiro_reject_frac'] < 0.20 and r['ks_reject_frac'] < 0.20
        for r in nr.values()
    )
    if is_gaussian:
        print("  -> Distributions appear approximately Gaussian.")
    else:
        print("  -> Non-Gaussian distributions detected — "
              "non-parametric tests preferred.\n")

    # ── S2b Welch's t-test ───────────────────────────────────────────────
    _sec("S2b  Welch's t-test (pairwise)")
    alpha_bonf = ALPHA_FAM / mv
    print(f"  alpha={ALPHA_FAM}, m={mv}, Bonferroni per-test={alpha_bonf:.2e}")
    for ci, cj in combinations(CLASS_NAMES, 2):
        _, pv = welch_t_per_sample(traces_f[ci], traces_f[cj])
        n_sig = (pv[vm] < alpha_bonf).sum()
        print(f"  {ci}-{cj}: {n_sig}/{mv} significant (Bonferroni)")
    print()

    # ── S2c KS-based POI identification ──────────────────────────────────
    _sec("S2c  KS-based POI identification (non-parametric)")
    ks_combined, pair_ks, poi_order = identify_pois_ks(
        traces_f, CLASS_NAMES, vm)
    n_top = 20
    top_pois = poi_order[:n_top]
    print(f"  Top-{n_top} POIs (samples): {top_pois.tolist()}")
    print(f"  Top-{n_top} POIs (us):      "
          f"{[round(s/FS*1e6, 4) for s in top_pois]}")
    print(f"  KS scores:                 "
          f"{ks_combined[top_pois].round(4).tolist()}\n")

    # ── S2d POI sweep ────────────────────────────────────────────────────
    _sec("S2d  POI sweep (accuracy vs #POIs)")
    sweep = poi_sweep(X, y, poi_order, CLASS_NAMES, N_FOLDS, SEED,
                      poi_counts=POI_SWEEP_RANGE)
    print()

    # ── S3 Classifiers on optimal POIs ───────────────────────────────────
    _sec("S3   Classifiers (LDA & QDA on optimal POIs)")
    cr = run_classifiers(X, y, CLASS_NAMES, poi_order, sweep, N_FOLDS, SEED)
    print("\n  Pairwise ROC-AUC:")
    ar = pairwise_auc(X, y, CLASS_NAMES, poi_order, sweep, N_FOLDS, SEED)
    print()
    force_gc()

    # ── S4 Figures ───────────────────────────────────────────────────────
    _sec("S4   Figures (9 publication-quality plots)")
    fig01_median_spread(traces_f, CLASS_NAMES, FS, vm)
    fig02_normality(X, y, CLASS_NAMES, nr)
    fig03_welch(traces_f, CLASS_NAMES, FS, vm)
    fig04_ks_pois(ks_combined, pair_ks, poi_order, CLASS_NAMES, FS, vm)
    fig05_poi_sweep(sweep)
    fig06_classifiers(y, CLASS_NAMES, cr)
    fig07_roc(ar, sweep)
    fig08_exemplar_traces(traces_f, CLASS_NAMES, FS, vm)
    fig09_alignment_waterfall(traces_f, FS, vm, focus_class="L1")

    for fig_num in plt.get_fignums():
        plt.close(fig_num)
    force_gc()

    el = _time.perf_counter() - T0
    print(f"\n{'='*72}")
    print(f"  Done in {el/60:.1f} min.  "
          f"{len(list(OUT_DIR.glob('*.png')))} figures -> {OUT_DIR.resolve()}")
    print(f"{'='*72}")
    print_memory("End")


def _sec(title):
    print("-" * 72)
    print(title)
    print("-" * 72)


if __name__ == "__main__":
    main()
