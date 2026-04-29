#!/usr/bin/env python3
"""
analysis.py — Publication-quality TDC trace visualisation
for SnoopyPower / Zynq-7000 Cortex-A9 cache side-channel experiments.

Reads  : traces/all_traces.csv   (one trace per row, comma-separated TDC weights)
Outputs: traces/figure_traces.pdf  (vector, camera-ready)
         traces/figure_traces.png  (300 dpi raster fallback)

Usage:
    python3 plot_traces.py
    python3 plot_traces.py --csv traces/all_traces.csv --pattern 2 --hit-index 3850
    python3 plot_traces.py --csv l1.csv l2.csv dram.csv --pattern 1 2 3 \
        --label "L1 Hit" "L2 Hit" "DRAM Miss" --hit-index 3850

Sampling: 250 MHz TDC → 4 ns per sample.
HPF     : Butterworth high-pass (default: ON), fc=3 MHz, order=1, zero-phase filtfilt.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe, must come before pyplot import
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import patheffects
import matplotlib.gridspec as gridspec
from scipy import signal

# ─── Global style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    # — fonts —
    "font.family":        "serif",
    "font.serif":         ["CMU Serif", "Latin Modern Roman",
                           "DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset":   "cm",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.title_fontsize": 8.5,
    # — layout —
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    # — axes —
    "axes.linewidth":     0.6,
    "axes.grid":          True,
    "grid.linewidth":     0.35,
    "grid.alpha":         0.30,
    "grid.linestyle":     "-",
    # — ticks —
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.minor.width":  0.3,
    "ytick.minor.width":  0.3,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.size":   2.0,
    "ytick.minor.size":   2.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    # — lines —
    "lines.linewidth":    1.0,
    # — legend —
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.70",
    "legend.fancybox":    False,
})

# ─── Colour palette ─────────────────────────────────────────────────────────
PALETTE = {
    1:  "#0077BB",   # L1 hit   — blue
    2:  "#EE7733",   # L2 hit   — orange
    3:  "#CC3311",   # DRAM miss — red
    10: "#009988",   # legacy    — teal
}
PALETTE_LIGHT = {
    1:  "#0077BB20",
    2:  "#EE773320",
    3:  "#CC331120",
    10: "#00998820",
}
FALLBACK_COLOR       = "#332288"
FALLBACK_COLOR_LIGHT = "#33228820"

PATTERN_LABEL = {
    1:  "Pattern 1 — L1 Hit",
    2:  "Pattern 2 — L2 Hit (L1 Miss)",
    3:  "Pattern 3 — DRAM Miss",
    10: "Pattern 10 — Legacy (no L1)",
}

# ─── Filtering ──────────────────────────────────────────────────────────────

def design_hpf(order: int, cutoff_hz: float, fs_hz: float):
    """Butterworth HPF -> (b, a). Uses SciPy 'fs' kw if available, else normalized Wn."""
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be > 0")
    if cutoff_hz >= fs_hz / 2:
        raise ValueError("cutoff_hz must be < fs/2")
    try:
        b, a = signal.butter(order, cutoff_hz, btype="highpass", fs=fs_hz)
    except TypeError:
        wn = cutoff_hz / (fs_hz / 2.0)
        b, a = signal.butter(order, wn, btype="highpass")
    return b, a

def apply_hpf(traces: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Zero-phase HPF on each trace with filtfilt.
    Handles NaN padding by filtering only the valid prefix per row.
    """
    out = traces.copy()

    # Fast path: no NaNs anywhere -> vectorized filtering along axis 1
    if not np.isnan(out).any():
        return signal.filtfilt(b, a, out, axis=1)

    # Safe path: per-row filtering ignoring NaN tail
    for i in range(out.shape[0]):
        row = out[i]
        m = ~np.isnan(row)
        if not m.any():
            continue
        x = row[m]
        # filtfilt needs a few samples; for 1st order this is trivial, but keep guard
        if x.size < 4:
            out[i, m] = x - np.mean(x)
            continue
        out[i, m] = signal.filtfilt(b, a, x)
    return out

# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_traces(path: str) -> np.ndarray:
    """Load CSV: one trace per row, return (N_traces, N_samples) float64 array."""
    if not os.path.isfile(path):
        sys.exit(f"Error: file not found: {path}")

    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                vals = [float(v) for v in line.split(",")]
                rows.append(vals)
            except ValueError:
                print(f"  [warn] skipping malformed line {lineno}", file=sys.stderr)

    if not rows:
        sys.exit(f"Error: no valid traces in {path}")

    # Pad to equal length (in case some rows are shorter)
    maxlen = max(len(r) for r in rows)
    arr = np.full((len(rows), maxlen), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r

    print(f"  Loaded {arr.shape[0]} traces × {arr.shape[1]} samples from {path}")
    return arr

# ─── Plot helpers ────────────────────────────────────────────────────────────

def maybe_draw_hit(ax, hit_x, label=None):
    if hit_x is None:
        return
    ax.axvline(hit_x, linestyle="--", linewidth=0.8, color="0.25", label=label)

# ─── Single-dataset figure (3 subplots) ──────────────────────────────────────

def plot_single(traces: np.ndarray, label: str, pattern_id: int,
                fs_mhz: float, outdir: str,
                hpf_on: bool, hpf_cutoff_hz: float, hpf_order: int,
                hit_index: int | None):
    """
    Three-panel figure for one pattern:
      (a) Heatmap of all traces
      (b) Overlay of individual traces (translucent) + mean
      (c) Mean ± std with shaded confidence band
    """
    N, S   = traces.shape
    dt_ns  = 1e3 / fs_mhz                 # 4.0 ns for 250 MHz
    t_ns   = np.arange(S) * dt_ns
    t_us   = t_ns / 1e3                   # µs axis for wide traces

    mu     = np.nanmean(traces, axis=0)
    sigma  = np.nanstd(traces, axis=0)

    color      = PALETTE.get(pattern_id, FALLBACK_COLOR)
    color_fill = PALETTE_LIGHT.get(pattern_id, FALLBACK_COLOR_LIGHT)

    # Decide time unit
    use_us = t_ns[-1] > 5000
    t_axis = t_us if use_us else t_ns
    t_unit = "µs" if use_us else "ns"

    hit_x = None
    if hit_index is not None and 0 <= hit_index < S:
        hit_x = t_axis[hit_index]

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.0, 7.0))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.2, 1.0],
                            hspace=0.32)

    # ── (a) Heatmap ─────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    extent = [t_axis[0], t_axis[-1], N - 0.5, -0.5]
    im = ax0.imshow(traces, aspect="auto", interpolation="none",
                    cmap="inferno", extent=extent, origin="upper")

    cb = fig.colorbar(im, ax=ax0, pad=0.015, aspect=30, shrink=0.92)
    cb.set_label("TDC weight", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax0.set_ylabel("Trace index")
    ax0.set_xlabel(f"Time ({t_unit})")
    ax0.set_title(f"(a)  Trace heatmap — {label}  ($n={N}$)",
                  fontweight="medium", loc="left")

    maybe_draw_hit(ax0, hit_x, label="Cache access" if hit_x is not None else None)
    if hit_x is not None:
        ax0.legend(loc="upper right", frameon=True)

    # ── (b) Overlay + mean ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # plot at most 200 individual traces for readability
    n_show = min(N, 200)
    alpha  = max(0.02, min(0.15, 20.0 / n_show))
    for i in range(n_show):
        ax1.plot(t_axis, traces[i], color=color, alpha=alpha,
                 linewidth=0.35, rasterized=True)

    ax1.plot(t_axis, mu, color=color, linewidth=1.5, zorder=10,
             label="Mean",
             path_effects=[patheffects.withStroke(linewidth=2.6,
                                                  foreground="white")])

    maybe_draw_hit(ax1, hit_x)
    ax1.set_ylabel("TDC weight")
    ax1.set_xlabel(f"Time ({t_unit})")
    ax1.set_title(f"(b)  Individual traces + mean — {label}",
                  fontweight="medium", loc="left")
    ax1.legend(loc="upper right", frameon=True)

    # ── (c) Mean ± σ ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    ax2.fill_between(t_axis, mu - sigma, mu + sigma,
                     color=color, alpha=0.18, linewidth=0, label=r"$\pm 1\sigma$")
    ax2.fill_between(t_axis, mu - 2*sigma, mu + 2*sigma,
                     color=color, alpha=0.08, linewidth=0, label=r"$\pm 2\sigma$")
    ax2.plot(t_axis, mu, color=color, linewidth=1.3, label="Mean",
             path_effects=[patheffects.withStroke(linewidth=2.2,
                                                  foreground="white")])

    maybe_draw_hit(ax2, hit_x)
    ax2.set_ylabel("TDC weight")
    ax2.set_xlabel(f"Time ({t_unit})")
    ax2.set_title(f"(c)  Mean ± standard deviation — {label}",
                  fontweight="medium", loc="left")
    ax2.legend(loc="upper right", ncol=3, frameon=True)

    # ── Minor ticks ─────────────────────────────────────────────────────
    for ax in (ax0, ax1, ax2):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # ── Annotation: sampling + filter info ───────────────────────────────
    dt_str = f"$\\Delta t = {dt_ns:.1f}$ ns"
    fs_str = f"$f_s = {fs_mhz:.0f}$ MHz"
    base   = f"{fs_str}  |  {dt_str}  |  $n = {N}$ traces  |  $S = {S}$ samples"
    if hpf_on:
        base += f"  |  HPF: Butter({hpf_order}) @ {hpf_cutoff_hz/1e6:.1f} MHz (filtfilt)"
    fig.text(0.50, 0.005, base, ha="center", fontsize=7,
             fontstyle="italic", color="0.45")

    # ── Save ─────────────────────────────────────────────────────────────
    tag = label.lower().replace(" ", "_").replace("/", "_")
    stem = os.path.join(outdir, f"figure_{tag}")

    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png")
    print(f"  Saved {stem}.pdf  and  {stem}.png")
    plt.close(fig)

# ─── Multi-dataset comparison (overlay of means) ─────────────────────────────

def plot_comparison(datasets: list, labels: list, pattern_ids: list,
                    fs_mhz: float, outdir: str,
                    hpf_on: bool, hpf_cutoff_hz: float, hpf_order: int,
                    hit_index: int | None):
    """
    Two-panel comparison figure:
      (a) Overlaid means
      (b) Overlaid σ (noise profile)
    """
    dt_ns = 1e3 / fs_mhz
    max_s = max(d.shape[1] for d in datasets)
    t_ns  = np.arange(max_s) * dt_ns
    t_us  = t_ns / 1e3
    use_us = t_ns[-1] > 5000
    t_axis = t_us if use_us else t_ns
    t_unit = "µs" if use_us else "ns"

    hit_x = None
    if hit_index is not None and 0 <= hit_index < max_s:
        hit_x = t_axis[hit_index]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 4.6),
                                    sharex=True,
                                    gridspec_kw={"hspace": 0.30})

    for traces, label, pid in zip(datasets, labels, pattern_ids):
        S     = traces.shape[1]
        t_loc = t_axis[:S]
        mu    = np.nanmean(traces, axis=0)
        sigma = np.nanstd(traces, axis=0)
        c     = PALETTE.get(pid, FALLBACK_COLOR)

        ax0.plot(t_loc, mu, color=c, linewidth=1.2, label=label,
                 path_effects=[patheffects.withStroke(linewidth=2.0,
                                                     foreground="white")])
        ax0.fill_between(t_loc, mu - sigma, mu + sigma,
                         color=c, alpha=0.12, linewidth=0)

        ax1.plot(t_loc, sigma, color=c, linewidth=1.0, label=label)

    maybe_draw_hit(ax0, hit_x, label="Cache access" if hit_x is not None else None)
    maybe_draw_hit(ax1, hit_x)

    ax0.set_ylabel("Mean TDC weight")
    title0 = "(a)  Mean traces — pattern comparison"
    if hpf_on:
        title0 += f"  (HPF {hpf_cutoff_hz/1e6:.1f} MHz)"
    ax0.set_title(title0, fontweight="medium", loc="left")
    ax0.legend(loc="upper right", frameon=True)

    ax1.set_ylabel(r"$\sigma$ (TDC weight)")
    ax1.set_xlabel(f"Time ({t_unit})")
    ax1.set_title("(b)  Noise profile ($\\sigma$) — pattern comparison",
                  fontweight="medium", loc="left")
    ax1.legend(loc="upper right", frameon=True)

    for ax in (ax0, ax1):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # footer
    footer = f"$f_s = {fs_mhz:.0f}$ MHz  |  $\\Delta t = {dt_ns:.1f}$ ns"
    if hpf_on:
        footer += f"  |  HPF: Butter({hpf_order}) @ {hpf_cutoff_hz/1e6:.1f} MHz (filtfilt)"
    fig.text(0.50, 0.01, footer, ha="center", fontsize=7,
             fontstyle="italic", color="0.45")

    stem = os.path.join(outdir, "figure_comparison")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png")
    print(f"  Saved {stem}.pdf  and  {stem}.png")
    plt.close(fig)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot SnoopyPower TDC traces (publication quality).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  # Single pattern (default HPF @ 3 MHz)
  python3 plot_traces.py --csv traces/all_traces.csv --pattern 2 --hit-index 3850

  # Compare three patterns (one CSV each)
  python3 plot_traces.py \\
      --csv traces/l1.csv traces/l2.csv traces/dram.csv \\
      --label "L1 Hit" "L2 Hit" "DRAM Miss" \\
      --pattern 1 2 3 \\
      --hit-index 3850

  # Disable HPF
  python3 plot_traces.py --csv traces/all_traces.csv --no-hpf
""")

    ap.add_argument("--csv", nargs="+",
                    default=["traces/all_traces.csv"],
                    help="Path(s) to CSV trace file(s).")
    ap.add_argument("--label", nargs="+",
                    default=None,
                    help="Human-readable label(s) for each CSV.")
    ap.add_argument("--pattern", nargs="+", type=int,
                    default=None,
                    help="Pattern ID(s) — controls colour. (1=L1, 2=L2, 3=DRAM, 10=legacy)")
    ap.add_argument("--fs", type=float, default=250.0,
                    help="TDC sampling frequency in MHz (default: 250).")
    ap.add_argument("--outdir", default="traces",
                    help="Output directory for figures (default: traces/).")
    ap.add_argument("--no-individual", action="store_true",
                    help="Skip per-dataset 3-panel figures; only produce comparison.")

    # HPF settings (default: ON @ 3 MHz, order 1) — matches your training snippet
    ap.add_argument("--no-hpf", action="store_true",
                    help="Disable high-pass filtering.")
    ap.add_argument("--hpf-cutoff", type=float, default=3e6,
                    help="High-pass cutoff frequency in Hz (default: 3e6).")
    ap.add_argument("--hpf-order", type=int, default=1,
                    help="Butterworth HPF order (default: 1).")

    # Cache hit index marker (optional)
    ap.add_argument("--hit-index", type=int, default=3850,
                    help="Optional sample index to mark with a vertical dashed line (default: 3850). "
                         "Use --hit-index -1 to disable.")

    args = ap.parse_args()

    if args.hit_index is not None and args.hit_index < 0:
        args.hit_index = None

    n_files = len(args.csv)

    # Default pattern IDs (updated default to 2 to match your provided parameters)
    if args.pattern is None:
        args.pattern = [2] * n_files

    # Default labels
    if args.label is None:
        args.label = [PATTERN_LABEL.get(p, f"Pattern {p}") for p in args.pattern]

    # Validate lengths
    if len(args.label) != n_files:
        sys.exit(f"Error: {n_files} CSV files but {len(args.label)} labels")
    if len(args.pattern) != n_files:
        sys.exit(f"Error: {n_files} CSV files but {len(args.pattern)} pattern IDs")

    os.makedirs(args.outdir, exist_ok=True)

    fs_hz = args.fs * 1e6
    dt_ns = 1e9 / fs_hz

    hpf_on = not args.no_hpf
    if hpf_on:
        b, a = design_hpf(args.hpf_order, args.hpf_cutoff, fs_hz)

    print(f"Sampling: {args.fs:.1f} MHz → Δt = {dt_ns:.2f} ns")
    if hpf_on:
        print(f"HPF     : Butter({args.hpf_order}) high-pass @ {args.hpf_cutoff/1e6:.2f} MHz (filtfilt)")
    if args.hit_index is not None:
        print(f"Marker  : hit-index = {args.hit_index}")
    print()

    # ── Load all ──────────────────────────────────────────────────────────
    datasets = []
    for csv_path in args.csv:
        tr = load_traces(csv_path)
        if hpf_on:
            tr = apply_hpf(tr, b, a)
        datasets.append(tr)

    # ── Per-dataset figures ───────────────────────────────────────────────
    if not args.no_individual:
        for traces, label, pid in zip(datasets, args.label, args.pattern):
            plot_single(
                traces, label, pid, args.fs, args.outdir,
                hpf_on=hpf_on, hpf_cutoff_hz=args.hpf_cutoff, hpf_order=args.hpf_order,
                hit_index=args.hit_index
            )

    # ── Comparison figure (if multiple datasets) ─────────────────────────
    if n_files > 1:
        plot_comparison(
            datasets, args.label, args.pattern, args.fs, args.outdir,
            hpf_on=hpf_on, hpf_cutoff_hz=args.hpf_cutoff, hpf_order=args.hpf_order,
            hit_index=args.hit_index
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
