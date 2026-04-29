# SnoopyPower — Analysis Pipeline

This directory holds the host-side analysis pipeline for SnoopyPower.
Given the L1 / L2 / DRAM trace CSVs produced by the on-board firmware
(`firmware/myapp sca`), it runs:

1. **Preprocessing** — high-pass filter, valid-region masking, crop to a
   user-defined window in microseconds
2. **Normality assessment** — Shapiro-Wilk, D'Agostino, Kolmogorov-Smirnov
3. **Welch's t-test (TVLA)** — pairwise leakage between classes, with
   Bonferroni correction
4. **KS-based POI selection** — non-parametric, distribution-free
5. **POI sweep** — accuracy vs. number of POIs, for both LDA and QDA
6. **Classification** — LDA / QDA on the optimal POI set, with stratified
   k-fold CV, ROC, and confusion matrices

Everything is laid out for fully reproducible runs and publication-quality
figures (PDF + PNG @ 300 dpi).

## Files

| File | Purpose |
|---|---|
| [`OS_characterization.py`](OS_characterization.py) | Streamlined v4 pipeline (the main entry point) |
| [`analysis.py`](analysis.py) | Trace visualisation: heatmaps, mean ± σ overlays, multi-pattern comparison |
| [`train_qda_lda.py`](train_qda_lda.py) | Cross-validated QDA / LDA training on cropped POI features |
| [`run_characterization.sh`](run_characterization.sh) | End-to-end orchestration over SSH (calibrate → collect → merge → analyse) |
| [`utils.py`](utils.py), [`config.py`](config.py) | Shared data loaders, byte-offset CSV indexing, configuration |
| [`requirements.txt`](requirements.txt) | Python dependencies for the host side |

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# Configure the SSH target (no defaults — you must set this)
export SNOOPYPOWER_HOST=pynq@<board-ip>
export SNOOPYPOWER_REMOTE_DIR=/home/pynq/SnoopyPower/firmware

# End-to-end: calibrate, collect, merge, analyse
./run_characterization.sh

# Or, if you already have CSVs:
./run_characterization.sh --skip-collect --tdc "0xFINE,0xCOARSE"
```

The pipeline writes:
- merged class CSVs to `traces/{l1,l2,DRAM}_traces.csv`
- the locked TDC calibration to `results/tdc_calibration.txt`
- figures to `figures/`

## Locking the TDC across runs

The first run calibrates the TDC delay line. Every subsequent run **must
reuse the same taps** to keep the dataset distribution consistent:

```bash
export SNOOPYPOWER_TDC_DELAY="0x00006006,0x00000303"
sudo -E ./myapp sca --pattern 1 --iters 5000 --mode memint
```

`run_characterization.sh` does this automatically once the calibration
sweep has converged.

## Data layout

Each CSV holds one TDC trace per row, comma-separated integer weights
(popcount of the thermometer-coded TDC word). The default sampling rate is
**250 MHz** (4 ns per sample); the analysis crops to **5.0 µs** windows by
default. Override via `OS_characterization.py`'s constants at the top of
the file or by supplying environment variables.

## Reproducibility

- All random seeds are configurable
- Configuration is captured alongside the figures
- The `--tdc` flag re-injects a known calibration so two runs on the same
  board produce comparable distributions
- Pattern order is randomised per cycle inside `run_characterization.sh`
  to break the systematic thermal confound that would otherwise correlate
  with the class label
