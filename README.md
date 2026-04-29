<p align="center">
  <img src="media/images/SnoopyPower.png" alt="SnoopyPower" width="320"/>
</p>

<h1 align="center">SnoopyPower</h1>

<p align="center">
  <em>An on-chip TDC power side-channel sensor for Linux-running Cortex-A9 SoCs</em><br/>
  <em>OS-platform cache-state characterization (L1 / L2 / DRAM) — fully reproducible.</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#patterns">Patterns</a> ·
  <a href="#analysis">Analysis</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#citation">Citation</a> ·
  <a href="#license">License</a>
</p>

---

## Overview

**SnoopyPower** characterizes microarchitectural side-channel leakage on a
**Xilinx Zynq-7000** (PynqZ1 / Zybo Z7-20) SoC running a stock Linux/PYNQ
userland, using an **on-chip Time-to-Digital Converter (TDC)** sensor
implemented in FPGA fabric.

The TDC shares the Power Distribution Network (PDN) with the dual ARM
Cortex-A9 cores. Voltage fluctuations induced by cache and memory events
modulate the propagation delay along the TDC's calibrated delay line,
producing a thermometer-coded digital word at **250 MHz**. SnoopyPower drives
the TDC + a hardware FIFO entirely from Linux userspace via `/dev/mem` MMIO —
no kernel module, no bare-metal flash, no JTAG.

This release ships **only the OS-platform characterization pipeline** —
specifically, deterministic L1 hit / L2 hit / DRAM miss measurement campaigns
and the publication-quality statistical analysis that turns those traces into
classifiers and leakage assessments.

<p align="center">
  <img src="media/images/ArchitectureZynq7000TDC.svg" alt="Architecture — Zynq-7000 with embedded TDC sensor" width="820"/>
</p>
<p align="center"><em>System architecture. The TDC sensor shares the Zynq-7000 PDN with the
Cortex-A9 cores, capturing voltage fluctuations induced by cache and memory
events through the FPGA fabric.</em></p>

### Key features

- **On-chip TDC power sensor** — 250 MHz sampling, sub-nanosecond resolution, no physical probe required
- **Deterministic cache-state control** — userspace eviction-based patterns for L1 hit, L2 hit, and DRAM miss, all funnelling into the same assembly measurement window
- **OS-native execution** — runs entirely from Linux userspace via `/dev/mem` MMIO; no kernel module
- **Strong dataset generation** — random target addresses and values from `/dev/urandom` via a 2 MiB anonymous `mmap()` arena, eliminating deterministic bias
- **FIFO-triggered measurement** — hardware FIFO synchronizes the TDC capture window with the memory access under test for cycle-accurate alignment
- **Auto-calibrating TDC** — per-channel delay-line tuning with polarity-aware sweep, plus a `SNOOPYPOWER_TDC_DELAY` env var to lock calibration across runs
- **Reproducible analysis pipeline** — Python scripts for HPF preprocessing, normality assessment, Welch's t-test (TVLA), KS-based POI selection, POI sweep, LDA / QDA classification, and ROC / effect-size figures

---

## Built upon SCAbox

SnoopyPower is **built upon [SCAbox](https://gitlab.emse.fr/securityinhardware/SCAbox)**,
the open framework for FPGA-based remote side-channel analysis developed at
École des Mines de Saint-Étienne (EMSE). The TDC bank and FIFO IP cores
under [`hw/ip_repo/`](hw/ip_repo/) preserve their original SCAbox vendor
namespace (`emse.sas:sca:tdc_bank`, `emse.sas:sca:fifo_and_ctrl`) inside their
`component.xml` so the Vivado IP catalog still recognises them.

If you use SnoopyPower in academic work, please cite SCAbox alongside this
project — see [Citation](#citation).

---

<a id="quickstart"></a>
## Quickstart

### Prerequisites

| Item | Details |
|---|---|
| **Board** | PynqZ1 or Zybo Z7-20 (Zynq-7000 with dual Cortex-A9) |
| **OS** | PYNQ image, or any Linux on the board with `/dev/mem` access |
| **Bitstream** | [`hw/snoopypower.bit`](hw/) (load before running) |
| **Toolchain** | native `gcc` on the board (or an ARM cross-compiler) |
| **Python (host)** | ≥ 3.8 with `numpy scipy scikit-learn matplotlib plotly` (see [`notebooks/requirements.txt`](notebooks/requirements.txt)) |

### 1. Load the bitstream on the board

```python
# From PYNQ Python
from pynq import Overlay
ol = Overlay("snoopypower.bit")    # snoopypower.hwh must be alongside
```

…or via the FPGA manager from a shell:

```bash
sudo cp hw/snoopypower.bit /lib/firmware/
echo snoopypower.bit | sudo tee /sys/class/fpga_manager/fpga0/firmware
```

### 2. Build the userspace application (on the board)

```bash
cd firmware
make clean && make
```

This produces `firmware/myapp`. The Makefile enables two compile-time flags:

| Flag | Enables |
|---|---|
| `SNOOPYPOWER_TDC` | TDC sensor driver, calibration, and channel readout |
| `SNOOPYPOWER_MEMORY` | Cache-state benchmark patterns + `/dev/urandom` arena |

### 3. Run a characterization campaign

```bash
# 5000 traces of an L1 cache hit (random addr + value per iteration)
sudo ./myapp sca --pattern 1 --iters 5000 --mode memint

# L2 hit (L1 miss)
sudo ./myapp sca --pattern 2 --iters 5000 --mode memint

# DRAM miss (full cache miss)
sudo ./myapp sca --pattern 3 --iters 5000 --mode memint
```

Traces are written to `firmware/traces/all_traces.csv` (one trace per row,
comma-separated TDC weights). The file is truncated at the start of each run.

### 4. Lock the TDC across runs (recommended)

The first run calibrates the TDC delay line. To keep the same calibration for
every subsequent run (so a classifier learns cache-state, not calibration
drift), copy the values printed at the end of calibration and re-export them:

```bash
export SNOOPYPOWER_TDC_DELAY="0x00006006,0x00000303"
sudo -E ./myapp sca --pattern 1 --iters 5000 --mode memint
```

### 5. End-to-end pipeline (host side)

The orchestration script in [`notebooks/run_characterization.sh`](notebooks/run_characterization.sh)
SSHes into the board, calibrates once, collects shuffled L1 / L2 / DRAM
traces with thermal settling, scp's the CSVs back, merges them, and runs
[`OS_characterization.py`](notebooks/OS_characterization.py):

```bash
# Configure the SSH target (no defaults — you must set this)
export SNOOPYPOWER_HOST=pynq@<board-ip>
export SNOOPYPOWER_REMOTE_DIR=/home/pynq/SnoopyPower/firmware

cd notebooks
./run_characterization.sh                          # full pipeline
./run_characterization.sh --skip-collect           # reuse existing CSVs
./run_characterization.sh --tdc "0x...,0x..."      # reuse TDC calibration
./run_characterization.sh --profile unpriv         # patterns 14/15/16
```

The pipeline writes traces to `notebooks/traces/`, figures to
`notebooks/figures/`, and the locked calibration to
`notebooks/results/tdc_calibration.txt`.

---

<a id="architecture"></a>
## Architecture

### Hardware

The TDC sensor is instantiated in the Zynq PL (Programmable Logic) and
exposed to the PS (Processing System) via AXI-Lite. Two IP cores derived
from SCAbox are mapped into the design:

| IP Core | Role | AXI-Lite handle |
|---|---|---|
| **FIFO & Controller** | Triggers and buffers TDC samples; synchronises the measurement window with the CPU memory access | `XPAR_FIFO_AND_CTRL_0` |
| **TDC Bank** | Multi-channel delay-line sensor; converts PDN voltage variations into a thermometer-coded digital word at 250 MHz | `XPAR_TDC_BANK_0` |

The FIFO write-enable signal brackets the assembly measurement window
(`streaming_triggered_ldrb`), so TDC samples are captured only during the
critical `ldrb` access — not before, not after.

### Software stack

```
┌───────────────────────────────────────────────────────────────┐
│  main_linux.c            CLI dispatcher (fifo / tdc / sca /   │
│                          selftest)                            │
├───────────────────────────────────────────────────────────────┤
│  membench_patterns.c     Cache-state setup (eviction-based)   │
│  membench_core.c         Timed load + FIFO callback           │
│  membench_addr.c         Address generators (DDR / cache-set) │
│  membench_prng.c         Deterministic LCG (legacy)           │
│  membench_rand.c         /dev/urandom arena (2 MiB mmap)      │
├───────────────────────────────────────────────────────────────┤
│  xfifo.c / xtdc.c        Low-level IP drivers                 │
│  mmio_linux.c            /dev/mem mmap() for AXI registers    │
└───────────────────────────────────────────────────────────────┘
```

All cache maintenance is performed via **eviction buffers** (64 KB for L1,
1 MB for L1+L2). This works from unprivileged Linux userspace — no kernel
module, no CP15 access, no `mlock`.

---

<a id="patterns"></a>
## Measurement patterns

Each pattern sets up a specific cache-hierarchy state, then executes the
**exact same** assembly measurement snippet. The TDC captures the PDN
signature of that single `ldrb` instruction.

Patterns 1 / 2 / 3 use **random target addresses** drawn from a 2 MiB
anonymous `mmap()` arena seeded by `/dev/urandom`, and write a **random byte
value** to DRAM before each measurement. This eliminates deterministic
address bias and produces strong datasets for classification and leakage
analysis.

| ID | Name | Pre-state setup | Expected source |
|---|---|---|---|
| `1` | **L1 Hit**     | random addr + value → touch target → `ldrb`                       | L1D read port |
| `2` | **L2 Hit**     | random addr + value → touch target → evict L1 (64 KB walk) → `ldrb` | L2 (PL310) refill into L1 |
| `3` | **DRAM Miss**  | random addr + value → evict L1+L2 (1 MB walk) → `ldrb`             | DDR3 controller + bus |
| `14`/`15`/`16` | Same as 1/2/3 with **unprivileged probe profile** (no priv'd barriers in the timed window) | – | – |
| `10` | **noL1 (legacy)** | deterministic addr, no explicit cache prep → `ldrb` | ambient / unknown |

### Why eviction buffers?

On the Cortex-A9 running Linux, privileged cache-maintenance instructions
(`MCR p15 ...`) require kernel mode. SnoopyPower uses **contention-based
eviction** instead: reading through a buffer larger than the cache capacity
forces every set × way to be replaced. This is deterministic, portable, and
introduces no kernel dependency.

```
Pattern 2 (L2 hit):     64 KB walk → overwrites all 256 L1 sets × 4 ways
Pattern 3 (DRAM miss):  1 MB walk  → overwrites all L2 sets × 8 ways + all L1
```

### Assembly measurement window

All patterns funnel into this naked function, which centers the `ldrb` in a
timing-stable spin window:

```c
__attribute__((naked,noinline))
void streaming_triggered_ldrb(const uint8_t *base)
{
    __asm__ __volatile__ (
        "eor    r3, r3, r3       \n"
        "add    r3, r0, r3       \n"
        "mov    r1, #100         \n"
        "1: subs r1, r1, #1      \n"
        "   bne  1b              \n"
        "dsb    sy               \n"
        "isb                     \n"
        "ldrb   r2, [r3]         \n"   /* ← THE access under test */
        "dsb    sy               \n"
        "mov    r1, #100         \n"
        "2: subs r1, r1, #1      \n"
        "   bne  2b              \n"
        "dsb    sy               \n"
        "isb                     \n"
        "bx     lr               \n"
    );
}
```

A second variant (`streaming_triggered_ldrb_unpriv`) is used for the
unprivileged probe profile.

---

<a id="analysis"></a>
## Analysis pipeline

The [`notebooks/`](notebooks/) directory contains the full statistical
analysis suite:

| Script | Purpose |
|---|---|
| [`OS_characterization.py`](notebooks/OS_characterization.py) | Streamlined v4 pipeline: HPF preprocessing, normality assessment (Shapiro-Wilk, D'Agostino, KS), Welch's t-test, KS-based POI identification, POI sweep (LDA/QDA), publication figures |
| [`analysis.py`](notebooks/analysis.py) | Trace visualisation: heat-maps, mean ± σ overlays, multi-pattern comparison |
| [`train_qda_lda.py`](notebooks/train_qda_lda.py) | Cross-validated QDA/LDA training on cropped POI features |
| [`run_characterization.sh`](notebooks/run_characterization.sh) | End-to-end orchestration (calibrate → collect → merge → analyse) |
| [`utils.py`](notebooks/utils.py), [`config.py`](notebooks/config.py) | Shared data loaders, byte-offset CSV indexing, configuration |

### Generated figures

All figures are produced as both PDF (vector, camera-ready) and PNG (300 dpi):

| Figure | Description |
|---|---|
| `fig_mean_traces`         | Mean TDC weight over time for each pattern with ±1σ shading |
| `fig_normality`           | Q-Q plots and Shapiro-Wilk / D'Agostino / KS tests |
| `fig_welch_ttest`         | Welch's t-test (TVLA) between pattern pairs with Bonferroni-corrected threshold |
| `fig_poi_sweep`           | LDA & QDA accuracy as a function of the number of points-of-interest |
| `fig_lda_scatter_cm`      | LDA projection of trace features with confusion matrix |
| `fig_roc`                 | Receiver operating characteristic for each pair of classes |
| `fig_effect_sizes`        | Cohen's d at every time sample |
| `fig_alignment_waterfall` | Interactive (HTML) waterfall of aligned traces |

---

## CLI reference

```
Usage:
  sudo ./myapp <subcommand> [options]

Subcommands:
  fifo       FIFO operations (SW mode)
  tdc        TDC operations (calibration, delay, state)
  sca        Side-channel cache-state characterization
  selftest   Hardware self-test
```

### `sca` subcommand

| Flag | Description | Default |
|---|---|---|
| `--pattern N` | Pattern ID: 1 (L1), 2 (L2), 3 (DRAM), 14/15/16 (unpriv variants), 10 (legacy) | *required* |
| `--iters N`   | Number of traces to collect | 1 |
| `--mode STR`  | Execution mode | `memint` |
| `--addr 0xADDR` | Direct virtual address (pattern 10 only) | auto-selected |
| `--hit-idx N` | Cache-set / table index selector (pattern 10) | 0 |
| `--start N`   | FIFO read window start | 0 |
| `--end N`     | FIFO read window end | depth-1 |
| `--raw`       | Emit raw FIFO markers (disables progress bar) | off |
| `--verbose`   | Verbose console output | off |

### `tdc` subcommand

| Flag | Description |
|---|---|
| `--calibrate K`         | Run auto-calibration with K iterations |
| `--avg CH --iters K`    | Average TDC weight and polarity for channel CH |
| `--state CH --reads R`  | Dump R raw STATE readings for channel CH |
| `--set-all F C`         | Set all channels to fine=F, coarse=C |
| `--set CH F C`          | Set one channel |
| `--get CH`              | Read channel delay (CH = -1 → raw registers) |
| `--info`                | Print TDC configuration summary |

### Environment variables

| Variable | Purpose |
|---|---|
| `SNOOPYPOWER_TDC_DELAY="FINE,COARSE"` | Skip the calibration sweep and write known taps directly. Required for cross-run dataset consistency. |

---

## Hardware requirements

| Resource | Specification |
|---|---|
| **SoC**       | Xilinx Zynq-7000 (XC7Z020) |
| **CPU**       | Dual-core ARM Cortex-A9 @ 667 MHz |
| **L1 D-Cache**| 32 KB, 4-way set-associative, 32 B lines, 256 sets |
| **L2 Cache**  | 512 KB (PL310), 8-way set-associative, 32 B lines |
| **DDR**       | 512 MB DDR3 |
| **TDC**       | On-chip, 250 MHz sampling, multi-channel |
| **Board**     | PynqZ1 or Zybo Z7-20 |

---

<a id="repository-layout"></a>
## Repository layout

```
SnoopyPowerOS/
├── hw/                              # FPGA design files
│   ├── SnoopyPower.xpr              # Vivado project
│   ├── snoopypower.bit              # Pre-built bitstream (load directly)
│   ├── snoopypower.hwh              # Hardware handoff for PYNQ overlay
│   ├── snoopypower.xsa              # Optional Vitis export
│   └── ip_repo/                     # SCAbox-derived custom IP cores
│
├── firmware/                        # Userspace measurement application
│   ├── main_linux.c                 # CLI entry point + HW init
│   ├── mmio_linux.c / .h            # /dev/mem AXI mapping
│   ├── shared_region.c              # Shared memory binding
│   ├── Makefile
│   ├── benches/                     # Measurement patterns
│   ├── drivers/                     # IP core drivers (xtdc, xfifo)
│   ├── include/                     # Shared headers
│   ├── utils/                       # PMU helpers
│   └── cache/                       # Linux-side cache-flush wrapper
│
├── notebooks/                       # Analysis & visualisation
│   ├── OS_characterization.py       # Statistical pipeline (v4)
│   ├── analysis.py                  # Trace plotting
│   ├── train_qda_lda.py             # QDA/LDA training
│   ├── run_characterization.sh      # End-to-end orchestrator
│   ├── utils.py, config.py
│   ├── requirements.txt
│   └── README.md
│
├── tools/                           # Companion utilities
│   ├── merge_dataset.c              # CSV concatenator (sorted)
│   └── Makefile
│
├── media/images/                    # Documentation assets
├── CITATION.cff
├── LICENSE
└── README.md
```

---

<a id="citation"></a>
## Citation

If you use SnoopyPower in academic work, please cite this repository
*and* the SCAbox papers that describe the TDC and RO sensors it builds upon:

> Joseph Gravellier, Jean-Max Dutertre, Yannick Teglia, Philippe Loubet-Moundi,
> Olivier Francis. **Remote Side-Channel Attacks on Heterogeneous SoC.**
> *Smart Card Research and Advanced Applications, 18th International
> Conference, CARDIS 2019*, Nov 2019, Prague, Czech Republic.

> Joseph Gravellier, Jean-Max Dutertre, Yannick Teglia, Philippe Loubet-Moundi.
> **High-Speed Ring Oscillator based Sensors for Remote Side-Channel Attacks
> on FPGAs.** *2019 International Conference on ReConFigurable Computing and
> FPGAs (ReConFig)*.

A `CITATION.cff` file is provided at the root of the repository; GitHub
will surface the citation automatically.

```bibtex
@misc{snoopypower2026,
  title  = {SnoopyPower: An on-chip TDC power side-channel sensor for
            Linux-running Cortex-A9 SoCs},
  author = {Quéré, Eliott},
  year   = {2026},
  note   = {Built upon SCAbox (École des Mines de Saint-Étienne).}
}

@inproceedings{gravellier2019remote,
  title     = {Remote Side-Channel Attacks on Heterogeneous SoC},
  author    = {Gravellier, Joseph and Dutertre, Jean-Max and Teglia, Yannick
               and Loubet-Moundi, Philippe and Francis, Olivier},
  booktitle = {Smart Card Research and Advanced Applications,
               18th International Conference, CARDIS 2019},
  year      = {2019},
  month     = nov,
  address   = {Prague, Czech Republic}
}

@inproceedings{gravellier2019highspeed,
  title     = {High-Speed Ring Oscillator based Sensors for Remote
               Side-Channel Attacks on FPGAs},
  author    = {Gravellier, Joseph and Dutertre, Jean-Max and Teglia, Yannick
               and Loubet-Moundi, Philippe},
  booktitle = {2019 International Conference on ReConFigurable Computing
               and FPGAs (ReConFig)},
  year      = {2019}
}
```

---

<a id="license"></a>
## License

SnoopyPower is released under the **MIT License** — see [`LICENSE`](LICENSE).

Copyright © 2026 Eliott Quéré.

The TDC and FIFO IP cores under [`hw/ip_repo/`](hw/ip_repo/) are derived
from [SCAbox](https://gitlab.emse.fr/securityinhardware/SCAbox) (EMSE,
École des Mines de Saint-Étienne). Refer to the upstream project for the
license that applies to those individual components.
# SnoopyPower
