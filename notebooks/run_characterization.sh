#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_characterization.sh — SnoopyPower OS-level cache characterization
#
# Steps:
#   1. Calibrate TDC (once)
#   2. Collect L1 / L2 / DRAM training traces (shuffled order, locked TDC)
#   3. Merge per-pattern CSVs into l1_traces.csv, l2_traces.csv, DRAM_traces.csv
#   4. Run OS_characterization.py: HPF + crop + normality + Welch +
#      KS POIs + POI sweep + LDA/QDA classification
#
# The board-side firmware lives in firmware/ and produces traces/all_traces.csv.
# This script SSHs into the board, runs the firmware, scp's the CSV back, and
# launches the offline analysis on the host.
#
# Configuration:
#   Override the host/path with env vars or a sourced .env file:
#     SNOOPYPOWER_HOST       e.g.  pynq@10.0.0.42        (no default)
#     SNOOPYPOWER_REMOTE_DIR e.g.  /home/pynq/SnoopyPower/firmware
#
# Usage:
#   SNOOPYPOWER_HOST=user@board ./run_characterization.sh
#   ./run_characterization.sh --skip-collect          # reuse existing CSVs
#   ./run_characterization.sh --tdc "0x...,0x..."     # reuse calibration
#   ./run_characterization.sh --profile unpriv        # patterns 14/15/16
# ==============================================================================

# ─── Configuration ────────────────────────────────────────────────────────
HOST="${SNOOPYPOWER_HOST:-}"
REMOTE_DIR="${SNOOPYPOWER_REMOTE_DIR:-/home/pynq/SnoopyPower/firmware}"
REMOTE_CSV="${REMOTE_DIR}/traces/all_traces.csv"

LOCAL_TRACES="./traces"
SCRIPTS_DIR="."
RESULTS_DIR="./results"
FIGURES_DIR="./figures"

# Trace collection
REPS=1
ITERS=18000
MODE="memint"
PROFILE="priv"          # priv -> 1/2/3, unpriv -> 14/15/16
PATTERNS=(1 2 3)

declare -A PATTERN_NAMES=(
  [1]="L1" [2]="L2" [3]="DRAM"
  [14]="L1" [15]="L2" [16]="DRAM"
)

set_patterns_for_profile() {
  case "$PROFILE" in
    priv)   PATTERNS=(1 2 3) ;;
    unpriv) PATTERNS=(14 15 16) ;;
    *) echo "[ERROR] Invalid --profile '$PROFILE' (priv|unpriv)"; exit 1 ;;
  esac
}

# Preprocessing parameters
FS=250e6
HPF_CUTOFF=3e6
HPF_ORDER=1
MAX_TRACE_LEN=6000
MAX_TRACES_PER_CLASS=6000

# Statistical parameters
N_FOLDS=5
ALPHA=0.01
USE_FLOAT32=true

# Thermal settling time between patterns (seconds)
SETTLE_TIME=3

# ─── Parse Arguments ──────────────────────────────────────────────────────
SKIP_COLLECT=false
TDC_DELAY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-collect) SKIP_COLLECT=true; shift ;;
    --tdc)          TDC_DELAY="$2"; shift 2 ;;
    --iters)        ITERS="$2"; shift 2 ;;
    --reps)         REPS="$2"; shift 2 ;;
    --profile)      PROFILE="$2"; shift 2 ;;
    --max-traces)   MAX_TRACES_PER_CLASS="$2"; shift 2 ;;
    --settle)       SETTLE_TIME="$2"; shift 2 ;;
    --host)         HOST="$2"; shift 2 ;;
    --remote-dir)   REMOTE_DIR="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [OPTIONS]

  --host USER@HOST        SSH target (or set SNOOPYPOWER_HOST env var)
  --remote-dir PATH       Remote firmware directory (default: $REMOTE_DIR)
  --skip-collect          Skip steps 1-3, reuse existing CSVs
  --tdc 'FINE,COARSE'     Reuse known TDC calibration (skip sweep)
  --iters N               Traces per pattern per cycle (default: $ITERS)
  --reps N                Collection cycles (default: $REPS)
  --profile priv|unpriv   Probe profile (default: $PROFILE)
  --max-traces N          Max traces per class (default: $MAX_TRACES_PER_CLASS)
  --settle N              Seconds between patterns (default: $SETTLE_TIME)
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ -z "$HOST" ]; then
  echo "[ERROR] No SSH host configured."
  echo "  Set SNOOPYPOWER_HOST=user@host or pass --host user@host"
  exit 1
fi

set_patterns_for_profile

# ─── Helpers ──────────────────────────────────────────────────────────────
ts() { date +"%Y%m%d_%H%M%S"; }

run_ssh() {
  ssh -t "$HOST" "bash -lc 'set -euo pipefail; $1'" 2>&1
}

shuffle_array() {
  local arr=("$@")
  local i n tmp
  n=${#arr[@]}
  for ((i = n - 1; i > 0; i--)); do
    j=$((RANDOM % (i + 1)))
    tmp=${arr[$i]}; arr[$i]=${arr[$j]}; arr[$j]=$tmp
  done
  echo "${arr[@]}"
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   SNOOPYPOWER  —  OS Cache Characterization (L1 / L2 / DRAM) ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "  Host:            $HOST"
echo "  Remote dir:      $REMOTE_DIR"
echo "  Reps:            $REPS"
echo "  Iters/pattern:   $ITERS"
echo "  Profile:         $PROFILE"
echo "  Patterns:        ${PATTERNS[*]} ($(for p in "${PATTERNS[@]}"; do printf "%s " "${PATTERN_NAMES[$p]}"; done))"
echo "  HPF:             $(python3 -c "print(${HPF_CUTOFF}/1e6)") MHz"
echo "  Max trace len:   $MAX_TRACE_LEN"
echo "  Max traces/cls:  $MAX_TRACES_PER_CLASS"
echo "  Settle time:     ${SETTLE_TIME}s"
echo

# ==============================================================================
# STEP 1: TDC CALIBRATION
# ==============================================================================
if [ -z "$TDC_DELAY" ]; then
  if [ -f "$RESULTS_DIR/tdc_calibration.txt" ] && $SKIP_COLLECT; then
    TDC_DELAY=$(cat "$RESULTS_DIR/tdc_calibration.txt")
    echo "[Step 1] Reusing saved TDC calibration: $TDC_DELAY"
  else
    echo "[Step 1] Calibrating TDC..."
    mkdir -p "$RESULTS_DIR"

    echo "  [1a] Verifying TDC register access..."
    AVG_OUTPUT=$(run_ssh \
      "cd '$REMOTE_DIR' && sudo ./myapp tdc --avg 0 --iters 2048" 2>&1) || {
        echo "[ERROR] tdc --avg failed — is the bitstream loaded and PL powered?"
        echo "$AVG_OUTPUT"
        exit 1
      }

    AVG_WEIGHT=$(echo "$AVG_OUTPUT" | sed -nE 's/.*AVG_WEIGHT=([0-9]+([.][0-9]+)?).*/\1/p' | head -1)
    if [ -z "$AVG_WEIGHT" ]; then
      echo "[ERROR] Could not read AVG_WEIGHT from TDC CH0 — IP not responding."
      echo "$AVG_OUTPUT"
      exit 1
    fi

    echo "  Pre-cal: AVG_WEIGHT=$AVG_WEIGHT (calibration sweep targets ~half-depth)"

    echo "  [1b] Running calibration sweep (8192 iterations)..."
    CAL_OUTPUT=$(run_ssh \
      "cd '$REMOTE_DIR' && sudo ./myapp tdc --calibrate 8192 --verbose" 2>&1) || {
        echo "[ERROR] tdc --calibrate failed"
        echo "$CAL_OUTPUT"
        exit 1
      }

    echo "$CAL_OUTPUT" | grep -E "Calibrat|fine=|coarse=|sweep|tap|best|score" || true

    CAL_FINE=$(  echo "$CAL_OUTPUT" | sed -nE 's/.*fine=(0x[0-9A-Fa-f]+).*/\1/p'   | tail -1)
    CAL_COARSE=$(echo "$CAL_OUTPUT" | sed -nE 's/.*coarse=(0x[0-9A-Fa-f]+).*/\1/p' | tail -1)

    if [ -z "$CAL_FINE" ] || [ -z "$CAL_COARSE" ]; then
      RAW64=$(echo "$CAL_OUTPUT" \
        | sed -nE 's/.*Calibration result delay:[[:space:]]*(0x[0-9A-Fa-f]+).*/\1/p' | head -1)
      if [ -n "$RAW64" ]; then
        HEX="${RAW64#0x}"
        HEX=$(printf "%016s" "$HEX" | tr ' ' '0')
        CAL_COARSE="0x${HEX:0:8}"
        CAL_FINE="0x${HEX:8:8}"
      fi
    fi

    if [ -z "$CAL_FINE" ] || [ -z "$CAL_COARSE" ]; then
      echo "[ERROR] Could not parse calibration result. Full output:"
      echo "$CAL_OUTPUT"
      exit 1
    fi

    if [ "$CAL_FINE" = "0x00000000" ] && [ "$CAL_COARSE" = "0x00000000" ]; then
      echo "[ERROR] Calibration returned all-zeros — sweep did not converge."
      echo "  Try power-cycling the board and re-running."
      exit 1
    fi

    TDC_DELAY="${CAL_FINE},${CAL_COARSE}"
    echo "$TDC_DELAY" > "$RESULTS_DIR/tdc_calibration.txt"
    echo "  Calibration result: fine=$CAL_FINE  coarse=$CAL_COARSE"

    echo "  [1d] Writing calibrated delay to hardware and verifying..."
    run_ssh \
      "cd '$REMOTE_DIR' && sudo ./myapp tdc --set-all $CAL_FINE $CAL_COARSE --get -1" \
      | grep -E "delay|fine|coarse" || true

    echo "  Thermal settling (5s)..."
    sleep 5
  fi
else
  echo "[Step 1] Using provided TDC calibration: $TDC_DELAY"
  mkdir -p "$RESULTS_DIR"
  echo "$TDC_DELAY" > "$RESULTS_DIR/tdc_calibration.txt"
fi

echo
echo "  ┌─────────────────────────────────────────────┐"
echo "  │ TDC LOCKED: $TDC_DELAY"
echo "  └─────────────────────────────────────────────┘"
echo

# ==============================================================================
# STEP 2: COLLECT L1, L2, AND DRAM TRACES
# ==============================================================================
if ! $SKIP_COLLECT; then
  echo "[Step 2] Collecting training traces (shuffled patterns, locked TDC)..."
  echo "  Patterns: ${PATTERNS[*]} → $(for p in "${PATTERNS[@]}"; do printf "%s " "${PATTERN_NAMES[$p]}"; done)"
  echo

  mkdir -p "$LOCAL_TRACES"

  declare -A RUN_COUNT
  for p in "${PATTERNS[@]}"; do RUN_COUNT[$p]=0; done

  total_runs=$(( ${#PATTERNS[@]} * REPS ))
  current_run=0

  for ((cycle=1; cycle<=REPS; cycle++)); do
    shuffled=($(shuffle_array "${PATTERNS[@]}"))
    echo "  [Cycle $cycle/$REPS] Order: ${shuffled[*]}"

    for p in "${shuffled[@]}"; do
      RUN_COUNT[$p]=$(( RUN_COUNT[$p] + 1 ))
      r=${RUN_COUNT[$p]}
      name=${PATTERN_NAMES[$p]}
      current_run=$((current_run + 1))

      local_dir="${LOCAL_TRACES}/pattern_${p}"
      mkdir -p "$local_dir"
      out_file="${local_dir}/all_traces_p${p}_run$(printf "%02d" "$r")_$(ts).csv"

      echo "    [$current_run/$total_runs] Pattern=$p ($name) Run=$r/$REPS"

      run_ssh "sudo rm -f '$REMOTE_CSV'" >/dev/null 2>&1
      run_ssh "cd '$REMOTE_DIR' && export SNOOPYPOWER_TDC_DELAY='${TDC_DELAY}' && sudo -E ./myapp sca --pattern '$p' --iters '$ITERS' --mode '$MODE'"
      run_ssh "sudo test -s '$REMOTE_CSV'" >/dev/null 2>&1
      scp -q "${HOST}:${REMOTE_CSV}" "$out_file"
      run_ssh "sudo rm -f '$REMOTE_CSV'" >/dev/null 2>&1

      lines=$(wc -l < "$out_file")
      echo "      → $out_file ($lines lines)"

      if [ "$current_run" -lt "$total_runs" ]; then
        echo "      Settling (${SETTLE_TIME}s)..."
        sleep "$SETTLE_TIME"
      fi
    done
  done
  echo "  Collection complete"
else
  echo "[Step 2] Skipping trace collection (--skip-collect)"
fi
echo

# ==============================================================================
# STEP 3: MERGE CSVs
# ==============================================================================
L1_CSV="${LOCAL_TRACES}/l1_traces.csv"
L2_CSV="${LOCAL_TRACES}/l2_traces.csv"
DRAM_CSV="${LOCAL_TRACES}/DRAM_traces.csv"

if ! $SKIP_COLLECT; then
  echo "[Step 3] Merging per-pattern CSVs..."

  : > "$L1_CSV"; : > "$L2_CSV"; : > "$DRAM_CSV"
  shopt -s nullglob

  for p in "${PATTERNS[@]}"; do
    class_name="${PATTERN_NAMES[$p]}"
    for f in "${LOCAL_TRACES}"/pattern_"$p"/all_traces_p"$p"_*.csv; do
      case "$class_name" in
        L1)   cat "$f" >> "$L1_CSV" ;;
        L2)   cat "$f" >> "$L2_CSV" ;;
        DRAM) cat "$f" >> "$DRAM_CSV" ;;
      esac
    done
  done
  shopt -u nullglob

  echo "  L1:   $(wc -l < "$L1_CSV") traces → $L1_CSV"
  echo "  L2:   $(wc -l < "$L2_CSV") traces → $L2_CSV"
  echo "  DRAM: $(wc -l < "$DRAM_CSV") traces → $DRAM_CSV"
else
  echo "[Step 3] Reusing existing CSVs:"
  echo "  L1:   $(wc -l < "$L1_CSV") traces"
  echo "  L2:   $(wc -l < "$L2_CSV") traces"
  echo "  DRAM: $(wc -l < "$DRAM_CSV") traces"
fi
echo

# ==============================================================================
# STEP 4: STATISTICAL LEAKAGE ASSESSMENT
# ==============================================================================
echo "[Step 4] Running OS_characterization.py..."
mkdir -p "$FIGURES_DIR"
python3 "${SCRIPTS_DIR}/OS_characterization.py"

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             OS CHARACTERIZATION COMPLETE                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "  TDC calibration: $TDC_DELAY"
echo "  Traces:"
echo "    L1:   $L1_CSV  ($(wc -l < "$L1_CSV") lines)"
echo "    L2:   $L2_CSV  ($(wc -l < "$L2_CSV") lines)"
echo "    DRAM: $DRAM_CSV ($(wc -l < "$DRAM_CSV") lines)"
echo "  Figures:         $FIGURES_DIR/"
echo "  Results:         $RESULTS_DIR/"
echo
echo "  To re-run analysis only (skip collection):"
echo "    $0 --skip-collect --tdc '$TDC_DELAY'"
echo
