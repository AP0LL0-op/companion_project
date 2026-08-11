#!/usr/bin/env bash
# Run the companion with full diagnostic capture, so an HSA fault leaves evidence behind.
#
# Runs inside a PTY (via `script`) so the session stays fully interactive -
# piping stdout would swallow the `you> ` prompt, which has no trailing newline.
#
# Captures into logs/<timestamp>/ beside this script:
#   session.log  - everything printed, including any traceback
#   gpu.csv      - VRAM / clocks / temp / power, sampled every second
#   kernel.log   - amdgpu/HSA kernel messages from the session window
#   summary.txt  - exit code, last turns before a crash, GPU state at fault
#
# Usage:  ./run_logged.sh [--voice --timing ...]     (args pass through)
#         SERIALIZE=1 ./run_logged.sh --voice        (slow; pins the faulting kernel)

set -uo pipefail
# Interpreter: override with COMPANION_PY, else the active env, else PATH.
PY="${COMPANION_PY:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PY="${PY:-$(command -v python3)}"
# Resolve our own directory rather than naming one: this has to work from
# any checkout location and any working directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$HERE/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DIR"

# --- GPU sampler ---------------------------------------------------------
(
  echo "wall,vram_used_mb,vram_free_mb,sclk,mclk,temp_c,power_w"
  while true; do
    used=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}')
    tot=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Total Memory/{print int($NF/1048576); exit}')
    # concise-info row fields: 5=temp 6=power 10=sclk 11=mclk
    read -r sclk mclk temp pwr <<<"$(rocm-smi 2>/dev/null | awk '/^0 /{gsub(/Mhz|°C|W/,""); print $10, $11, $5, $6}')"
    echo "$(date +%H:%M:%S.%3N),${used:-},$(( ${tot:-0} - ${used:-0} )),${sclk:-},${mclk:-},${temp:-},${pwr:-}"
    sleep 1
  done
) > "$DIR/gpu.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

START=$(date '+%Y-%m-%d %H:%M:%S')
CMD="$PY -u $HERE/companion.py $*"
if [ "${SERIALIZE:-0}" = "1" ]; then
  # One kernel at a time, so a fault is attributed to the operation that
  # actually caused it rather than surfacing downstream. Much slower.
  CMD="env AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 $CMD"
  echo "SERIALIZE mode: slow, but faults point at the real kernel." >&2
fi

echo "Logging to $DIR  (talk normally; Ctrl-D or 'exit' to finish)" >&2
# -q quiet, -e return the child's exit status, -f flush after each write
script -q -e -f -c "$CMD" "$DIR/session.log"
RC=$?

journalctl -k --no-pager --since "$START" 2>/dev/null \
  | grep -iE "amdgpu|hsa|gpu|fault|reset|segfault" > "$DIR/kernel.log"

{
  echo "exit code: $RC"
  echo "started:   $START"
  echo "ended:     $(date '+%Y-%m-%d %H:%M:%S')"
  echo "turns:     $(grep -c 'you>' "$DIR/session.log" 2>/dev/null || echo 0)"
  echo
  if grep -qE "HSA_STATUS_ERROR|unspecified launch failure" "$DIR/session.log" 2>/dev/null; then
    echo "=== HSA FAULT DETECTED ==="
    echo "--- turns leading up to it ---"
    # Her prompt is her configured name, so derive the pattern rather than
    # hardcoding one that stopped being true the moment she was renamed.
    HERNAME=$(grep -h "^COMPANION_NAME=" "$HERE/.env" 2>/dev/null | cut -d= -f2- | tr '[:upper:]' '[:lower:]')
    grep -aE "you>|${HERNAME:-[[:alnum:]_-]+}>|time to first sound" "$DIR/session.log" | tail -12
    echo
    echo "--- fault ---"
    grep -aE "HSA_STATUS_ERROR|aborting with error|unspecified launch|Queue.*idle" "$DIR/session.log" | head -6
    echo
    echo "--- GPU state, last ~12s before exit ---"
    tail -13 "$DIR/gpu.csv"
    echo
    echo "--- kernel messages ---"
    head -20 "$DIR/kernel.log"
  else
    echo "no HSA fault this session"
  fi
} > "$DIR/summary.txt"

echo >&2
cat "$DIR/summary.txt" >&2
echo >&2
echo "logs: $DIR" >&2
