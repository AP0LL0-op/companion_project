#!/usr/bin/env bash
#
# install.sh — set up the local voice companion stack on this machine.
#
# The stack is four moving parts that each have to agree with the others:
# a PyTorch build matching your GPU vendor, a llama.cpp built for your exact
# GPU architecture, a set of model weights sized to your VRAM, and a working
# capture device. Get any one wrong and the failure surfaces somewhere else
# entirely — this script exists so that none of it has to be guessed at.
#
# It runs in four phases and does nothing irreversible before the third:
#
#   1. DETECT   read the machine. No writes, no network, no sudo.
#   2. PLAN     derive every decision from what was found, and show the
#               reasoning — which wheel index, which GPU target, which model
#               tier, which context size, and why.
#   3. CONFIRM  print the whole plan and wait for a yes.
#   4. DEPLOY   build it, then verify what was built and report versions.
#
# This script NEVER calls sudo. Where a system package is missing it tells you
# the exact command for your distribution and stops. Everything it installs
# itself lives in a virtualenv and under --prefix, and can be deleted by
# removing those two directories.
#
# Usage:
#   ./install.sh                 detect, plan, confirm, install
#   ./install.sh --check         phases 1-2 only; also a post-install health check
#   ./install.sh --dry-run       full plan, print every command, run none
#   ./install.sh --yes           accept the plan and every model without asking
#   ./install.sh --help          all options
#
# Re-running is safe. Each step checks whether it is already done, so an
# interrupted run picks up where it stopped rather than starting over.

set -Eeuo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SELF/install.log"
STATE="$SELF/.install-state"

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

# llama.cpp and model weights. Defaults to a sibling of the checkout, so a
# clone is self-contained and nothing is written to a path this project
# invented on someone else's machine.
PREFIX="${COMPANION_PREFIX:-$(dirname "$SELF")/runtime}"
MODE="install"                              # install | check | dry-run
ASSUME_YES=0
WANT_BACKEND=""       # rocm | cuda | cpu — override autodetect
WANT_GFX=""           # e.g. gfx1100 — override detected AMD target
WANT_SM=""            # e.g. 86 — override detected CUDA arch
WANT_PYTHON=""
ENV_MODE="auto"       # auto | venv | conda | current
DO_TORCH=1
DO_DEPS=1
DO_LLAMA=1
DO_MODELS=1
DO_BENCH=1
VERBOSE=0

usage() {
  sed -n '2,40p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'
  cat <<'EOF'

Options:
  --check                 Detect and plan only. Never writes. Use this after
                          installing too — it re-runs every verification and
                          prints the version table.
  --dry-run               Plan and print each command that would run.
  --yes, -y               Non-interactive. Accepts the plan and downloads every
                          model in the plan without prompting.
  --prefix DIR            Where llama.cpp and model weights go.
                          Default: ../runtime beside the checkout
                          (or $COMPANION_PREFIX)
  --backend rocm|cuda|cpu Override GPU backend detection.
  --gfx TARGET            Override the AMD GPU target (e.g. gfx1100).
  --sm ARCH               Override the NVIDIA compute capability (e.g. 89).
  --python PATH           Interpreter to build the environment from.
  --env venv|conda|current
                          Where Python packages go. "current" installs into
                          whatever interpreter is active — only do that in an
                          environment you already made for this.
  --no-torch              Skip the torch install (it is already correct).
  --no-deps               Skip requirements.txt.
  --no-llama              Skip building llama.cpp.
  --no-models             Skip all model downloads.
  --no-bench              Skip the batch-1 GEMV benchmark.
  --verbose               Echo every command as it runs.
  -h, --help              This.

Exit codes:
  0 success   1 usage   2 unmet requirement   3 build failure
  4 download failure    5 verification failure
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)     MODE="check" ;;
    --dry-run)   MODE="dry-run" ;;
    -y|--yes)    ASSUME_YES=1 ;;
    --prefix)    PREFIX="${2:?--prefix needs a directory}"; shift ;;
    --backend)   WANT_BACKEND="${2:?--backend needs rocm|cuda|cpu}"; shift ;;
    --gfx)       WANT_GFX="${2:?--gfx needs a target}"; shift ;;
    --sm)        WANT_SM="${2:?--sm needs an arch}"; shift ;;
    --python)    WANT_PYTHON="${2:?--python needs a path}"; shift ;;
    --env)       ENV_MODE="${2:?--env needs venv|conda|current}"; shift ;;
    --no-torch)  DO_TORCH=0 ;;
    --no-deps)   DO_DEPS=0 ;;
    --no-llama)  DO_LLAMA=0 ;;
    --no-models) DO_MODELS=0 ;;
    --no-bench)  DO_BENCH=0 ;;
    --verbose)   VERBOSE=1 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; echo "try --help" >&2; exit 1 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BLU=$'\033[34m'
else
  B=""; DIM=""; R=""; RED=""; GRN=""; YEL=""; BLU=""
fi

log()  { printf '%s\n' "$*" >>"$LOG"; }
say()  { printf '%s\n' "$*"; log "$*"; }
head1() { say ""; say "${B}$*${R}"; say "${DIM}$(printf '─%.0s' $(seq 1 68))${R}"; }
step() { printf '%s\n' "  ${BLU}▸${R} $*"; log "STEP: $*"; }
ok()   { printf '%s\n' "  ${GRN}✓${R} $*"; log "OK: $*"; }
warn() { printf '%s\n' "  ${YEL}!${R} $*"; log "WARN: $*"; }
note() { printf '%s\n' "    ${DIM}$*${R}"; log "NOTE: $*"; }

# die CODE MESSAGE [FIX...]
# Every fatal exit goes through here so the user always gets the same three
# things: what failed, what to do about it, and where the log is.
die() {
  local code="$1"; shift
  local msg="$1"; shift
  printf '\n%s\n' "${RED}${B}✗ $msg${R}" >&2
  log "FATAL($code): $msg"
  if [[ $# -gt 0 ]]; then
    printf '\n%s\n' "  ${B}To fix:${R}" >&2
    for line in "$@"; do printf '    %s\n' "$line" >&2; log "  FIX: $line"; done
  fi
  printf '\n  %s\n\n' "${DIM}Full log: $LOG${R}" >&2
  exit "$code"
}

trap 'rc=$?; [[ $rc -ne 0 ]] && printf "\n%s\n" "${RED}✗ install.sh aborted at line $LINENO (exit $rc): ${BASH_COMMAND}${R}" >&2 && printf "  %s\n\n" "${DIM}Full log: $LOG${R}" >&2; exit $rc' ERR

# run — execute a command, honouring --dry-run, logging output.
run() {
  log "RUN: $*"
  if [[ "$MODE" == "dry-run" ]]; then
    printf '    %s\n' "${DIM}would run: $*${R}"
    return 0
  fi
  if [[ $VERBOSE -eq 1 ]]; then
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
  fi
  "$@" >>"$LOG" 2>&1
}

# Requirement table. Rows are collected during detection and printed together,
# so the user sees the whole picture at once instead of a scroll of checks.
ROWS=()
row() { ROWS+=("$1|$2|$3|$4|$5"); }   # status|name|required|found|note

print_rows() {
  local st name req found note colour mark
  [[ ${#ROWS[@]} -eq 0 ]] && return 0
  printf '  %-22s %-16s %-22s\n' "COMPONENT" "REQUIRED" "FOUND"
  printf '  %s\n' "${DIM}$(printf '─%.0s' $(seq 1 66))${R}"
  for r in "${ROWS[@]}"; do
    IFS='|' read -r st name req found note <<<"$r"
    case "$st" in
      OK)   colour="$GRN"; mark="✓" ;;
      WARN) colour="$YEL"; mark="!" ;;
      MISS) colour="$RED"; mark="✗" ;;
      *)    colour="$DIM"; mark="·" ;;
    esac
    printf '  %s%s%s %-20s %-16s %-22s\n' "$colour" "$mark" "$R" "$name" "$req" "$found"
    [[ -n "$note" ]] && printf '    %s\n' "${DIM}$note${R}"
  done
}

have() { command -v "$1" >/dev/null 2>&1; }

# ver_ge A B — true if version A >= version B
ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }

confirm() {
  [[ $ASSUME_YES -eq 1 ]] && return 0
  [[ "$MODE" == "dry-run" ]] && return 0
  local reply
  read -r -p "  $1 [y/N] " reply </dev/tty || return 1
  [[ "$reply" =~ ^[Yy] ]]
}

: >"$LOG"
log "install.sh started $(date -Is)"
log "argv: $* | mode=$MODE prefix=$PREFIX"

say ""
say "${B}A local voice companion: installer${R}"
say "${DIM}$SELF${R}"

# ===========================================================================
# PHASE 1 — DETECT
# ===========================================================================

head1 "1. Detecting"

# --- operating system -------------------------------------------------------
D_OS="$(uname -s)"
D_DISTRO="unknown"; D_DISTRO_VER=""; D_PKG=""
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  D_DISTRO="${ID:-unknown}"; D_DISTRO_VER="${VERSION_ID:-}"
fi
case "$D_DISTRO" in
  ubuntu|debian|linuxmint|pop) D_PKG="apt" ;;
  fedora|rhel|centos|rocky|almalinux) D_PKG="dnf" ;;
  arch|manjaro|endeavouros) D_PKG="pacman" ;;
  opensuse*|sles) D_PKG="zypper" ;;
esac
if [[ "$D_OS" != "Linux" ]]; then
  die 2 "This stack is Linux-only. Detected: $D_OS" \
    "ROCm does not exist on macOS or Windows, and the audio path is ALSA." \
    "On Windows, WSL2 with a CUDA GPU is the closest workable option."
fi
row OK "OS" "Linux" "$D_DISTRO ${D_DISTRO_VER}" ""

# --- package name map, per distro, for the instruct-don't-install path -------
pkg_for() {
  # pkg_for <generic>  ->  distro package name
  case "$D_PKG:$1" in
    apt:cmake) echo "cmake" ;;      dnf:cmake) echo "cmake" ;;
    pacman:cmake) echo "cmake" ;;   zypper:cmake) echo "cmake" ;;
    apt:build) echo "build-essential" ;;
    dnf:build) echo "gcc gcc-c++ make" ;;
    pacman:build) echo "base-devel" ;;
    zypper:build) echo "gcc gcc-c++ make" ;;
    apt:git) echo "git" ;; dnf:git) echo "git" ;; pacman:git) echo "git" ;; zypper:git) echo "git" ;;
    apt:alsa) echo "alsa-utils" ;;  dnf:alsa) echo "alsa-utils" ;;
    pacman:alsa) echo "alsa-utils" ;; zypper:alsa) echo "alsa-utils" ;;
    apt:python) echo "python3 python3-venv python3-dev" ;;
    dnf:python) echo "python3 python3-devel" ;;
    pacman:python) echo "python" ;; zypper:python) echo "python3 python3-devel" ;;
    apt:curl) echo "curl" ;; dnf:curl) echo "curl" ;; pacman:curl) echo "curl" ;; zypper:curl) echo "curl" ;;
    *) echo "$1" ;;
  esac
}
install_cmd() {
  case "$D_PKG" in
    apt)    echo "sudo apt install $*" ;;
    dnf)    echo "sudo dnf install $*" ;;
    pacman) echo "sudo pacman -S $*" ;;
    zypper) echo "sudo zypper install $*" ;;
    *)      echo "install with your package manager: $*" ;;
  esac
}

# --- python -----------------------------------------------------------------
D_PY=""
for cand in "$WANT_PYTHON" python3.12 python3.11 python3.13 python3.10 python3 python; do
  [[ -z "$cand" ]] && continue
  if have "$cand"; then D_PY="$(command -v "$cand")"; break; fi
done
if [[ -z "$D_PY" ]]; then
  row MISS "Python" ">= 3.10" "not found" ""
  die 2 "No Python interpreter found." "$(install_cmd "$(pkg_for python)")"
fi
D_PY_VER="$("$D_PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
if ver_ge "$D_PY_VER" "3.10"; then
  if ver_ge "$D_PY_VER" "3.14"; then
    row WARN "Python" ">= 3.10" "$D_PY_VER" "3.14+ is ahead of the torch wheels; 3.12 is the tested version"
  else
    row OK "Python" ">= 3.10" "$D_PY_VER" ""
  fi
else
  row MISS "Python" ">= 3.10" "$D_PY_VER" ""
  die 2 "Python $D_PY_VER is too old (transformers and torch need 3.10+)." \
    "$(install_cmd "$(pkg_for python)")" \
    "or point the installer at a newer one:  ./install.sh --python /usr/bin/python3.12"
fi

# --- GPU --------------------------------------------------------------------
# Detection order matters: ask the vendor runtime first (authoritative about
# architecture and VRAM), and only fall back to lspci, which can name a card
# but cannot tell you whether its driver stack is actually installed.
D_VENDOR="none"; D_GPU="unknown"; D_GFX=""; D_SM=""; D_VRAM_GB=0
D_ROCM_VER=""; D_CUDA_VER=""; D_DRIVER=""

detect_amd() {
  have rocminfo || return 1
  # A machine can present several agents: the discrete card plus an APU's
  # integrated one. Pick the target with the most VRAM, not the first listed —
  # on this reference machine the iGPU (gfx1036) enumerates second but a
  # naive `head -1` on some systems picks the wrong one.
  local best_gfx="" best_vram=0 gfx vram
  while read -r gfx; do
    [[ -z "$gfx" ]] && continue
    vram=0
    if have rocm-smi; then
      vram=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/VRAM Total Memory/{print $NF; exit}')
      vram=$(( ${vram:-0} / 1073741824 ))
    fi
    if [[ -z "$best_gfx" || $vram -gt $best_vram ]]; then best_gfx="$gfx"; best_vram=$vram; fi
  done < <(rocminfo 2>/dev/null | awk '/^  Name: *gfx/{print $2}' | sort -u)
  [[ -z "$best_gfx" ]] && return 1

  D_VENDOR="amd"; D_GFX="$best_gfx"
  # Largest VRAM across all agents, rounded rather than truncated: a 16 GiB
  # card reports 17163091968 bytes, which floors to 15 and would drop the
  # machine a model tier for no reason.
  if have rocm-smi; then
    D_VRAM_GB=$(rocm-smi --showmeminfo vram 2>/dev/null \
      | awk '/VRAM Total Memory/{print $NF}' | sort -n | tail -1 \
      | awk '{printf "%d", ($1 / 1073741824) + 0.5}')
  fi
  # The marketing name has to come from the same agent block as the gfx target.
  # rocminfo lists the CPU as an agent too, and its marketing name is the
  # processor model — taking the first one reports a Ryzen as your GPU.
  D_GPU="$(rocminfo 2>/dev/null | awk -v want="$best_gfx" '
      $1 == "Name:" && $2 == want { hit = 1; next }
      hit && /Marketing Name:/ { sub(/^[^:]*: **/, ""); sub(/ *$/, ""); print; exit }')"
  [[ -z "$D_GPU" ]] && D_GPU="AMD $D_GFX"
  if [[ -r /opt/rocm/.info/version ]]; then
    D_ROCM_VER="$(cat /opt/rocm/.info/version)"
  elif have hipconfig; then
    D_ROCM_VER="$(hipconfig --version 2>/dev/null | head -1)"
  fi
  return 0
}

detect_nvidia() {
  have nvidia-smi || return 1
  local line
  line="$(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
          --format=csv,noheader,nounits 2>/dev/null | head -1)" || return 1
  [[ -z "$line" ]] && return 1
  D_VENDOR="nvidia"
  D_GPU="$(cut -d, -f1 <<<"$line" | xargs)"
  D_VRAM_GB=$(( $(cut -d, -f2 <<<"$line" | xargs) / 1024 ))
  local cc; cc="$(cut -d, -f3 <<<"$line" | xargs)"
  D_SM="${cc//./}"
  D_DRIVER="$(cut -d, -f4 <<<"$line" | xargs)"
  if have nvcc; then
    D_CUDA_VER="$(nvcc --version 2>/dev/null | awk '/release/{print $NF}' | tr -d 'V,')"
  fi
  return 0
}

if [[ -n "$WANT_BACKEND" ]]; then
  case "$WANT_BACKEND" in
    rocm) detect_amd || true; D_VENDOR="amd" ;;
    cuda) detect_nvidia || true; D_VENDOR="nvidia" ;;
    cpu)  D_VENDOR="cpu" ;;
    *) die 1 "--backend must be rocm, cuda, or cpu" ;;
  esac
else
  detect_amd || detect_nvidia || true
fi
[[ -n "$WANT_GFX" ]] && D_GFX="$WANT_GFX"
[[ -n "$WANT_SM"  ]] && D_SM="$WANT_SM"

if [[ "$D_VENDOR" == "none" ]]; then
  # Nothing answered. Say what the hardware is anyway — "no GPU found" when
  # there is clearly a GPU in the box is the least useful message possible.
  local_card="$(lspci 2>/dev/null | grep -Ei 'vga|3d controller|display' | head -1 | cut -d: -f3- | xargs || true)"
  if [[ -n "$local_card" ]]; then
    row MISS "GPU runtime" "ROCm or CUDA" "driver stack absent" "card present: $local_card"
    if grep -qi 'nvidia' <<<"$local_card"; then
      die 2 "An NVIDIA card is present but nvidia-smi does not run." \
        "Install the NVIDIA driver and CUDA toolkit for your distribution:" \
        "  https://developer.nvidia.com/cuda-downloads" \
        "Then re-run. Verify first with:  nvidia-smi"
    elif grep -qiE 'amd|advanced micro|radeon' <<<"$local_card"; then
      die 2 "An AMD card is present but rocminfo does not run." \
        "Install ROCm for your distribution:" \
        "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/" \
        "Add yourself to the render/video groups, then log out and back in:" \
        "  sudo usermod -aG render,video \$USER" \
        "Verify first with:  rocminfo | grep gfx"
    fi
  fi
  row WARN "GPU" "ROCm or CUDA" "none usable" "will plan a CPU-only install — synthesis will not run in real time"
  D_VENDOR="cpu"
else
  if [[ "$D_VENDOR" == "amd" ]]; then
    row OK "GPU" "any ROCm target" "$D_GPU ($D_GFX)" "${D_VRAM_GB} GiB VRAM"
    if [[ -n "$D_ROCM_VER" ]]; then
      row OK "ROCm" ">= 6.0" "$D_ROCM_VER" ""
    else
      row WARN "ROCm" ">= 6.0" "version unknown" "rocminfo works but /opt/rocm/.info/version is missing"
    fi
  else
    row OK "GPU" "any CUDA target" "$D_GPU (sm_$D_SM)" "${D_VRAM_GB} GiB VRAM, driver $D_DRIVER"
    if [[ -n "$D_CUDA_VER" ]]; then
      row OK "CUDA toolkit" ">= 12.1" "$D_CUDA_VER" ""
    else
      row WARN "CUDA toolkit" ">= 12.1" "nvcc not found" "needed only to build llama.cpp; torch ships its own runtime"
    fi
  fi
fi

# --- AMD architecture support table -----------------------------------------
# Officially supported targets need nothing extra. The rest work through
# HSA_OVERRIDE_GFX_VERSION, which lies to ROCm about the architecture so it
# loads the nearest supported kernel set. That is a real technique, not a hack
# of last resort, but it is not guaranteed and is worth stating plainly.
D_GFX_SUPPORT="unknown"; D_HSA_OVERRIDE=""
case "$D_GFX" in
  "") ;;
  gfx90a|gfx942|gfx950)                 D_GFX_SUPPORT="official" ;;   # CDNA 2/3/4
  gfx1030|gfx1100|gfx1101|gfx1102)      D_GFX_SUPPORT="official" ;;   # RDNA 2/3
  gfx1200|gfx1201)                      D_GFX_SUPPORT="official" ;;   # RDNA 4
  gfx1151|gfx1150)                      D_GFX_SUPPORT="official" ;;   # Strix Halo/Point
  gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036)
      D_GFX_SUPPORT="override"; D_HSA_OVERRIDE="10.3.0" ;;
  gfx1103)  D_GFX_SUPPORT="override"; D_HSA_OVERRIDE="11.0.0" ;;
  gfx900|gfx906) D_GFX_SUPPORT="legacy" ;;
  *) D_GFX_SUPPORT="untested" ;;
esac

# --- build tools ------------------------------------------------------------
MISSING_PKGS=()
if have cmake; then
  D_CMAKE="$(cmake --version | head -1 | awk '{print $3}')"
  if ver_ge "$D_CMAKE" "3.21"; then row OK "cmake" ">= 3.21" "$D_CMAKE" ""
  else row MISS "cmake" ">= 3.21" "$D_CMAKE" ""; MISSING_PKGS+=("$(pkg_for cmake)"); fi
else
  row MISS "cmake" ">= 3.21" "not found" ""; MISSING_PKGS+=("$(pkg_for cmake)")
fi

if have git; then row OK "git" "any" "$(git --version | awk '{print $3}')" ""
else row MISS "git" "any" "not found" ""; MISSING_PKGS+=("$(pkg_for git)"); fi

if have cc || have gcc; then
  row OK "C/C++ compiler" "any" "$( (cc --version 2>/dev/null || gcc --version) | head -1 | cut -c1-30)" ""
else
  row MISS "C/C++ compiler" "any" "not found" ""; MISSING_PKGS+=("$(pkg_for build)")
fi


if [[ "$D_VENDOR" == "amd" ]]; then
  if have hipcc; then
    row OK "hipcc" "any" "$(hipcc --version 2>/dev/null | awk '/HIP version/{print $3}')" ""
  else
    row MISS "hipcc" "any" "not found" "part of the ROCm dev packages; llama.cpp cannot be built for HIP without it"
  fi
elif [[ "$D_VENDOR" == "nvidia" ]]; then
  if have nvcc; then row OK "nvcc" ">= 12.1" "${D_CUDA_VER:-present}" ""
  else row WARN "nvcc" ">= 12.1" "not found" "install the CUDA toolkit to build llama.cpp with GPU offload"; fi
fi

# --- audio ------------------------------------------------------------------
if have arecord && have aplay; then
  row OK "ALSA utils" "arecord/aplay" "$(arecord --version | awk '{print $3}')" ""
  D_CAPTURE="$(arecord -l 2>/dev/null | grep -c '^card' || true)"
  if [[ "${D_CAPTURE:-0}" -gt 0 ]]; then
    row OK "Capture device" ">= 1" "$D_CAPTURE found" ""
  else
    row WARN "Capture device" ">= 1" "none" "voice mode will not work; typed mode still will"
  fi
else
  row MISS "ALSA utils" "arecord/aplay" "not found" ""; MISSING_PKGS+=("$(pkg_for alsa)")
fi
if have wpctl; then
  row OK "PipeWire" "optional" "present" "use wpctl for mic gain — amixer changes get overwritten"
fi

# --- disk -------------------------------------------------------------------
# Measure the filesystem the prefix will live on WITHOUT creating it. Detection
# promises to write nothing, and `mkdir -p` here broke that promise: --check
# left an empty directory behind on a machine it was only supposed to inspect.
# Walk up to the nearest parent that exists instead; df reports the same
# filesystem either way.
_df_target="$PREFIX"
while [[ ! -d "$_df_target" && "$_df_target" != "/" ]]; do
  _df_target="$(dirname "$_df_target")"
done
D_DISK_GB=$(df -BG --output=avail "$_df_target" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [[ "${D_DISK_GB:-0}" -ge 30 ]]; then
  row OK "Disk at prefix" ">= 30 GiB" "${D_DISK_GB} GiB free" "$PREFIX"
else
  row WARN "Disk at prefix" ">= 30 GiB" "${D_DISK_GB} GiB free" "$PREFIX — weights alone are ~14 GiB, plus the llama.cpp build"
fi

# --- existing install -------------------------------------------------------
D_LLAMA_BIN="$PREFIX/llama.cpp/build/bin/llama-server"
if [[ -x "$D_LLAMA_BIN" ]]; then
  D_LLAMA_VER="$(cd "$PREFIX/llama.cpp" 2>/dev/null && git log -1 --format=%h 2>/dev/null || echo present)"
  row OK "llama.cpp" "built" "$D_LLAMA_VER" "$D_LLAMA_BIN"
else
  row MISS "llama.cpp" "built" "not built" "will clone and build into $PREFIX/llama.cpp"
fi

print_rows

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  die 2 "Missing system packages. This installer does not use sudo — run this yourself:" \
    "" "  $(install_cmd "${MISSING_PKGS[*]}")" "" \
    "Then re-run ./install.sh — everything already done will be skipped."
fi

# ===========================================================================
# PHASE 2 — PLAN
# ===========================================================================

head1 "2. Plan"

# --- python environment -----------------------------------------------------
P_VENV="$SELF/.venv"
P_ENV_KIND="$ENV_MODE"
if [[ "$ENV_MODE" == "auto" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != "base" ]]; then
    P_ENV_KIND="current"
  else
    P_ENV_KIND="venv"
  fi
fi
case "$P_ENV_KIND" in
  venv)    P_PY="$P_VENV/bin/python" ;;
  current) P_PY="$D_PY"; P_VENV="${CONDA_PREFIX:-$(dirname "$(dirname "$D_PY")")}" ;;
  conda)
    have conda || have mamba || die 2 "--env conda given but neither conda nor mamba is on PATH." \
      "Install Miniforge:  https://github.com/conda-forge/miniforge"
    P_VENV="$HOME/.conda/envs/companion"; P_PY="$P_VENV/bin/python" ;;
  *) die 1 "--env must be venv, conda, or current" ;;
esac

# --- torch wheel index ------------------------------------------------------
# Do not assume a URL exists. PyTorch publishes a specific set of vendor tags
# and retires old ones; probe and take the newest that answers, so this script
# does not rot the moment the index changes.
#
# The probe uses Python's urllib rather than curl. Python is a hard requirement
# of this stack and has already been located; curl is not, and making the whole
# install fail on a missing curl for one HEAD request is not a trade worth
# making.
P_TORCH_INDEX=""; P_TORCH_TAG=""
probe_index() {
  "$D_PY" - "$1" <<'PY' >/dev/null 2>&1
import sys, urllib.request, urllib.error
url = f"https://download.pytorch.org/whl/{sys.argv[1]}/torch/"
req = urllib.request.Request(url, method="HEAD")
try:
    with urllib.request.urlopen(req, timeout=8) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}
pick_torch_tag() {
  local -a candidates=("$@")
  for tag in "${candidates[@]}"; do
    if probe_index "$tag"; then echo "$tag"; return 0; fi
  done
  return 1
}
case "$D_VENDOR" in
  amd)
    rocm_major_minor="$(cut -d. -f1,2 <<<"${D_ROCM_VER:-7.0}")"
    # Prefer an exact match to the installed ROCm, then walk backwards. A torch
    # built for an older ROCm runs on a newer one; the reverse does not.
    P_TORCH_TAG="$(pick_torch_tag \
      "rocm${rocm_major_minor}" rocm7.1 rocm7.0 rocm6.4 rocm6.3 rocm6.2 || true)"
    ;;
  nvidia)
    cuda_major_minor="$(cut -d. -f1,2 <<<"${D_CUDA_VER:-12.8}")"
    P_TORCH_TAG="$(pick_torch_tag \
      "cu${cuda_major_minor//./}" cu130 cu129 cu128 cu126 cu124 cu121 || true)"
    ;;
  cpu) P_TORCH_TAG="cpu" ;;
esac
if [[ -z "$P_TORCH_TAG" ]]; then
  warn "Could not reach download.pytorch.org to confirm a wheel index."
  case "$D_VENDOR" in
    amd) P_TORCH_TAG="rocm7.0" ;; nvidia) P_TORCH_TAG="cu128" ;; *) P_TORCH_TAG="cpu" ;;
  esac
  note "Falling back to $P_TORCH_TAG — check it by hand if the torch install fails."
fi
P_TORCH_INDEX="https://download.pytorch.org/whl/${P_TORCH_TAG}"

# --- llama.cpp build flags --------------------------------------------------
P_CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
case "$D_VENDOR" in
  amd)
    # Both variable names are passed on purpose: llama.cpp renamed
    # AMDGPU_TARGETS to GPU_TARGETS, and an unused -D is only a warning, so
    # this builds correctly against old and new checkouts alike.
    P_CMAKE_FLAGS+=(-DGGML_HIP=ON "-DAMDGPU_TARGETS=$D_GFX" "-DGPU_TARGETS=$D_GFX")
    P_BACKEND_DESC="HIP / ROCm, targeting $D_GFX"
    ;;
  nvidia)
    P_CMAKE_FLAGS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$D_SM")
    P_BACKEND_DESC="CUDA, targeting sm_$D_SM"
    ;;
  cpu)
    P_BACKEND_DESC="CPU only — no GPU offload"
    ;;
esac

# --- VRAM budget ------------------------------------------------------------
# CSM and its working set need roughly 6 GiB and live in the companion.py process.
# llama-server has to fit in what is left, together with its KV cache. This is
# the arithmetic behind -c 16384 on a 16 GiB card, and it is the single number
# most worth getting right: too large and synthesis stalls mid-word from
# driver eviction, which reads as a mysterious audio bug rather than a
# memory-pressure one.
P_CSM_GB=6
P_VRAM_KNOWN=1
if [[ "${D_VRAM_GB:-0}" -le 0 ]]; then
  # Either there is no GPU, or the vendor tool answered but the query failed
  # (common when --backend is forced on a machine without that vendor's card).
  # Sizing anything from a zero here produces a negative budget and a silently
  # wrong plan, so say so instead and fall to the most conservative tier.
  P_VRAM_KNOWN=0
  D_VRAM_GB=0
fi
P_LLM_BUDGET=$(( D_VRAM_GB - P_CSM_GB ))
[[ $P_LLM_BUDGET -lt 0 ]] && P_LLM_BUDGET=0
# Quantisation is a preference list, not one name. Publishers do not agree on
# which quants to ship: Q4_K_M is the most common in the wild, but the ggml-org
# gemma-4 repos carry Q4_0 and Q8_0 and no K-quant at all. So the installer
# states what it wants in priority order and takes the best one that exists,
# rather than hardcoding a filename that is right for exactly one repo.
if   [[ $D_VRAM_GB -ge 22 ]]; then P_TIER="E4B"; P_QUANTS="Q8_0 Q4_K_M Q4_0"; P_CTX=32768
elif [[ $D_VRAM_GB -ge 15 ]]; then P_TIER="E4B"; P_QUANTS="Q4_K_M Q4_0 Q8_0"; P_CTX=16384
elif [[ $D_VRAM_GB -ge 11 ]]; then P_TIER="E4B"; P_QUANTS="Q4_K_M Q4_0";      P_CTX=16384
elif [[ $D_VRAM_GB -ge 8  ]]; then P_TIER="E2B"; P_QUANTS="Q4_K_M Q4_0";      P_CTX=8192
else                               P_TIER="E2B"; P_QUANTS="Q4_K_M Q4_0";      P_CTX=4096
fi
[[ "$D_VENDOR" == "cpu" ]] && { P_TIER="E2B"; P_QUANTS="Q4_K_M Q4_0"; P_CTX=4096; }
P_QUANTS="${COMPANION_QUANTS:-$P_QUANTS}"

# The audio/vision encoder is small enough that quantising it buys little and
# costs input fidelity, so full precision is preferred and Q8_0 is the fallback
# for repos that only publish one. This is a choice, not the accident of
# whichever filename happened to sort first.
P_MMPROJ_QUANTS="BF16 F16 Q8_0"

# Any audio-capable GGUF repo works here; llama.cpp's docs/multimodal.md is the
# authoritative list. Override without editing:  COMPANION_LLM_REPO=... ./install.sh
P_LLM_REPO="${COMPANION_LLM_REPO:-ggml-org/gemma-4-${P_TIER}-it-GGUF}"
P_CSM_REPO="sesame/csm-1b"
P_LORA_REPO="shb777/csm-maya-exp2"

# --- GEMV workaround --------------------------------------------------------
# fast_gemv.py exists because rocBLAS picks a 128x256 macro-tile GEMM for
# batch-1 GEMV on gfx1030 — ~5% GPU utilization. Whether that heuristic is bad
# on YOUR card is a measurement, not a guess, so the default is to benchmark
# it after install and set the toggle from the result.
case "$D_VENDOR:$D_GFX" in
  amd:gfx103*) P_GEMV="benchmark"; P_GEMV_WHY="known-bad on RDNA2; expected to win" ;;
  amd:*)       P_GEMV="benchmark"; P_GEMV_WHY="unmeasured on this architecture" ;;
  nvidia:*)    P_GEMV="benchmark"; P_GEMV_WHY="cuBLAS usually picks correctly; expected to lose" ;;
  *)           P_GEMV="off";       P_GEMV_WHY="CPU backend — not applicable" ;;
esac
[[ $DO_BENCH -eq 0 ]] && { P_GEMV="off"; P_GEMV_WHY="--no-bench given"; }

# --- print the plan ---------------------------------------------------------
say ""
say "  ${B}Backend${R}          $P_BACKEND_DESC"
[[ "$D_GFX_SUPPORT" == "override" ]] && \
  note "gfx target $D_GFX is not officially supported by ROCm — will set HSA_OVERRIDE_GFX_VERSION=$D_HSA_OVERRIDE"
[[ "$D_GFX_SUPPORT" == "untested" ]] && \
  note "gfx target $D_GFX is not in this installer's support table — proceeding, but it is unverified"
[[ "$D_GFX_SUPPORT" == "legacy" ]] && \
  note "gfx target $D_GFX (GCN/Vega) was dropped by recent ROCm — expect to need an older ROCm"
say "  ${B}Python env${R}       $P_ENV_KIND at $P_VENV"
say "  ${B}torch${R}            from $P_TORCH_INDEX"
say "  ${B}llama.cpp${R}        $PREFIX/llama.cpp  (cmake ${P_CMAKE_FLAGS[*]})"
if [[ $P_VRAM_KNOWN -eq 1 ]]; then
  say "  ${B}VRAM budget${R}      ${D_VRAM_GB} GiB total − ${P_CSM_GB} GiB for CSM = ${P_LLM_BUDGET} GiB for the language model"
else
  say "  ${B}VRAM budget${R}      ${YEL}unknown${R} — could not read a VRAM figure from this machine"
  note "planning the smallest tier and the shortest context, which is safe but probably pessimistic"
  note "if you know the number, size it yourself: the model plus its KV cache must leave ~${P_CSM_GB} GiB for CSM"
fi
say "  ${B}Context size${R}     -c $P_CTX  ${DIM}(raising this is what starves CSM and stalls speech)${R}"
say "  ${B}Model tier${R}       $P_LLM_REPO — quant preference: $P_QUANTS, plus its mmproj"
say "  ${B}Voice${R}            $P_CSM_REPO ${DIM}(gated)${R} + LoRA $P_LORA_REPO"
say "  ${B}fast_gemv${R}        $P_GEMV — $P_GEMV_WHY"
say ""
say "  ${DIM}Nothing outside $P_VENV, $PREFIX and this directory is modified.${R}"
say "  ${DIM}sudo is never called.${R}"

if [[ "$MODE" == "check" ]]; then
  say ""
  say "${DIM}--check: stopping before any changes.${R}"
  exit 0
fi

say ""
if ! confirm "Proceed with this plan?"; then
  say ""
  say "Nothing was changed."
  exit 0
fi

# ===========================================================================
# PHASE 3 — DEPLOY
# ===========================================================================

# Which steps have completed, so an interrupted run resumes instead of
# redoing a 20-minute build. --dry-run must not write it: a plan that leaves
# state behind is not a plan.
mark_done() { [[ "$MODE" == "dry-run" ]] && return 0
              grep -qxF "$1" "$STATE" 2>/dev/null || echo "$1" >>"$STATE"; }
is_done()   { grep -qxF "$1" "$STATE" 2>/dev/null; }
[[ "$MODE" == "dry-run" ]] || touch "$STATE" 2>/dev/null || true

# --- python environment -----------------------------------------------------
head1 "3. Python environment"

if [[ "$P_ENV_KIND" == "venv" ]]; then
  if [[ -x "$P_PY" ]]; then
    ok "virtualenv already exists: $P_VENV"
  else
    step "creating virtualenv at $P_VENV"
    run "$D_PY" -m venv "$P_VENV" \
      || die 3 "Could not create the virtualenv." \
           "On Debian/Ubuntu the venv module is a separate package:" \
           "  $(install_cmd "$(pkg_for python)")"
    ok "created"
  fi
elif [[ "$P_ENV_KIND" == "conda" ]]; then
  if [[ -x "$P_PY" ]]; then ok "conda env already exists: $P_VENV"
  else
    step "creating conda env"
    run conda create -y -p "$P_VENV" "python=${D_PY_VER%.*}" \
      || die 3 "conda env creation failed."
    ok "created"
  fi
else
  warn "installing into the active interpreter: $P_PY"
  note "packages will land in ${CONDA_PREFIX:-that environment}, not an isolated venv"
fi

[[ "$MODE" == "dry-run" ]] || [[ -x "$P_PY" ]] || die 3 "Expected an interpreter at $P_PY and there isn't one."
PIP=("$P_PY" -m pip)
run "${PIP[@]}" install --upgrade pip setuptools wheel || warn "pip self-upgrade failed; continuing"

# --- torch ------------------------------------------------------------------
head1 "4. PyTorch ($P_TORCH_TAG)"

torch_ok() {
  [[ -x "$P_PY" ]] || return 1
  "$P_PY" - <<'PY' 2>/dev/null
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

if [[ $DO_TORCH -eq 0 ]]; then
  warn "--no-torch: skipping"
elif torch_ok && is_done "torch:$P_TORCH_TAG"; then
  ok "torch already installed from $P_TORCH_TAG"
else
  step "installing torch from $P_TORCH_INDEX  (this is a large download)"
  run "${PIP[@]}" install --index-url "$P_TORCH_INDEX" torch \
    || die 3 "torch install failed." \
         "The wheel index may not carry a build for Python $D_PY_VER." \
         "Check what exists:  $P_TORCH_INDEX" \
         "Then pin it by hand:  ${P_PY} -m pip install --index-url <index> torch==<version>"
  mark_done "torch:$P_TORCH_TAG"
  ok "installed"
fi

# --- python deps ------------------------------------------------------------
head1 "5. Dependencies"

if [[ $DO_DEPS -eq 0 ]]; then
  warn "--no-deps: skipping"
else
  step "installing requirements.txt"
  run "${PIP[@]}" install -r "$SELF/requirements.txt" \
    || die 3 "Dependency install failed. See $LOG."
  ok "installed"

  # silero-vad's dependency metadata reaches for torchaudio, and pip resolves
  # the CUDA build of it, which breaks a ROCm environment on libcudart.so.13.
  # listen.py never imports the package — it loads the bundled TorchScript
  # directly — so --no-deps here is correct, not a shortcut.
  step "checking torchaudio did not get pulled in"
  if "$P_PY" -c "import torchaudio" >/dev/null 2>&1; then
    warn "torchaudio is installed — on ROCm this is the CUDA build and will break things"
    if confirm "Remove torchaudio?"; then
      run "${PIP[@]}" uninstall -y torchaudio && ok "removed"
    else
      note "left in place at your request; if torch imports start failing, this is why"
    fi
  else
    ok "absent, as it should be"
  fi
fi

# --- llama.cpp --------------------------------------------------------------
head1 "6. llama.cpp ($P_BACKEND_DESC)"

if [[ $DO_LLAMA -eq 0 ]]; then
  warn "--no-llama: skipping"
elif [[ -x "$D_LLAMA_BIN" ]] && is_done "llama:$D_VENDOR:${D_GFX}${D_SM}"; then
  ok "already built for this target: $D_LLAMA_BIN"
else
  if [[ ! -d "$PREFIX/llama.cpp/.git" ]]; then
    step "cloning llama.cpp into $PREFIX/llama.cpp"
    run git clone --depth 1 https://github.com/ggml-org/llama.cpp "$PREFIX/llama.cpp" \
      || die 3 "git clone failed. Check network access to github.com."
  else
    ok "source already present"
  fi

  step "configuring"
  if [[ "$D_VENDOR" == "amd" ]]; then
    HIPCXX="$( (hipconfig -l 2>/dev/null || echo /opt/rocm/llvm/bin) )/clang"
    HIP_PATH="$(hipconfig -R 2>/dev/null || echo /opt/rocm)"
    export HIPCXX HIP_PATH
    log "HIPCXX=$HIPCXX HIP_PATH=$HIP_PATH"
  fi
  run cmake -S "$PREFIX/llama.cpp" -B "$PREFIX/llama.cpp/build" "${P_CMAKE_FLAGS[@]}" \
    || die 3 "cmake configure failed." \
         "The most common cause is a GPU target the toolchain does not know." \
         "Detected target: ${D_GFX:-sm_$D_SM}. Override it with --gfx or --sm." \
         "Full cmake output is in $LOG."

  step "building (this takes several minutes)"
  run cmake --build "$PREFIX/llama.cpp/build" --config Release -j "$(nproc)" \
    || die 3 "llama.cpp build failed." \
         "Out-of-memory during compilation is common on many-core machines —" \
         "retry with fewer jobs:" \
         "  cmake --build $PREFIX/llama.cpp/build --config Release -j 4" \
         "Full compiler output is in $LOG."
  mark_done "llama:$D_VENDOR:${D_GFX}${D_SM}"
  ok "built: $D_LLAMA_BIN"
fi

# --- models -----------------------------------------------------------------
head1 "7. Models"

# All model operations go through one Python helper rather than the hf CLI:
# the CLI's name and flags have changed across releases, while huggingface_hub
# is a library dependency we already pin. It also lets a gated repo report the
# actual reason it refused.
HF_HELPER="$SELF/.install-hf.py"
[[ "$MODE" == "dry-run" ]] && HF_HELPER="$(mktemp -t install-hf-XXXXXX.py)"
cat >"$HF_HELPER" <<'PY'
"""Model download helper for install.sh. Not part of the runtime."""
import os, sys

def _api():
    from huggingface_hub import HfApi
    return HfApi()

def whoami():
    from huggingface_hub import whoami as w
    try:
        print(w()["name"]); return 0
    except Exception:
        print(""); return 1

def pick(repo, kind, quants=""):
    """Resolve one GGUF in repo.

    kind is main | mmproj | draft. Exit codes are distinct on purpose, so the
    caller can tell "this repo does not carry that file" (3) apart from "the
    listing call failed" (2) — reporting the first when it was really the
    second is how an install turns into a mystery.

    quants is a space-separated preference list; the first one present wins.
    """
    try:
        files = [f for f in _api().list_repo_files(repo) if f.endswith(".gguf")]
    except Exception as e:
        print(f"ERROR could not list {repo}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    base = lambda f: f.rsplit("/", 1)[-1].lower()
    if kind == "mmproj":
        cand = [f for f in files if base(f).startswith("mmproj")]
    elif kind == "draft":
        cand = [f for f in files if base(f).startswith("mtp")]
    else:
        # The main weights are whatever is neither the encoder nor the draft
        # head. Filtering by prefix rather than by quant name matters: the draft
        # head ships in the same quants as the model it drafts for.
        cand = [f for f in files
                if not base(f).startswith(("mmproj", "mtp"))]
    if not cand:
        print(f"ERROR {repo} has no '{kind}' GGUF", file=sys.stderr)
        print("available: " + ", ".join(files), file=sys.stderr)
        return 3
    for q in quants.split():
        hit = [f for f in cand if q.lower() in base(f)]
        if hit:
            print(sorted(hit, key=len)[0]); return 0
    if len(cand) == 1:
        # No preference matched, but there is only one file it could be. Some
        # repos publish an mmproj with no quant in the name at all; refusing
        # that on a technicality would be worse than taking the obvious answer.
        print(cand[0]); return 0
    if quants:
        print(f"ERROR none of [{quants}] in {repo} for '{kind}'", file=sys.stderr)
        print("available: " + ", ".join(cand), file=sys.stderr)
        return 3
    print(sorted(cand, key=len)[0]); return 0

def gated(repo):
    """0 if the repo is readable, 4 if gated/unauthorized, 2 on anything else."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    try:
        hf_hub_download(repo, "config.json")
        return 0
    except GatedRepoError:
        print(f"ERROR gated: {repo}", file=sys.stderr); return 4
    except RepositoryNotFoundError:
        print(f"ERROR not found (or requires auth): {repo}", file=sys.stderr); return 4
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}", file=sys.stderr); return 2

def file(repo, filename, dest):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, filename, local_dir=dest)
    print(p); return 0

def snapshot(repo):
    from huggingface_hub import snapshot_download
    print(snapshot_download(repo)); return 0

if __name__ == "__main__":
    fn = {"whoami": whoami, "pick": pick, "gated": gated,
          "file": file, "snapshot": snapshot}[sys.argv[1]]
    sys.exit(fn(*sys.argv[2:]))
PY

MODELS_DIR="$PREFIX/models"
mkdir -p "$MODELS_DIR"

# ask_model NAME SIZE DESCRIPTION -> 0 if the user wants it
ask_model() {
  say ""
  say "  ${B}$1${R}  ${DIM}~$2${R}"
  say "    $3"
  confirm "Download $1?"
}

MISSING_MODELS=()

if [[ $DO_MODELS -eq 0 ]]; then
  warn "--no-models: skipping all downloads"
  MISSING_MODELS+=("language model" "mmproj" "CSM-1B" "voice LoRA")
else
  HF_USER="$("$P_PY" "$HF_HELPER" whoami 2>/dev/null || true)"
  if [[ -n "$HF_USER" ]]; then ok "Hugging Face: logged in as $HF_USER"
  else warn "Hugging Face: not logged in — gated repos will be refused"
       note "run:  $P_VENV/bin/hf auth login"
  fi

  # fetch_gguf LABEL KIND QUANTS OUTVAR — resolve, download, record the path.
  # Resolution and download are separate failures with separate messages: one
  # means the repo does not carry the file, the other means the transfer broke.
  fetch_gguf() {
    local label="$1" kind="$2" quants="$3" outvar="$4" file rc
    if [[ "$MODE" == "dry-run" ]]; then
      note "would resolve the $kind GGUF in $P_LLM_REPO (quants: ${quants:-any}) and download it"
      return 0
    fi
    step "resolving the $kind file in $P_LLM_REPO"
    set +e
    file="$("$P_PY" "$HF_HELPER" pick "$P_LLM_REPO" "$kind" "$quants" 2>>"$LOG")"
    rc=$?
    set -e
    case $rc in
      0) ok "$file" ;;
      2) warn "could not reach Hugging Face to list $P_LLM_REPO"
         note "this is a network or auth problem, not a missing file — see $LOG"
         MISSING_MODELS+=("$label"); return 1 ;;
      *) warn "$P_LLM_REPO carries no $kind file matching [${quants:-any}]"
         note "the available filenames are listed in $LOG"
         note "override the repo with COMPANION_LLM_REPO=... ./install.sh"
         MISSING_MODELS+=("$label"); return 1 ;;
    esac
    step "downloading $file"
    if run "$P_PY" "$HF_HELPER" file "$P_LLM_REPO" "$file" "$MODELS_DIR"; then
      printf -v "$outvar" '%s' "$MODELS_DIR/$file"
      ok "${!outvar}"
    else
      warn "download of $file failed — see $LOG"
      MISSING_MODELS+=("$label"); return 1
    fi
  }

  # 1. the language model
  if ask_model "$P_LLM_REPO  [$P_QUANTS]" "3-8 GiB" \
     "The brain. Must accept audio natively — a speech-to-text stage in front of a text model throws away prosody, which is most of what was actually said."; then
    fetch_gguf "language model" main "$P_QUANTS" P_LLM_PATH || true
  else
    MISSING_MODELS+=("language model")
  fi

  # 2. mmproj — not optional, this is the audio encoder
  if ask_model "mmproj for $P_LLM_REPO" "200-400 MiB" \
     "REQUIRED for voice. This is the audio/vision encoder; without it the model is text-only and --voice cannot work at all."; then
    fetch_gguf "mmproj" mmproj "$P_MMPROJ_QUANTS" P_MMPROJ_PATH || true
  else
    warn "skipped — voice mode will not work without it"
    MISSING_MODELS+=("mmproj")
  fi

  # 3. speculative draft head — optional, and only some repos publish one
  if ask_model "MTP draft head for $P_LLM_REPO" "250-500 MiB" \
     "Optional. A matching multi-token-prediction head lets llama.cpp speculate ahead, which cuts latency. Skipping it costs speed, nothing else."; then
    fetch_gguf "MTP draft head" draft "$P_QUANTS" P_DRAFT_PATH || true
  fi

  # 4. CSM — gated
  if ask_model "$P_CSM_REPO" "6.7 GiB" \
     "The voice. GATED: you must accept the license at huggingface.co/$P_CSM_REPO first."; then
    step "checking access"
    if run "$P_PY" "$HF_HELPER" gated "$P_CSM_REPO"; then
      ok "access granted"
      step "downloading (large)"
      run "$P_PY" "$HF_HELPER" snapshot "$P_CSM_REPO" \
        || { warn "download failed — see $LOG"; MISSING_MODELS+=("CSM-1B"); }
      ok "in the Hugging Face cache"
    else
      warn "access refused for $P_CSM_REPO"
      note "1. accept the license:  https://huggingface.co/$P_CSM_REPO"
      note "2. log in:              $P_VENV/bin/hf auth login"
      note "3. re-run:              ./install.sh --no-torch --no-deps --no-llama"
      MISSING_MODELS+=("CSM-1B")
    fi
  else
    MISSING_MODELS+=("CSM-1B")
  fi

  # 5. the voice adapter
  if ask_model "$P_LORA_REPO" "54 MiB" \
     "A voice, as a LoRA over CSM — this is Sesame's Maya voice, credited in the README. Licence cc-by-nc-sa-4.0, non-commercial. Speaker id 4 only; a different id sounds wrong or garbled. Skipping this gives you base CSM, which works fine."; then
    run "$P_PY" "$HF_HELPER" snapshot "$P_LORA_REPO" \
      || { warn "download failed"; MISSING_MODELS+=("voice LoRA"); }
    ok "downloaded"
  else
    note "running without an adapter: drop the PeftModel lines in tts.py:load_model"
    MISSING_MODELS+=("voice LoRA")
  fi
fi

# --- configuration ----------------------------------------------------------
head1 "8. Configuration"

# Both names are optional here. Leaving either blank is not a half-finished
# install: companion.py asks on first run for whatever is still unset, which is
# the path anyone who clones and runs directly takes anyway. This just saves
# them the question if they happen to be standing here already.
if [[ -f "$SELF/.env" ]]; then
  ok ".env already exists — not touching it"
elif [[ "$MODE" != "dry-run" ]]; then
  say ""
  say "  ${DIM}Both optional — blank means you'll be asked on first run instead.${R}"
  read -r -p "  What should she call you? " OP_NAME </dev/tty || OP_NAME=""
  say "  ${DIM}And her name. There is no default: what to call her is yours to pick,"
  say "  not something this repo should decide for you. It goes into her own"
  say "  character prompt, so she uses it about herself.${R}"
  read -r -p "  What do you want to call her? " HER_NAME </dev/tty || HER_NAME=""
  {
    echo "# Written by install.sh on $(date -Is). Untracked, per-machine."
    [[ -n "$OP_NAME"  ]] && echo "COMPANION_USER=$OP_NAME"
    [[ -n "$HER_NAME" ]] && echo "COMPANION_NAME=$HER_NAME"
    [[ -n "$D_HSA_OVERRIDE" ]] && echo "HSA_OVERRIDE_GFX_VERSION=$D_HSA_OVERRIDE"
  } >"$SELF/.env"
  ok "wrote .env"
  [[ -z "$HER_NAME" ]] && note "she'll be named on first run"
fi

if [[ -f "$SELF/core_memory.md" ]]; then
  ok "core_memory.md exists"
else
  step "seeding core_memory.md from the example"
  [[ "$MODE" == "dry-run" ]] || cp "$SELF/core_memory.example.md" "$SELF/core_memory.md"
  ok "created — rewrite it as yourself; it is who she thinks you are"
fi

if [[ -f "$SELF/origin.md" ]]; then
  ok "origin.md exists"
else
  note "no origin.md — she will have no stated backstory, which is a valid choice"
  note "see origin.example.md if you want to write one"
fi

# --- generated launcher -----------------------------------------------------
# llama-server being an ad-hoc process is the single most common way this stack
# breaks: a reboot loses it, and companion.py's failure to connect looks like a bug
# in companion.py. This puts the exact flags for THIS machine somewhere durable.
LAUNCHER="$PREFIX/start-llama.sh"
step "writing $LAUNCHER"
if [[ "$MODE" != "dry-run" ]]; then
  {
    echo "#!/usr/bin/env bash"
    echo "# Generated by install.sh on $(date -Is) for ${D_GPU} (${D_GFX:-sm_$D_SM}), ${D_VRAM_GB} GiB."
    echo "# Every flag here was chosen by a measured failure — see BUILD.md section 4."
    echo "set -euo pipefail"
    echo ""
    if [[ "$D_VENDOR" == "amd" ]]; then
      echo "export HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES:-0}"
      [[ -n "$D_HSA_OVERRIDE" ]] && echo "export HSA_OVERRIDE_GFX_VERSION=$D_HSA_OVERRIDE"
    elif [[ "$D_VENDOR" == "nvidia" ]]; then
      echo "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}"
    fi
    echo ""
    echo "exec $D_LLAMA_BIN \\"
    echo "  -m ${P_LLM_PATH:-<set-me>.gguf} \\"
    echo "  --mmproj ${P_MMPROJ_PATH:-<set-me>-mmproj.gguf} \\"
    # The draft head is only emitted when one was actually downloaded — a
    # -md pointing at a file that isn't there stops the server from starting,
    # which is a worse outcome than not speculating.
    [[ -n "${P_DRAFT_PATH:-}" ]] && echo "  -md ${P_DRAFT_PATH} --spec-type draft-mtp \\"
    echo "  -ngl 99 -fa on -c $P_CTX -ub 512 \\"
    echo "  --host 127.0.0.1 --port 8080 \\"
    echo "  --reasoning off --parallel 1 \\"
    echo "  --temp 0.6 --top-k 64 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 \\"
    echo "  \"\$@\""
  } >"$LAUNCHER"
  chmod +x "$LAUNCHER"
fi
ok "written"
note "--reasoning off and --parallel 1 are load-bearing: 8x and 777ms respectively"
note "-c $P_CTX was sized to your VRAM. Raising it starves CSM and stalls speech mid-word."

# ===========================================================================
# PHASE 4 — VERIFY
# ===========================================================================

head1 "9. Verification"

VERIFY_FAIL=0
vrow() { row "$1" "$2" "$3" "$4" "$5"; }
ROWS=()

if [[ "$MODE" == "dry-run" ]]; then
  say "  ${DIM}--dry-run: nothing was built, so nothing to verify.${R}"
else
  # torch sees the GPU
  TORCH_INFO="$("$P_PY" - <<'PY' 2>>"$LOG" || echo "FAIL"
import torch, json
d = {"torch": torch.__version__,
     "hip": getattr(torch.version, "hip", None),
     "cuda": torch.version.cuda,
     "avail": torch.cuda.is_available()}
if d["avail"]:
    p = torch.cuda.get_device_properties(0)
    d["dev"] = p.name
    d["arch"] = getattr(p, "gcnArchName", "") or f"sm_{p.major}{p.minor}"
    d["vram"] = round(p.total_memory / 2**30, 1)
print(json.dumps(d))
PY
)"
  if [[ "$TORCH_INFO" == "FAIL" ]]; then
    vrow MISS "torch import" "succeeds" "failed" "see $LOG"
    VERIFY_FAIL=1
  else
    TV="$("$P_PY" -c "import json,sys;print(json.loads(sys.argv[1])['torch'])" "$TORCH_INFO")"
    TA="$("$P_PY" -c "import json,sys;d=json.loads(sys.argv[1]);print(d.get('arch','-') if d['avail'] else 'no GPU')" "$TORCH_INFO")"
    if grep -q '"avail": true' <<<"$TORCH_INFO"; then
      vrow OK "torch" "imports + sees GPU" "$TV" "device arch: $TA"
    else
      vrow WARN "torch" "imports + sees GPU" "$TV" "no GPU visible to torch — synthesis will run on CPU, far slower than real time"
    fi
  fi

  # a real matmul, not just an import
  if "$P_PY" - <<'PY' >>"$LOG" 2>&1
import torch
if torch.cuda.is_available():
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    assert torch.isfinite(a @ a).all()
PY
  then vrow OK "GPU compute" "matmul runs" "ok" ""
  else vrow MISS "GPU compute" "matmul runs" "failed" "kernels do not run on this target — check HSA_OVERRIDE_GFX_VERSION"; VERIFY_FAIL=1
  fi

  # transformers + peft
  if "$P_PY" -c "import transformers, peft, httpx, numpy, scipy" >>"$LOG" 2>&1; then
    TFV="$("$P_PY" -c "import transformers;print(transformers.__version__)")"
    vrow OK "dependencies" "import" "transformers $TFV" ""
  else
    vrow MISS "dependencies" "import" "failed" "see $LOG"; VERIFY_FAIL=1
  fi

  # silero VAD file, which is what listen.py actually loads
  if "$P_PY" - <<'PY' >>"$LOG" 2>&1
import os, silero_vad, torch
p = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.jit")
assert os.path.exists(p), p
torch.jit.load(p, map_location="cpu")
PY
  then vrow OK "Silero VAD" "loadable .jit" "ok" ""
  else vrow WARN "Silero VAD" "loadable .jit" "missing" "voice mode unavailable; typed mode still works"
  fi

  # llama-server binary
  if [[ -x "$D_LLAMA_BIN" ]]; then
    LV="$("$D_LLAMA_BIN" --version 2>&1 | head -1 | cut -c1-40 || echo built)"
    vrow OK "llama-server" "runs" "$LV" "$D_LLAMA_BIN"
  else
    vrow MISS "llama-server" "runs" "not built" ""; VERIFY_FAIL=1
  fi

  # capture device
  if have arecord && arecord -l 2>/dev/null | grep -q '^card'; then
    vrow OK "microphone" "a capture device" "present" "run 'python miccheck.py' before first use — most install problems start here"
  else
    vrow WARN "microphone" "a capture device" "none" "typed mode only"
  fi
fi

print_rows

# --- GEMV benchmark ---------------------------------------------------------
if [[ "$P_GEMV" == "benchmark" && "$MODE" != "dry-run" && $VERIFY_FAIL -eq 0 ]]; then
  head1 "10. Batch-1 GEMV benchmark"
  say "  ${DIM}Measuring on your card rather than assuming. These are CSM's actual"
  say "  decode shapes; on gfx1030 rocBLAS is up to 18x slower than einsum here.${R}"
  say ""
  GEMV_RESULT="$("$P_PY" - <<'PY' 2>>"$LOG" || echo "ERROR"
import torch, time
if not torch.cuda.is_available():
    print("SKIP"); raise SystemExit
shapes = [("down_proj", 8192, 1024), ("up/gate", 1024, 8192),
          ("q/o_proj", 1024, 1024), ("kv_proj", 1024, 256)]
def bench(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6
wins = 0
for name, K, N in shapes:
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(K, device="cuda", dtype=torch.float16)
    a = bench(lambda: torch.nn.functional.linear(x, W))
    b = bench(lambda: torch.einsum("ij,j->i", W, x))
    wins += b < a
    print(f"  {name:<10} [1,{K}]@[{K},{N}]  rocBLAS/cuBLAS {a:7.1f}us   einsum {b:7.1f}us   {a/b:5.2f}x")
print("VERDICT " + ("on" if wins >= 3 else "off"))
PY
)"
  if [[ "$GEMV_RESULT" == "ERROR" || "$GEMV_RESULT" == "SKIP" ]]; then
    warn "benchmark did not run — leaving fast_gemv at its default (on)"
  else
    printf '%s\n' "$GEMV_RESULT" | grep -v VERDICT
    log "$GEMV_RESULT"
    if grep -q "VERDICT on" <<<"$GEMV_RESULT"; then
      say ""
      ok "einsum wins on this card — fast_gemv stays enabled"
      grep -q '^COMPANION_NO_FAST_GEMV' "$SELF/.env" 2>/dev/null \
        && sed -i '/^COMPANION_NO_FAST_GEMV/d' "$SELF/.env"
    else
      say ""
      ok "the stock kernels win on this card — disabling fast_gemv"
      note "this is expected on cuBLAS and on newer AMD architectures"
      grep -q '^COMPANION_NO_FAST_GEMV' "$SELF/.env" 2>/dev/null \
        || echo "COMPANION_NO_FAST_GEMV=1" >>"$SELF/.env"
    fi
  fi
fi

# ===========================================================================
# SUMMARY
# ===========================================================================

head1 "Done"

if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
  warn "Not everything was downloaded. Still missing:"
  for m in "${MISSING_MODELS[@]}"; do note "· $m"; done
  note "re-run to be asked again — everything already done is skipped:"
  note "  ./install.sh --no-torch --no-deps --no-llama"
  say ""
fi

if [[ $VERIFY_FAIL -ne 0 ]]; then
  die 5 "Verification failed. The install is incomplete." \
    "Re-run the checks on their own at any time:" \
    "  ./install.sh --check" \
    "Full log: $LOG"
fi

say "  ${B}Start the language server${R} (leave it running; it is not a service):"
say "      $LAUNCHER"
say ""
say "  ${B}Check your microphone${R} — more install problems come from here than anywhere else:"
say "      $P_PY $SELF/miccheck.py"
say ""
say "  ${B}Talk to her${R}:"
say "      $P_PY $SELF/companion.py --voice --timing"
say "      $P_PY $SELF/companion.py            ${DIM}# typed, Enter alone for push-to-talk${R}"
say ""
say "  ${DIM}Voice mode assumes headphones — on speakers the mic hears her and"
say "  barge-in becomes a feedback loop.${R}"
say ""
say "  ${DIM}Health check at any time:  ./install.sh --check${R}"
say "  ${DIM}Full log:                  $LOG${R}"
say ""

rm -f "$HF_HELPER"
log "install.sh finished $(date -Is)"
