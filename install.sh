#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kyudong Kim
#
# LapMoE installer
#
# One command that sets up a local LLM server able to run a mid-size Mixture-of-Experts model
# (Qwen3.6-35B-A3B) on a modest PC or laptop (e.g. an 8 GB NVIDIA GPU + 16 GB RAM):
#   1. builds llama.cpp for your NVIDIA GPU (CUDA)      -> $PREFIX/src/llama.cpp/build/bin
#   2. installs llama-swap (OpenAI-compatible router)  -> $PREFIX/bin/llama-swap
#   3. downloads a GGUF quant chosen for your RAM      -> $PREFIX/models
#   4. writes $PREFIX/config.yaml, start.sh, test.sh   (and optionally a systemd user service)
#
# Linux only (x86_64 / aarch64), NVIDIA GPU required (no CPU-only mode). Needs no root unless you pass --install-deps.
# Third-party components keep their own licenses: llama.cpp (MIT), llama-swap (MIT),
# Qwen3.6 weights via Unsloth GGUF (Apache-2.0). Run with --help for options.
#
set -Eeuo pipefail

VERSION="0.1.0"

# ---------------------------------------------------------------------------------------------
# Defaults (override with flags or environment variables)
# ---------------------------------------------------------------------------------------------
PREFIX="${PREFIX:-$HOME/lapmoe}"
BACKEND="${BACKEND:-auto}"                 # auto (= cuda) | cuda | vulkan (experimental, untested)
PROFILE="${PROFILE:-auto}"                 # auto | mid | small
QUANT="${QUANT:-}"                         # e.g. UD-IQ3_XXS (mid profile); empty = choose by RAM
MODEL_PATH="${MODEL_PATH:-}"               # use an existing GGUF instead of downloading
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
# Validated build: upstream commit a01eb2646 (build 9612) -- the one all measurements were made with.
LLAMA_REF="${LLAMA_REF:-a01eb26468a6cb2bf2c3bfd1cc3b5d9e64daefcd}"
SWAP_VERSION="${SWAP_VERSION:-v250}"      # tag, or "latest"
PORT="${PORT:-8080}"
CTX="${CTX:-}"                             # context size override
HF_TOKEN="${HF_TOKEN:-}"                   # optional, only needed for gated/private repos
JOBS="${JOBS:-}"                           # parallel compile jobs (default: min(cores, RAM/3 GiB))
DO_BUILD=1; DO_SWAP=1; DO_MODEL=1
DO_SYSTEMD=0; INSTALL_DEPS=0; DRY_RUN=0; ASSUME_YES=0; FORCE=0; DEMO=0

# Model presets ---------------------------------------------------------------------------------
MID_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
MID_PREFIX="Qwen3.6-35B-A3B"
# quant -> approx file size in MiB (used only to choose; exact size/sha256 come from Hugging Face)
MID_ORDER=("UD-Q4_K_M" "UD-Q3_K_M" "UD-IQ3_XXS" "UD-Q2_K_XL" "UD-IQ2_XXS")
declare -A MID_MB=( ["UD-Q4_K_M"]=21109 ["UD-Q3_K_M"]=15831 ["UD-IQ3_XXS"]=12599 ["UD-Q2_K_XL"]=11730 ["UD-IQ2_XXS"]=10300 )
SMALL_REPO="unsloth/Qwen3-8B-GGUF"
SMALL_PREFIX="Qwen3-8B"
RAM_HEADROOM_MB=3072                       # OS + desktop + KV/compute buffers kept free of the model

# ---------------------------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then B=$'\033[1;34m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=; G=; Y=; R=; N=; fi
# ---- mascot: ASCII banner + expression faces (TTY only; --plain, NO_COLOR or a pipe turn it all off) ----
PLAIN="${PLAIN:-0}"; [ -t 1 ] || PLAIN=1; [ -n "${NO_COLOR:-}" ] && PLAIN=1
C_BOLD=$'\033[1m'; C_CYAN=$'\033[36m'; C_BLUE=$'\033[1;34m'
[ "$PLAIN" = 1 ] && { C_BOLD=; C_CYAN=; C_BLUE=; }
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in *UTF-8*|*utf8*|*UTF8*) KO=1;; *) KO=0;; esac

banner() {
  [ "$PLAIN" = 1 ] && return 0
  local cat=('    /\_/\    ' '   ( o.o )   ' '    > ^ <    ' '   /|___|\   ' '  /_______\  ' '             ')
  local lap=(' _                 ' '| |    __ _ _ __   ' "| |   / _\` | '_ \\  " '| |__| (_| | |_) | ' '|_____\__,_| .__/  ' '           |_|     ')
  local moe=(' __  __       _____ ' '|  \/  | ___ | ____|' '| |\/| |/ _ \|  _|  ' '| |  | | (_) | |___ ' '|_|  |_|\___/|_____|' '                    ')
  local i; printf '\n'
  for i in 0 1 2 3 4 5; do
    printf '%s%s%s%s%s%s%s%s\n' "$C_CYAN" "${cat[$i]}" "$N" "$C_BOLD" "${lap[$i]}" "$C_BLUE" "${moe[$i]}" "$N"
  done
  printf '             %sBig models. Small laptops. No excuses.%s\n\n' "$C_BOLD" "$N"
}

# face <cando|happy|tired|sleepy|surprised|angry> <message>: the mascot's expression sheet, as emoticons
face() {
  if [ "$PLAIN" = 1 ]; then printf '   %s\n' "$2"; return 0; fi
  local f k
  case "$1" in
    cando)     f='(^_^)b';    [ "$KO" = 1 ] && k='할 수 있어!' || k="I can do this!";;
    happy)     f='\(^o^)/';   [ "$KO" = 1 ] && k='신난다!'     || k="yay!";;
    tired)     f='(>_<;)';    [ "$KO" = 1 ] && k='힘들어…'     || k="so heavy...";;
    sleepy)    f='(-_-)zZ';   [ "$KO" = 1 ] && k='졸려…'       || k="sleepy...";;
    surprised) f='(O_O;)?!';  [ "$KO" = 1 ] && k='어?!'        || k="huh?!";;
    angry)     f='(-_-#)';    [ "$KO" = 1 ] && k='화난다!'     || k="grr!";;
  esac
  printf '  %s%-9s%s %s%s%s  %s\n' "$C_CYAN" "$f" "$N" "$C_BOLD" "$k" "$N" "$2"
}

done_art() {
  [ "$PLAIN" = 1 ] && return 0
  printf '%s\n' '' '        /\_/\' '       ( ^.^ )    all set!' '       /|___|\' '      /_______\' '     |___________|' "     '-----------'" ''
}

# --demo: preview every piece of ASCII art and every expression, plus a terminal check. Installs nothing.
demo_art() {
  PLAIN=0
  local tty utf8 color width h
  tty="yes"; [ -t 1 ] || tty="no (output is piped, so colours are off)"
  utf8="yes"; [ "$KO" = 1 ] || utf8="no (LANG='${LANG:-}') -> English expression names are used"
  color="yes"; [ -n "$C_BLUE" ] || color="no (not a terminal, or NO_COLOR is set)"
  width="$(tput cols 2>/dev/null || echo '?')"
  printf '%s== terminal check ==%s\n' "$C_BOLD" "$N"
  printf '   interactive terminal : %s\n   UTF-8 / Korean text  : %s\n   colours              : %s\n   width                : %s columns (the banner needs 54)\n' "$tty" "$utf8" "$color" "$width"
  printf '\n%s== banner (shown when the installer starts) ==%s\n' "$C_BOLD" "$N"; banner
  printf '%s== expressions (the mascot reacts to what the installer is doing) ==%s\n' "$C_BOLD" "$N"
  face cando     "start:      let's set this laptop up"
  face tired     "building:   compiling llama.cpp (10-25 min, the laptop gets warm)"
  face sleepy    "download:   a 13 GB model, safe to interrupt and resume"
  face surprised "warning:    something looks odd but the install can continue"
  face angry     "checksum:   the file does not match what the server publishes"
  face happy     "finished:   everything is installed"
  printf '\n%s== warnings and errors ==%s\n' "$C_BOLD" "$N"
  warn "sample warning: this quant is larger than the RAM you can spare"
  ( die "sample error: this line is only printed by --demo" ) 2>&1 || true
  printf '\n%s== finish ==%s' "$C_BOLD" "$N"; done_art
  printf '%s== the same faces with English names (what a non-UTF-8 terminal sees) ==%s\n' "$C_BOLD" "$N"
  ( KO=0; for h in cando tired sleepy surprised angry happy; do face "$h" "$h"; done )
  printf '\n%s== plain mode (logs, pipes, --plain, NO_COLOR) ==%s\n' "$C_BOLD" "$N"
  ( PLAIN=1; log "LapMoE installer"; face tired "compiling llama.cpp"; face happy "everything is installed" )
  printf '\n   This was only a preview: nothing was installed or changed.\n'
}

log()  { printf '%s==>%s %s\n' "$B" "$N" "$*"; }
ok()   { printf ' %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { if [ "$PLAIN" = 1 ]; then printf ' ! %s\n' "$*" >&2; else printf ' %s(O_O;)?!%s %s\n' "$Y" "$N" "$*" >&2; fi; }
die()  { [ "$PLAIN" = 1 ] || printf '%s(O_O;)?! %s%s\n' "$R" "$([ "$KO" = 1 ] && echo '어?!' || echo 'huh?!')" "$N" >&2; printf '%sERROR:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
trap 'printf "%sERROR:%s failed at line %s (exit %s). Re-running the script resumes where it stopped.\n" "$R" "$N" "$LINENO" "$?" >&2' ERR

run() { if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] %s\n' "$*"; else "$@"; fi; }
have() { command -v "$1" >/dev/null 2>&1; }
confirm() {
  [ "$ASSUME_YES" = 1 ] && return 0
  [ -t 0 ] || die "$1 -- not a terminal; re-run with --yes to accept"
  local a; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]
}

usage() {
cat <<EOF
LapMoE installer v$VERSION

Usage: ./install.sh [options]

  --prefix DIR           install location                          (default: \$HOME/lapmoe)
  --backend NAME         auto | cuda | vulkan (experimental)       (default: auto = cuda)
  --profile NAME         auto | mid | small                        (default: auto = by RAM)
  --quant NAME           mid-profile quant, e.g. UD-IQ3_XXS        (default: chosen by RAM)
  --model-path FILE      use an existing GGUF instead of downloading
  --ctx N                context size override
  --jobs N               parallel compile jobs                     (default: min(cores, RAM/3GiB))
  --port N               llama-swap port                           (default: 8080)
  --llama-ref REF        llama.cpp commit/tag to build             (default: validated commit)
  --llama-repo URL       llama.cpp git URL                         (default: ggml-org/llama.cpp)
  --swap-version TAG     llama-swap release tag or "latest"        (default: $SWAP_VERSION)
  --hf-token TOKEN       Hugging Face token (or env HF_TOKEN)
  --no-build | --no-swap | --no-model    skip a step
  --systemd              install + start a systemd *user* service
  --install-deps         apt-get install build prerequisites (uses sudo; Debian/Ubuntu)
  --plain                no ASCII art / colours (automatic when output is not a terminal)
  --demo                 preview the ASCII art and mascot expressions, then exit (installs nothing)
  --dry-run              show what would happen, change nothing
  -y, --yes              do not ask for confirmation
  --force                redo steps even if their output already exists
  -h, --help
EOF
}

# ---------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2;;        --backend) BACKEND="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;      --quant) QUANT="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;; --ctx) CTX="$2"; shift 2;;    --jobs) JOBS="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;            --llama-ref) LLAMA_REF="$2"; shift 2;;
    --llama-repo) LLAMA_REPO="$2"; shift 2;; --swap-version) SWAP_VERSION="$2"; shift 2;;
    --hf-token) HF_TOKEN="$2"; shift 2;;
    --no-build) DO_BUILD=0; shift;;         --no-swap) DO_SWAP=0; shift;;     --no-model) DO_MODEL=0; shift;;
    --systemd) DO_SYSTEMD=1; shift;;        --install-deps) INSTALL_DEPS=1; shift;;
    --plain) PLAIN=1; shift;;                --demo|--art-test) DEMO=1; shift;;                --dry-run) DRY_RUN=1; shift;;           -y|--yes) ASSUME_YES=1; shift;;   --force) FORCE=1; shift;;
    -h|--help) usage; exit 0;;
    *) usage >&2; die "unknown option: $1";;
  esac
done
[ "$DEMO" = 1 ] && { demo_art; exit 0; }
case "$BACKEND" in auto|cuda|vulkan) ;; *) die "--backend must be auto|cuda|vulkan (there is no CPU-only mode)";; esac
case "$PROFILE" in auto|mid|small) ;; *) die "--profile must be auto|mid|small";; esac
[[ "$PORT" =~ ^[0-9]+$ ]] || die "--port must be a number"

PREFIX="$(mkdir -p "$(dirname "$PREFIX")" 2>/dev/null; cd "$(dirname "$PREFIX")" 2>/dev/null && pwd)/$(basename "$PREFIX")"
SRC_DIR="$PREFIX/src/llama.cpp"; BIN_DIR="$PREFIX/bin"; MODELS_DIR="$PREFIX/models"; DL_DIR="$PREFIX/downloads"
LLAMA_BIN="$SRC_DIR/build/bin/llama-server"

# ---------------------------------------------------------------------------------------------
# 0. Detect hardware, decide backend / profile / quant
# ---------------------------------------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || die "Linux only (found $(uname -s))"
ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in x86_64) SWAP_ARCH=amd64;; aarch64|arm64) SWAP_ARCH=arm64;; *) die "unsupported CPU architecture: $ARCH_RAW";; esac

RAM_MB="$(awk '/MemTotal/{printf "%d",$2/1024}' /proc/meminfo)"
CORES="$(nproc)"
GPU_NAME=""; VRAM_MB=0; CUDA_CC=""
if have nvidia-smi; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"; VRAM_MB="${VRAM_MB:-0}"
  CUDA_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr -d '. ' | paste -sd';' - || true)"
fi
# Test hooks: simulate other hardware to check the selection logic (use with --dry-run).
RAM_MB="${LSM_TEST_RAM_MB:-$RAM_MB}"; GPU_NAME="${LSM_TEST_GPU-$GPU_NAME}"; VRAM_MB="${LSM_TEST_VRAM_MB-$VRAM_MB}"
NVCC=""
if have nvcc; then NVCC="$(command -v nvcc)"; elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc; fi
HAS_VULKAN=0; if have pkg-config && pkg-config --exists vulkan 2>/dev/null && have glslc; then HAS_VULKAN=1; fi

if [ "$BACKEND" = auto ]; then BACKEND=cuda; fi
if [ "$BACKEND" = cuda ]; then
  [ -n "$GPU_NAME" ] || die "no NVIDIA GPU found (nvidia-smi is missing or reports nothing). LapMoE needs an NVIDIA GPU with up-to-date drivers; there is no CPU-only mode. (--backend vulkan is experimental and untested.)"
  if [ -z "$NVCC" ] && [ "$DO_BUILD" = 1 ]; then
    die "NVIDIA GPU found ($GPU_NAME) but no CUDA toolkit (nvcc). Install it from https://developer.nvidia.com/cuda-downloads (or apt install nvidia-cuda-toolkit), then re-run."
  fi
fi
if [ "$BACKEND" = vulkan ] && [ "$HAS_VULKAN" = 0 ] && [ "$DO_BUILD" = 1 ]; then
  die "--backend vulkan needs the Vulkan development files and glslc (apt install libvulkan-dev glslc)"
fi

USABLE_MB=$(( RAM_MB - RAM_HEADROOM_MB ))
if [ "$PROFILE" = auto ]; then
  PROFILE=small
  if [ -n "$QUANT" ]; then PROFILE=mid
  else for q in "${MID_ORDER[@]}"; do [ "${MID_MB[$q]}" -le "$USABLE_MB" ] && { PROFILE=mid; QUANT="$q"; break; }; done
  fi
fi
if [ "$PROFILE" = mid ]; then
  [ -n "$QUANT" ] || { for q in "${MID_ORDER[@]}"; do [ "${MID_MB[$q]}" -le "$USABLE_MB" ] && { QUANT="$q"; break; }; done; QUANT="${QUANT:-UD-IQ2_XXS}"; }
  MODEL_REPO="$MID_REPO"; MODEL_FILE="$MID_PREFIX-$QUANT.gguf"
  [ -n "${MID_MB[$QUANT]:-}" ] && [ "${MID_MB[$QUANT]}" -gt "$USABLE_MB" ] && warn "$QUANT (~$(( MID_MB[$QUANT] / 1024 )) GiB) is larger than the ~$(( USABLE_MB / 1024 )) GiB of RAM this machine can spare: expect heavy paging/slowness. Pick a smaller --quant."
else
  QUANT="UD-Q3_K_XL"; [ "$USABLE_MB" -lt 5500 ] && QUANT="UD-IQ3_XXS"
  MODEL_REPO="$SMALL_REPO"; MODEL_FILE="$SMALL_PREFIX-$QUANT.gguf"
fi
[ -n "$MODEL_PATH" ] && { MODEL_FILE="$(basename "$MODEL_PATH")"; DO_MODEL=0; }

# Offload recommendation. The "tuned" preset is the exact setup measured on an RTX 2070 Super Max-Q (8 GB) + 16 GB RAM.
TUNED_OK=0
if [ "$PROFILE" = mid ] && [ "$QUANT" = UD-IQ3_XXS ] && [ "$BACKEND" = cuda ] && [ "$VRAM_MB" -ge 7000 ] && [ "$VRAM_MB" -le 9300 ] && [ "$RAM_MB" -ge 15000 ]; then TUNED_OK=1; fi

banner
log "LapMoE installer v$VERSION$([ "$DRY_RUN" = 1 ] && echo '  [DRY RUN]')"
cat <<EOF
   machine : ${RAM_MB} MiB RAM, ${CORES} cores, GPU: ${GPU_NAME:-none} (${VRAM_MB} MiB)$( [ -n "$GPU_NAME" ] && [ -n "$CUDA_CC" ] && echo ", CUDA cc $CUDA_CC")
   backend : $BACKEND$( [ "$BACKEND" = cuda ] && echo "  (nvcc: $NVCC)")
   model   : $MODEL_REPO / $MODEL_FILE   (profile: $PROFILE)
   offload : $( [ "$TUNED_OK" = 1 ] && echo "8 GB-class validated preset available (n-cpu-moe 28, -ub 2048, ctx 131072)" || echo "automatic (llama.cpp --fit)")
   install : $PREFIX      port: $PORT
   steps   : build=$DO_BUILD swap=$DO_SWAP model=$DO_MODEL systemd=$DO_SYSTEMD
EOF
face cando "let's set this laptop up"
[ "$DRY_RUN" = 1 ] || { [ "$ASSUME_YES" = 1 ] || confirm "Proceed?" || die "aborted"; }

# ---------------------------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------------------------
if [ "$INSTALL_DEPS" = 1 ]; then
  have apt-get || die "--install-deps supports apt (Debian/Ubuntu) only; install: git cmake a C++17 compiler make curl python3 tar"
  log "installing build prerequisites with sudo apt-get"
  run sudo apt-get update
  run sudo apt-get install -y build-essential cmake git curl ca-certificates python3 tar
fi
missing=()
for c in curl python3 tar sha256sum; do have "$c" || missing+=("$c"); done
if [ "$DO_BUILD" = 1 ]; then for c in git cmake make g++; do have "$c" || missing+=("$c"); done; fi
if [ ${#missing[@]} -gt 0 ] && [ "$DRY_RUN" = 0 ]; then
  die "missing tools: ${missing[*]}  -> on Debian/Ubuntu: sudo apt-get install -y build-essential cmake git curl python3  (or re-run with --install-deps)"
fi
[ ${#missing[@]} -gt 0 ] && warn "missing tools (would stop a real run): ${missing[*]}"
run mkdir -p "$PREFIX" "$BIN_DIR" "$MODELS_DIR" "$DL_DIR"

# ---------------------------------------------------------------------------------------------
# 2. llama.cpp
# ---------------------------------------------------------------------------------------------
build_llama() {
  log "llama.cpp @ ${LLAMA_REF:0:12} ($BACKEND)"
  if [ -x "$LLAMA_BIN" ] && [ "$FORCE" = 0 ]; then ok "already built: $LLAMA_BIN (use --force to rebuild)"; return; fi
  run mkdir -p "$SRC_DIR"
  if [ "$DRY_RUN" = 1 ]; then
    printf '   [dry-run] git fetch --depth 1 %s %s && cmake -B build ... && cmake --build build -j<N>\n' "$LLAMA_REPO" "$LLAMA_REF"; return
  fi
  if [ ! -d "$SRC_DIR/.git" ]; then git -C "$SRC_DIR" init -q; git -C "$SRC_DIR" remote add origin "$LLAMA_REPO"; fi
  git -C "$SRC_DIR" remote set-url origin "$LLAMA_REPO"
  git -C "$SRC_DIR" fetch --depth 1 origin "$LLAMA_REF"
  git -C "$SRC_DIR" checkout -q --force FETCH_HEAD
  # The server runs with --no-webui, so skip the embedded Web UI: no Node.js/npm and no extra downloads at build time.
  local flags=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF)
  case "$BACKEND" in
    cuda)   flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$NVCC"); [ -n "$CUDA_CC" ] && flags+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_CC");;
    vulkan) flags+=(-DGGML_VULKAN=ON);;
  esac
  # CUDA compile jobs need ~3 GiB RAM each: cap parallelism so a 16 GB machine does not swap itself to death
  local jobs=$(( RAM_MB / 3000 )); [ "$jobs" -gt "$CORES" ] && jobs="$CORES"; [ "$jobs" -lt 1 ] && jobs=1
  [ -n "$JOBS" ] && jobs="$JOBS"
  face tired "compiling llama.cpp with $jobs parallel jobs (10-25 min the first time; the laptop will get warm)"
  [ "$BACKEND" = cuda ] && export CUDACXX="$NVCC" PATH="$(dirname "$NVCC"):$PATH"
  cmake -S "$SRC_DIR" -B "$SRC_DIR/build" "${flags[@]}"
  cmake --build "$SRC_DIR/build" --config Release -j "$jobs" --target llama-server llama-cli llama-bench
  [ -x "$LLAMA_BIN" ] || die "build finished but $LLAMA_BIN is missing"
  ok "built: $("$LLAMA_BIN" --version 2>&1 | grep -m1 version || echo "$LLAMA_BIN")"
}

# ---------------------------------------------------------------------------------------------
# 3. llama-swap
# ---------------------------------------------------------------------------------------------
install_swap() {
  log "llama-swap $SWAP_VERSION"
  if [ -x "$BIN_DIR/llama-swap" ] && [ "$FORCE" = 0 ]; then ok "already installed: $("$BIN_DIR/llama-swap" -version 2>&1 | head -1) (use --force to reinstall)"; return; fi
  if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] download llama-swap %s linux_%s from github.com/mostlygeek/llama-swap, verify sha256, extract to %s\n' "$SWAP_VERSION" "$SWAP_ARCH" "$BIN_DIR"; return; fi
  local api="https://api.github.com/repos/mostlygeek/llama-swap/releases"
  [ "$SWAP_VERSION" = latest ] && api="$api/latest" || api="$api/tags/$SWAP_VERSION"
  local info; info="$(curl -fsSL "$api")" || die "cannot query $api (rate limited? try again later or set --swap-version)"
  local line; line="$(printf '%s' "$info" | python3 -I -c '
import json, sys
d = json.load(sys.stdin); arch = sys.argv[1]
for a in d.get("assets", []):
    n = a["name"]
    if n.endswith(f"_linux_{arch}.tar.gz"):
        print(d["tag_name"], n, a["browser_download_url"], (a.get("digest") or "").replace("sha256:", "")); break
' "$SWAP_ARCH")"
  [ -n "$line" ] || die "no linux_$SWAP_ARCH asset in release $SWAP_VERSION"
  local tag name url sum; read -r tag name url sum <<<"$line"
  curl -fL --retry 3 "${PROG[@]}" -o "$DL_DIR/$name" "$url"
  if [ -n "$sum" ]; then
    [ "$(sha256sum "$DL_DIR/$name" | cut -d' ' -f1)" = "$sum" ] || { rm -f "$DL_DIR/$name"; die "sha256 mismatch for $name"; }
    ok "sha256 verified ($tag)"
  else warn "release has no digest field; skipped checksum verification"; fi
  tar -xzf "$DL_DIR/$name" -C "$BIN_DIR" llama-swap
  chmod +x "$BIN_DIR/llama-swap"
  ok "installed: $("$BIN_DIR/llama-swap" -version 2>&1 | head -1)"
}

# ---------------------------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------------------------
PROG=(-sS); [ -t 2 ] && PROG=(-#)             # progress bar only on a terminal, quiet in logs
hf_curl() { if [ -n "$HF_TOKEN" ]; then curl -fL -H "Authorization: Bearer $HF_TOKEN" "$@"; else curl -fL "$@"; fi; }

fetch_model() {
  log "model $MODEL_REPO / $MODEL_FILE"
  local dest="$MODELS_DIR/$MODEL_FILE"
  if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] download -> %s (resumable), verify size + sha256 from the Hugging Face API\n' "$dest"; return; fi
  local meta; meta="$(hf_curl -sS "https://huggingface.co/api/models/$MODEL_REPO/tree/main" | python3 -I -c '
import json, sys
for f in json.load(sys.stdin):
    if f["path"] == sys.argv[1]:
        print(f.get("size", 0), (f.get("lfs") or {}).get("oid", "")); break
' "$MODEL_FILE")" || true
  [ -n "$meta" ] || die "$MODEL_FILE not found in $MODEL_REPO (check --quant / network / --hf-token)"
  local size sha; read -r size sha <<<"$meta"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ] && [ "$FORCE" = 0 ]; then
    ok "already downloaded ($(( size / 1048576 )) MiB); verifying checksum"
  else
    local need=$(( size + 1073741824 )) avail
    avail=$(( $(df -Pk "$MODELS_DIR" | awk 'NR==2{print $4}') * 1024 ))
    [ "$avail" -ge "$need" ] || die "not enough disk space in $MODELS_DIR: need $(( need / 1048576 )) MiB, have $(( avail / 1048576 )) MiB"
    confirm "Download $(( size / 1048576 )) MiB ($MODEL_FILE)?" || die "aborted"
    face sleepy "downloading $(( size / 1048576 )) MiB - safe to interrupt, re-running resumes"
    hf_curl -C - --retry 5 --retry-delay 5 "${PROG[@]}" -o "$dest.part" "https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE"
    mv "$dest.part" "$dest"
  fi
  [ "$(stat -c %s "$dest")" = "$size" ] || die "size mismatch for $dest"
  if [ -n "$sha" ]; then
    log "verifying sha256 (takes a minute)"
    [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ] || { rm -f "$dest"; face angry "sha256 does not match what Hugging Face publishes"; die "sha256 mismatch: file removed, re-run to download again"; }
    ok "sha256 verified"
  else warn "no checksum available from the API; only the size was verified"; fi
}

# ---------------------------------------------------------------------------------------------
# 5. llama-swap config + helper scripts
# ---------------------------------------------------------------------------------------------
esc() { printf '%s' "$1" | sed -e 's/[\/&|]/\\&/g'; }

write_config() {
  log "writing config"
  local cfg="$PREFIX/config.yaml" ctx_auto="${CTX:-65536}" model_file="$MODEL_FILE" mdir="$MODELS_DIR"
  [ -n "$MODEL_PATH" ] && { mdir="$(dirname "$MODEL_PATH")"; }
  if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] write %s, start.sh, test.sh\n' "$cfg"; return; fi
  [ -f "$cfg" ] && [ "$FORCE" = 0 ] && { cp -p "$cfg" "$cfg.bak.$(date +%Y%m%d_%H%M%S)"; warn "existing config backed up (.bak.*) and replaced"; }
  local default="mid"; [ "$PROFILE" = small ] && default="small"; [ "$TUNED_OK" = 1 ] && default="mid-8gb-tuned"
  {
    cat <<'EOF'
# Generated by LapMoE install.sh -- edit freely; re-running the installer backs this file up.
healthCheckTimeout: 300
logLevel: info
startPort: 9100

macros:
  llama_bin: "@@LLAMA_BIN@@"
  models_dir: "@@MODELS_DIR@@"
  # KV cache in q4_0 keeps long contexts cheap; --host 127.0.0.1 keeps the server private to this machine.
  common_flags: >
    --jinja --flash-attn on --no-webui --host 127.0.0.1
    --cache-type-k q4_0 --cache-type-v q4_0

models:
EOF
    if [ "$PROFILE" = mid ]; then
      cat <<'EOF'
  # Automatic offload: llama.cpp --fit picks how many layers/experts go to the GPU so that it fits in VRAM.
  "mid":
    cmd: |
      ${llama_bin} --port ${PORT}
      -m ${models_dir}/@@MODEL_FILE@@
      -fit on -fitt 1024 -ub 1024 --ctx-size @@CTX_AUTO@@ ${common_flags}
    ttl: 1800
EOF
      if [ "$TUNED_OK" = 1 ]; then cat <<'EOF'

  # Validated preset: RTX 2070 Super Max-Q (8 GB VRAM) + 16 GB RAM, Qwen3.6-35B-A3B UD-IQ3_XXS.
  # 28 MoE layers' experts stay in RAM, the rest on the GPU; -ub 2048 doubles prompt-processing speed.
  # If it fails to start with an out-of-memory error, lower --ctx-size or raise --n-cpu-moe.
  "mid-8gb-tuned":
    cmd: |
      ${llama_bin} --port ${PORT}
      -m ${models_dir}/@@MODEL_FILE@@
      --n-cpu-moe 28 -ngl 99 --moe-cache off -ub 2048 ${common_flags} --ctx-size @@CTX_TUNED@@
    ttl: 1800
EOF
      fi
    else
      cat <<'EOF'
  "small":
    cmd: |
      ${llama_bin} --port ${PORT}
      -m ${models_dir}/@@MODEL_FILE@@
      -fit on -fitt 512 -ub 512 --ctx-size @@CTX_AUTO@@ ${common_flags}
    ttl: 1800
EOF
    fi
    cat <<'EOF'

# Only one model is resident at a time; requests for a different model swap it in.
groups:
  main:
    swap: true
    exclusive: true
    members: [@@MEMBERS@@]
EOF
  } | sed -e "s|@@LLAMA_BIN@@|$(esc "$LLAMA_BIN")|g" -e "s|@@MODELS_DIR@@|$(esc "$mdir")|g" -e "s|@@MODEL_FILE@@|$(esc "$model_file")|g" \
          -e "s|@@CTX_AUTO@@|$ctx_auto|g" -e "s|@@CTX_TUNED@@|${CTX:-131072}|g" \
          -e "s|@@MEMBERS@@|$( [ "$PROFILE" = small ] && echo '"small"' || { [ "$TUNED_OK" = 1 ] && echo '"mid", "mid-8gb-tuned"' || echo '"mid"'; } )|g" > "$cfg"
  # add an alias so clients can simply ask for model "default"
  python3 -I - "$cfg" "$default" <<'PY'
import re, sys
p, name = sys.argv[1], sys.argv[2]
s = open(p).read()
s = re.sub(r'(  "%s":\n)' % re.escape(name), r'\1    aliases: ["default"]\n', s, count=1)
open(p, "w").write(s)
PY
  ok "wrote $cfg   (default model: $default)"

  cat > "$PREFIX/start.sh" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")"
exec ./bin/llama-swap --config config.yaml --listen 127.0.0.1:$PORT
EOF
  cat > "$PREFIX/test.sh" <<EOF
#!/usr/bin/env bash
# Smoke test: lists models, then asks the default model for a short reply (the first call loads the model).
set -e
curl -fsS http://127.0.0.1:$PORT/v1/models | python3 -I -c 'import json,sys; [print("model:", m["id"], "-", m.get("status",{}).get("value","")) for m in json.load(sys.stdin)["data"]]'
time curl -fsS http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}' \\
  | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print("reply:", d["choices"][0]["message"]["content"]); t=d.get("timings",{}); print("decode tok/s:", round(t.get("predicted_per_second",0),1))'
EOF
  chmod +x "$PREFIX/start.sh" "$PREFIX/test.sh"
}

install_systemd() {
  [ "$DO_SYSTEMD" = 1 ] || return 0
  log "systemd user service"
  have systemctl || die "systemctl not found"
  local unit="$HOME/.config/systemd/user/lapmoe.service"
  if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] write %s; systemctl --user enable --now lapmoe\n' "$unit"; return; fi
  mkdir -p "$(dirname "$unit")"
  cat > "$unit" <<EOF
[Unit]
Description=llama-swap (local LLM server, LapMoE)
After=network.target

[Service]
Type=simple
WorkingDirectory=$PREFIX
ExecStart=$BIN_DIR/llama-swap --config $PREFIX/config.yaml --listen 127.0.0.1:$PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now lapmoe.service
  ok "service lapmoe started (journalctl --user -u lapmoe -f)"
}

# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------
[ "$DO_BUILD" = 1 ] && build_llama
[ "$DO_SWAP" = 1 ]  && install_swap
[ "$DO_MODEL" = 1 ] && fetch_model
write_config
install_systemd

log "done"
face happy "everything is installed"
done_art
cat <<EOF
   start   : $PREFIX/start.sh          (or re-run with --systemd)
   test    : $PREFIX/test.sh           (first call loads the model, ~10-60 s)
   API     : http://127.0.0.1:$PORT/v1   (OpenAI-compatible; model name "default")
   config  : $PREFIX/config.yaml
EOF
