# common.sh - sourced by the node scripts. Layout (everything under ROOT, derived from this file's location when the
# repository is deployed at $ROOT/src/inksweep; override with INKSWEEP_ROOT):
#   $ROOT/src/inksweep          this repository
#   $ROOT/venv-inksweep         inksweep's venv          $ROOT/venv-inksweep-ref   the reference pipeline's venv
#   $ROOT/data                  checkpoints, villa checkout, the PHerc. 343 release, CT cache, renders, runs
#   $ROOT/logs                  logs and PID files
# Durable copies go to $DURABLE_DIR when it is set (nothing in $ROOT survives the node).
# GPU / CPU pinning: GPUS (default 0) and CPUS (default 0-11); jobs run with nice -n 10.
set -euo pipefail
NODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$NODE_DIR/.." && pwd)
ROOT=${INKSWEEP_ROOT:-$(cd "$REPO/../.." && pwd)}
VENV=$ROOT/venv-inksweep
REFVENV=$ROOT/venv-inksweep-ref
DATA=$ROOT/data
LOGS=$ROOT/logs
CKPT=$DATA/checkpoints
REFCKPT=$DATA/checkpoints-ref
VILLA_COMMIT=0fb38c45b488e192566ea83ae36369cb6db92652      # villa commit of every PHerc. 343 ink map
VILLA=$DATA/villa-${VILLA_COMMIT:0:8}
WINNER_COMMIT=e8370f1f88a211166056df62ea6fcfffe4a1063e     # Nieuwlaar/pherc343-first-letters (read and run as data)
WINNER=$DATA/pherc343-first-letters
PYV=${PYV:-3.14.10}             # an exact final CPython release (see setup_inksweep.sh)
UV=${UV:-/usr/local/bin/uv}
GPUS=${GPUS:-0}
CPUS=${CPUS:-0-11}
DURABLE_DIR=${DURABLE_DIR:-}
mkdir -p "$DATA" "$LOGS"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} UV_LINK_MODE=copy

# log_to NAME: tee everything to $LOGS/NAME.log, record the PID, report the exit status
log_to() {
  local name=$1
  LOG=$LOGS/$name.log
  echo $$ > "$LOGS/$name.pid"
  exec > >(tee -a "$LOG") 2>&1
  echo "== $(date -u +%FT%TZ) $name start (pid $$, host $(hostname))"
  trap 'rc=$?; echo "== $(date -u +%FT%TZ) '"$name"' exit $rc"; rm -f "$LOGS/'"$name"'.pid"' EXIT
}

require_ready() {
  [[ -f $VENV/READY ]] || { echo "inksweep venv not ready: run node/setup_inksweep.sh"; exit 3; }
  [[ -f $VENV/TESTS_OK ]] || echo "WARNING: the test suite did not pass at setup (see $LOGS/setup_inksweep.log)"
}

# pinned: run "$@" on the configured GPUs and CPUs at low priority
pinned() { CUDA_VISIBLE_DEVICES=$GPUS nice -n 10 taskset -c "$CPUS" "$@"; }

durable_copy() {  # durable_copy SRC [SUBDIR]: copy the small result files (json, md, tsv, png, logs,
                  # ensemble.tif) to the durable folder, if configured; per-depth maps stay on the node
  [[ -n $DURABLE_DIR ]] || { echo "(DURABLE_DIR not set: no durable copy of $1)"; return 0; }
  local dest=$DURABLE_DIR/${2:-$(basename "$1")}
  mkdir -p "$dest"
  rsync -a --prune-empty-dirs --include='*/' --include='*.json' --include='*.md' --include='*.tsv' \
        --include='*.png' --include='*.log' --include='ensemble.tif' --exclude='*' "$1"/ "$dest"/
  echo "copied results of $1 -> $dest"
}

IS=$VENV/bin/inksweep
PY=$VENV/bin/python
RPY=$REFVENV/bin/python
