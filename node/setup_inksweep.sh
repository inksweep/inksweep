#!/bin/bash
# setup_inksweep.sh - idempotent setup of inksweep and of the reference pipeline on a Linux GPU node.
#
#   bash node/setup_inksweep.sh       [UV=/path/to/uv] [PYV=3.14.10] [EXTRA_READERS="readerv3,d9v4c,v8in,..."]
#
# Steps (each skipped when done; log: $ROOT/logs/setup_inksweep.log):
#   1 uv >= 0.9 (an older uv only knows 3.14 pre-releases, on which torch's cp314 wheels segfault; a pinned uv is
#     then installed under $ROOT/tools, in a venv built by the old uv, never with `python -m venv`), CPython $PYV (an
#     exact final release), two venvs: $ROOT/venv-inksweep and $ROOT/venv-inksweep-ref; a venv whose interpreter is a
#     pre-release or another version is recreated
#   2 torch + torchvision from the PyTorch cu129 index (as an extra index), everything else from PyPI
#     (node/requirements.txt); then OpenCV's manylinux2014 wheel, pinned by sha256, on Linux x86_64 (the manylinux_2_28
#     wheels bundle an OpenSSL whose FIPS self-test aborts `import cv2` on FIPS kernels)
#   3 ScrollPrize/villa at $VILLA_COMMIT (blob-less sparse checkout of vesuvius/ and ink-detection/; the GPL
#     volume-cartographer tree is not checked out), its vesuvius package installed with --no-deps into both venvs
#   4 Nieuwlaar/pherc343-first-letters at $WINNER_COMMIT (the reference recipe), its outputs sha256-checked
#   5 inksweep (editable, --no-deps) into its venv
#   6 checkpoints, every file pinned by sha256, into $DATA/checkpoints; the reference's file names in
#     $DATA/checkpoints-ref (links to the same verified files; the recipe's own fetch_checkpoints.sh verifies them and
#     rebuilds the dense_native .pth)
#   7 import checks; the test suite (pytest) -> TESTS_OK; READY
set -euo pipefail
source "$(dirname "$0")/common.sh"
log_to setup_inksweep
PYV=${PYV:-3.14.10}
UV_MIN=0.9
UV_PIN=${UV_PIN:-0.13.6}
TOOLS=$ROOT/tools
TORCH_SPEC=${TORCH_SPEC:-torch==2.13.0+cu129}
TORCHVISION_SPEC=${TORCHVISION_SPEC:-torchvision==0.28.0+cu129}     # timm (imported by villa's model builder) needs it
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}
OPENCV_WHEEL_URL=https://files.pythonhosted.org/packages/2b/97/8170e9819764c47e436c130d3ff6cfb73b58f923eae9d3a03d8982b04aec/opencv_python_headless-5.0.0.93-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
OPENCV_WHEEL_SHA256=09a872a157c1376ab922a69bbf22f9a95bcc7b658a9d8b436a60212b02b2eeb4
EXTRA_READERS=${EXTRA_READERS:-}
step() { echo "-- $*"; }
version_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]; }   # $1 >= $2
uv_version() { "$1" --version | awk '{print $2}'; }
check_python() {  # check_python VENV: a final release of the pinned version
  "$1/bin/python" - "$PYV" <<'EOF'
import platform, sys
want, have = sys.argv[1], platform.python_version()
same = have == want if want.count(".") >= 2 else have.startswith(want + ".")
print(f"{sys.executable}: Python {have} ({sys.version_info.releaselevel})")
sys.exit(0 if sys.version_info.releaselevel == "final" and same else 1)
EOF
}

step "1 uv, python, venvs"
[[ -x $UV ]] || UV=$(command -v uv || true)
[[ -n $UV && -x $UV ]] || { echo "uv not found: set UV=/path/to/uv (uv >= $UV_MIN)"; exit 2; }
if ! version_ge "$(uv_version "$UV")" "$UV_MIN"; then
  echo "uv $(uv_version "$UV") is older than $UV_MIN: installing uv $UV_PIN under $TOOLS"
  if [[ ! -x $TOOLS/uv-$UV_PIN/bin/uv ]]; then
    "$UV" venv -q --python 3.12 "$TOOLS/uv-$UV_PIN"
    "$UV" pip install -q --python "$TOOLS/uv-$UV_PIN/bin/python" "uv==$UV_PIN"
  fi
  UV=$TOOLS/uv-$UV_PIN/bin/uv
fi
echo "uv: $UV ($(uv_version "$UV"))"
"$UV" python install "$PYV"
for v in "$VENV" "$REFVENV"; do
  if [[ -x $v/bin/python ]] && ! check_python "$v"; then
    echo "recreating $v: not a final Python $PYV"
    rm -rf "$v"
  fi
  [[ -x $v/bin/python ]] || "$UV" venv "$v" --python "$PYV"
  check_python "$v" || { echo "$v: not a final Python $PYV"; exit 2; }
done

step "2 torch + pinned packages"
for v in "$VENV" "$REFVENV"; do
  if [[ ! -f $v/.deps_ok ]]; then
    # torch and torchvision from the PyTorch index (kept as an *extra* index so the pins resolve) ...
    "$UV" pip install --python "$v/bin/python" --extra-index-url "$TORCH_INDEX" "$TORCH_SPEC" "$TORCHVISION_SPEC"
    # ... everything else from PyPI only (the installed torch satisfies the packages that depend on it)
    "$UV" pip install --python "$v/bin/python" -r "$NODE_DIR/requirements.txt"
    "$v/bin/python" -c "import torch; v = torch.__version__; assert v.startswith('${TORCH_SPEC#torch==}'.split('+')[0]), v; print('torch', v)"
    touch "$v/.deps_ok"
  fi
done
if [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]; then
  [[ $(cat /proc/sys/crypto/fips_enabled 2>/dev/null || echo 0) == 1 ]] && echo "FIPS kernel: OpenCV's manylinux2014 build is required"
  WHL=$DATA/wheels/$(basename "$OPENCV_WHEEL_URL")
  if [[ ! -f $WHL ]] || ! echo "$OPENCV_WHEEL_SHA256  $WHL" | sha256sum -c --quiet - 2>/dev/null; then
    mkdir -p "$DATA/wheels"
    curl -fsSL --retry 5 "$OPENCV_WHEEL_URL" -o "$WHL.part"
    echo "$OPENCV_WHEEL_SHA256  $WHL.part" | sha256sum -c --quiet - || { echo "OpenCV wheel sha256 mismatch"; exit 2; }
    mv "$WHL.part" "$WHL"
  fi
  for v in "$VENV" "$REFVENV"; do
    if [[ ! -f $v/.opencv_manylinux2014_ok ]]; then
      "$UV" pip install --python "$v/bin/python" --reinstall --no-deps "$WHL"
      touch "$v/.opencv_manylinux2014_ok"
    fi
  done
fi
for v in "$VENV" "$REFVENV"; do
  "$v/bin/python" -c "import cv2; print('cv2', cv2.__version__)" || { echo "import cv2 fails in $v"; exit 2; }
done

step "3 villa $VILLA_COMMIT"
if [[ ! -d $VILLA/.git ]]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git "$VILLA"
  git -C "$VILLA" sparse-checkout set --cone vesuvius ink-detection
fi
if [[ $(git -C "$VILLA" rev-parse HEAD 2>/dev/null || true) != "$VILLA_COMMIT" ]]; then
  git -C "$VILLA" fetch -q --filter=blob:none origin "$VILLA_COMMIT" || true
  git -C "$VILLA" checkout -q "$VILLA_COMMIT"
fi
[[ $(git -C "$VILLA" rev-parse HEAD) == "$VILLA_COMMIT" ]] || { echo "villa is not at $VILLA_COMMIT"; exit 2; }
for v in "$VENV" "$REFVENV"; do
  if [[ ! -f $v/.villa_ok ]]; then
    "$UV" pip install --python "$v/bin/python" --no-deps -e "$VILLA/vesuvius"
    touch "$v/.villa_ok"
  fi
done

step "4 the PHerc. 343 release $WINNER_COMMIT"
[[ -d $WINNER/.git ]] || git clone -q https://github.com/Nieuwlaar/pherc343-first-letters "$WINNER"
git -C "$WINNER" checkout -q "$WINNER_COMMIT"
(cd "$WINNER/outputs" && sha256sum -c --quiet SHA256SUMS) || { echo "release outputs: sha256 mismatch"; exit 2; }
echo "release outputs sha256 OK"

step "5 inksweep"
"$UV" pip install --python "$PY" --no-deps -e "$REPO"

step "6 checkpoints"
READERS="dnative,hecate,ink9_s42,ink9_s43${EXTRA_READERS:+,$EXTRA_READERS}"
"$IS" fetch --readers "$READERS" --ckpt-dir "$CKPT"
mkdir -p "$REFCKPT"
ln -sf "$CKPT/ink_9um/hybrid_3d2d-seed42/step-075000.pth" "$REFCKPT/hybrid_3d2d-seed42_step-075000.pth"
ln -sf "$CKPT/ink_9um/hybrid_3d2d-seed43/step-075000.pth" "$REFCKPT/hybrid_3d2d-seed43_step-075000.pth"
ln -sf "$CKPT/hecate/hecate_9.6um.pth" "$REFCKPT/hecate_9.6um.pth"
ln -sf "$CKPT/hecate/hecate.py" "$REFCKPT/hecate.py"
ln -sf "$CKPT/dense_native/dense_native_016000.safetensors" "$REFCKPT/dense_native_016000.safetensors"
ln -sf "$CKPT/dense_native/train_dense_native.json" "$REFCKPT/train_dense_native.json"
PY=$RPY bash "$WINNER/ink/scripts/fetch_checkpoints.sh" "$WINNER/ink/models.tsv" "$REFCKPT"

step "7 checks"
command -v gcc > /dev/null || echo "WARNING: no C compiler: the reference's villa infer compiles its model (torch.compile)"
"$PY" - <<'EOF'
import torch, numpy, cv2, inksweep
import vesuvius.ink_detection.models.model, vesuvius.ink_detection.inference.inference_runtime
print("inksweep", inksweep.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "numpy", numpy.__version__, "cv2", cv2.__version__)
assert torch.cuda.is_available(), "CUDA not available"
EOF
"$RPY" -c "import vesuvius.ink_detection.inference.infer as m, cv2, zarr, torch; print('reference venv:', m.__file__)"
if (cd "$REPO" && CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m pytest -q tests); then
  touch "$VENV/TESTS_OK"
else
  rm -f "$VENV/TESTS_OK"; echo "WARNING: tests failed (setup continues; see above)"
fi
"$PY" -c "import json, inksweep.runner as r; print(json.dumps(r.software()))" > "$VENV/READY"
echo "READY: $VENV/READY"
