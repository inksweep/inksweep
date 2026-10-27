#!/bin/bash
# run_inksweep_vc3d_compare.sh - inksweep's MIT renderer against VC3D on the PHerc. 343 segment, layer by layer.
#
#   VC3D_APPRUN=/path/to/VC3D-41701aa-2026-09-24-linux-x86_64.AppImage   (or its extracted squashfs-root/AppRun)
#   [CT_ZARR=/path/to/local_level0.zarr] [THREADS=4] [CACHE_GB=4] GPUS=0 CPUS=0-11 bash node/run_inksweep_vc3d_compare.sh
#
# Needs node/run_inksweep_343.sh first (the segment's mesh, inksweep's render $DATA/pherc0343/sv and its CT chunk
# cache). VC3D (GPL-3.0) is only run here as an external program; nothing of it is in this repository.
#   1 VC3D: the sha256 of the AppImage (or of the extracted vc_render_tifxyz) against the 41701aa build the release
#     names in ink/Dockerfile, and any shared library it cannot find
#   2 CT: CT_ZARR if given (a local copy of resolution level 0); otherwise a local level-0 copy of exactly the CT chunks
#     inksweep's render read (`inksweep export-ct`, hard links into that cache), so both renderers sample the same bytes
#     (where inksweep's renderer skips a tile of a folded mesh, VC3D then reads zeros as well). VC3D_REMOTE_URL=<volume
#     URL> streams the CT instead, the alternative the release's render script documents (CT_ZARR: an empty cache)
#   3 the VC3D render with the release's own mesh/render_66_layers.sh: vc_render_tifxyz --group-idx 0 --scale 1
#     --num-slices 66 --slice-step 1, no --surface-interpolation, QT_QPA_PLATFORM=offscreen
#   4 inksweep compare-render, A = VC3D, B = inksweep render: per layer, at d = +1, and a +-2 px shift search at d = +1
# Prints: the VC3D build check, both render times, the per-layer table and the summary (min r, max |diff|, equal
# voxels, valid-mask agreement, d = +1). Report: $DATA/pherc0343/vc3d/compare_render.{json,md}.
set -euo pipefail
source "$(dirname "$0")/common.sh"
log_to run_inksweep_vc3d_compare
require_ready
N=concat_w047-w048_R5B2_z9500-11000
VOL_URL=${VOL_URL:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr}
W=$DATA/pherc0343
SV=$W/sv
MESH=$W/mesh/$N.tifxyz
V=$W/vc3d
OUT=$V/${N}_66_vc3d.zarr
THREADS=${THREADS:-4}
CACHE_GB=${CACHE_GB:-4}
VC3D_APPIMAGE_SHA256=1d88a1f396751561863b0ff50b3eed35a1282455bd03d0884ba2540c34004990   # ink/Dockerfile of the release
VC3D_RENDER_SHA256=89c01e525b25545b7fb94be13e833a11bbd10c88312b020ea491e886962a3fb0     # its usr/bin/vc_render_tifxyz
[[ -n ${VC3D_APPRUN:-} && -x ${VC3D_APPRUN:-} ]] || { echo "set VC3D_APPRUN to the VC3D AppImage or squashfs-root/AppRun"; exit 3; }
[[ -f $SV/surface.json && -d $MESH ]] || { echo "inksweep's render or the mesh is missing: run node/run_inksweep_343.sh"; exit 3; }
[[ -f $WINNER/mesh/render_66_layers.sh ]] || { echo "the release checkout is missing: run node/setup_inksweep.sh"; exit 3; }
mkdir -p "$V"

echo "-- 1 VC3D build"
if [[ $(basename "$VC3D_APPRUN") == AppRun ]]; then
  APPDIR=$(cd "$(dirname "$VC3D_APPRUN")" && pwd)
  BIN=$APPDIR/usr/bin/vc_render_tifxyz
  if [[ -f $BIN ]]; then
    got=$(sha256sum "$BIN" | cut -d' ' -f1)
    if [[ $got == "$VC3D_RENDER_SHA256" ]]; then echo "vc_render_tifxyz: the release's VC3D 41701aa build ($got)"
    else echo "vc_render_tifxyz sha256 $got: NOT the release's 41701aa build ($VC3D_RENDER_SHA256)"; fi
    missing=$(LD_LIBRARY_PATH="$APPDIR/usr/lib:$APPDIR/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" ldd "$BIN" \
              2>/dev/null | grep "not found" || true)
    if [[ -n $missing ]]; then
      echo "shared libraries VC3D cannot find:"; echo "$missing"
      echo "(the release lists for Ubuntu 24.04: libglx0 libegl1 libopengl0 libxkbcommon0 libfontconfig1 libxcb-cursor0"
      echo " libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xkb1 libxkbcommon-x11-0 libdbus-1-3)"
    fi
  else
    echo "no usr/bin/vc_render_tifxyz next to $VC3D_APPRUN: build not checked"
  fi
else
  got=$(sha256sum "$VC3D_APPRUN" | cut -d' ' -f1)
  if [[ $got == "$VC3D_APPIMAGE_SHA256" ]]; then echo "AppImage: the release's VC3D 41701aa ($got)"
  else echo "AppImage sha256 $got: NOT the release's VC3D 41701aa AppImage ($VC3D_APPIMAGE_SHA256)"; fi
  export APPIMAGE_EXTRACT_AND_RUN=1             # run without FUSE (the AppImage extracts itself to a temporary folder)
fi

echo "-- 2 CT"
REMOTE=()
if [[ -n ${VC3D_REMOTE_URL:-} ]]; then
  CT=${CT_ZARR:-$V/ct_stream_cache}
  mkdir -p "$CT"
  REMOTE=(--remote-url "$VC3D_REMOTE_URL")
  echo "CT streamed from $VC3D_REMOTE_URL (VC3D cache $CT)"
elif [[ -n ${CT_ZARR:-} ]]; then
  CT=$CT_ZARR
  [[ -f $CT/0/.zarray ]] || { echo "CT_ZARR=$CT has no resolution level 0 (0/.zarray)"; exit 3; }
  echo "CT: $CT (given)"
else
  CT=$DATA/ct/PHerc0343_20250521140437_level0.zarr
  "$IS" export-ct --volume "$VOL_URL" --cache "$DATA/ct/PHerc0343_20250521140437" --out "$CT"
  echo "CT: $CT (the level-0 chunks inksweep's render read)"
fi

echo "-- 3 VC3D render"
if [[ ! -f $V/vc3d_render.json ]]; then
  rm -rf "$OUT"
  t0=$(date +%s)
  if [[ ${#REMOTE[@]} -eq 0 ]]; then
    pinned bash "$WINNER/mesh/render_66_layers.sh" "$VC3D_APPRUN" "$CT" "$MESH" "$OUT" "$THREADS" "$CACHE_GB"
  else                                          # the same command line plus the streaming flag the script documents
    pinned env QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS="$THREADS" "$VC3D_APPRUN" vc_render_tifxyz --volume "$CT" \
        --group-idx 0 --scale 1 --segmentation "$MESH" --num-slices 66 --slice-step 1 --cache-gb "$CACHE_GB" \
        --zarr-output "$OUT" "${REMOTE[@]}"
  fi
  t1=$(date +%s)
  printf '{"vc3d_render_s": %d, "threads": %d, "cache_gb": %s, "ct": "%s", "remote": "%s"}\n' \
      $((t1 - t0)) "$THREADS" "$CACHE_GB" "$CT" "${VC3D_REMOTE_URL:-}" > "$V/vc3d_render.json"
fi
[[ -f $OUT/0/.zarray || -f $OUT/.zarray ]] || { echo "VC3D wrote no zarr array at $OUT"; ls -la "$OUT" || true; exit 4; }
"$PY" - "$V/vc3d_render.json" "$SV/surface.json" <<'EOF'
import json, sys
v, s = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
t = s.get("seconds", {})
print(f"render times: VC3D {v['vc3d_render_s']} s with {v['threads']} OpenMP threads; inksweep render {t.get('render')} s "
      f"(+ {t.get('fetch')} s fetching the CT chunks)")
EOF

echo "-- 4 compare: A = VC3D, B = inksweep render"
pinned "$IS" compare-render "$OUT" "$SV" --names vc3d,inksweep --max-shift 2 --report "$V/compare_render.json"
durable_copy "$V" pherc0343/vc3d
