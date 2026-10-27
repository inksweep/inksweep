#!/bin/bash
# run_inksweep_343.sh - PHerc. 343 end to end with inksweep: the release's segment, a 66-layer clean-room render
# from the open-data bucket, the full four-model sweep (d -15..+15, both orientations, two tile phases merged per
# half), the ensemble and the image at d = +1.
#
#   MODE=exact|fast (default exact)  GPUS=0  CPUS=0-11  [DURABLE_DIR=...]  bash node/run_inksweep_343.sh
#
# Steps:
#   1 the segment concat_w047-w048 from the release's two meshes (the release's build_concat.py, which also writes the
#     node-source map used by the tile-phase merge); checked bit-identical to the released segment
#   2 the voxel size from the volume's metadata.json (never from the file name)
#   3 render 66 layers 1 voxel apart, without flipping the normals (the release's render); the normal must point away
#     from the scroll axis (checked against the release's umbilicus); CT chunks fetched once into $DATA/ct
#   4 inksweep run (recipe pherc0343, phases 0,0 and 40,200, select merge on the w047 half)
#   5 the image: inksweep's preview at d = +1 over the release's area, and the release's own
#     make_submission_image.py on inksweep's maps (exported under the release's names)
# Prints: render share of outward normals, sweep wall time, per-reader tiles/s, the image paths.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE=${MODE:-exact}
log_to "run_inksweep_343_$MODE"
require_ready
N=concat_w047-w048_R5B2_z9500-11000
VOL_URL=${VOL_URL:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr}
W=$DATA/pherc0343
RUN=$W/run_$MODE
mkdir -p "$W"

echo "-- 1 segment"
if [[ ! -f $W/mesh/${N}_node_source.npz ]]; then
  "$RPY" "$WINNER/mesh/build_concat.py" --w047 "$WINNER/outputs/w047_R5B2_z9500-11000.tifxyz" \
      --w048 "$WINNER/outputs/w048_R5B2_z9500-11000.tifxyz" \
      --letter-boxes "$WINNER/inputs/${N}_letter_boxes.json" --out-dir "$W/mesh"
fi
for f in x.tif y.tif z.tif; do
  a=$(sha256sum "$W/mesh/$N.tifxyz/$f" | cut -d' ' -f1); b=$(sha256sum "$WINNER/outputs/$N.tifxyz/$f" | cut -d' ' -f1)
  [[ $a == "$b" ]] || { echo "segment $f differs from the release ($a != $b)"; exit 4; }
done
echo "segment bit-identical to the release"

echo "-- 2 voxel size"
if [[ -z ${VOXEL_UM:-} ]]; then
  META_URL=
  for u in "$VOL_URL/metadata.json" "${VOL_URL%.zarr}/metadata.json" "$(dirname "$VOL_URL")/metadata.json"; do
    if curl -fsS "$u" -o "$W/metadata.json" 2>/dev/null; then META_URL=$u; break; fi
  done
  [[ -n $META_URL ]] || { echo "no metadata.json found next to $VOL_URL; set VOXEL_UM"; exit 4; }
  read -r VOXEL_UM VOXEL_KEY < <("$PY" - "$W/metadata.json" <<'EOF'
import json, sys
found = []
def walk(o, path):
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k in ("pixel_size_um", "voxel_size_um"):
                    found.append((0, float(v), p))               # micrometres
                elif k == "samplePixelSize":
                    found.append((1, float(v) * 1000.0, p))      # ESRF scan metadata: millimetres
            walk(v, p)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]")
walk(json.load(open(sys.argv[1])), "")
found.sort()
print(f"{found[0][1]:.6g} {found[0][2]}" if found else "")
EOF
)
  [[ -n ${VOXEL_UM:-} ]] || { echo "voxel size not found in $META_URL (pixel_size_um, voxel_size_um, samplePixelSize); set VOXEL_UM"; exit 4; }
  printf '{"voxel_size_um": %s, "key": "%s", "source": "%s"}\n' "$VOXEL_UM" "$VOXEL_KEY" "$META_URL" > "$W/voxel_size.json"
else
  printf '{"voxel_size_um": %s, "key": "VOXEL_UM", "source": "environment"}\n' "$VOXEL_UM" > "$W/voxel_size.json"
fi
echo "voxel size $VOXEL_UM um: $(cat "$W/voxel_size.json")"
"$PY" -c "assert abs($VOXEL_UM - 8.64) < 0.01, 'PHerc0343 volume 20250521140437 should be 8.64 um'"

echo "-- 3 render"
SV=$W/sv
if [[ ! -f $SV/surface.json ]]; then
  pinned "$IS" render "$W/mesh/$N.tifxyz" --volume "$VOL_URL" --cache "$DATA/ct/PHerc0343_20250521140437" \
      --out "$SV" --layers 66 --normal-sign keep --umbilicus "$WINNER/inputs/umbilicus_PHerc0343.json" \
      --voxel-um "$VOXEL_UM" --procs 11 --sample PHerc0343 --volume-id 20250521140437
fi
"$PY" - "$SV/surface.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
share = s["outward_share_plus_d"]
print(f"render: flip_normals={s['flip_normals']} outward share of +d {share:.4f} ({s['positive_d']}), "
      f"chunks {s['chunks']} (absent {s['absent_chunks']}), seconds {s['seconds']}")
assert share > 0.8, "+d must point away from the scroll axis"
EOF
ln -sfn "$SV/layers.zarr" "$W/${N}_66.zarr"          # the release's file name, for its scripts

echo "-- 4 sweep ($MODE)"
EXTRA=()
[[ $MODE == fast ]] && EXTRA+=(--compile)
pinned "$IS" run "$SV" --out "$RUN" --name "$N" --recipe pherc0343 --mode "$MODE" --phase 0,0 --phase 40,200 \
    --phase-merge select --phase-mask "npz:$W/mesh/${N}_node_source.npz:1,4" --ckpt-dir "$CKPT" --no-download \
    --write-depths=1 --display-depth 1 --area 380,105,6021,2560 "${EXTRA[@]}"
"$PY" - "$RUN/ink.json" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
t = j["timings"]
print(f"sweep wall {t['sweep_wall_s']:.0f} s, total {t['total_s']:.0f} s, maps {sum(1 for m in j['maps'] if not m.get('skipped'))}")
for r, v in t["per_reader"].items():
    print(f"  {r:10s} {v['maps']:4d} maps {v['tiles']:9d} tiles {v['seconds']:8.0f} s  {v['tiles'] / max(v['seconds'], 1e-9):7.0f} tiles/s")
e = j.get("ensemble", {}).get("ensemble", {})
print("ensemble stretch", e.get("p1"), e.get("p99_5"))
EOF

echo "-- 5 image"
REFN=$RUN/reference_names
"$IS" export-reference "$RUN" "$REFN" --name "$N"
mkdir -p "$RUN/release_ensemble" "$RUN/image"
(cd "$RUN/release_ensemble" && "$RPY" "$WINNER/ink/scripts/ensemble_maps.py" --surface "$W/${N}_66.zarr" --name "$N" \
    --maps "$REFN" --out . --stat-depths=-15..15 --write-depths 1)
M=$REFN
"$RPY" "$WINNER/ink/scripts/make_submission_image.py" --mesh "$W/mesh/$N.tifxyz" \
    --umbilicus "$WINNER/inputs/umbilicus_PHerc0343.json" --ct "$W/${N}_66.zarr" --overlay-style dimmed-ct \
    --caption compact --area 380,105,6021,2560 --max-cm2 12 --ct-layer d+1 \
    --ink "$M/${N}__dnative__L66s26_reverse.tif" "$M/${N}__hecate__L66zc33.50_reverse.png" \
          "$M/${N}__hybrid_3d2d-seed42__L66s26_reverse.tif" "$M/${N}__hybrid_3d2d-seed43__L66s26_reverse.tif" \
    --ensemble --ink-stats "$RUN/release_ensemble/${N}__stats.json" \
    --rows "$WINNER/inputs/${N}_rows.json" --letters "$WINNER/inputs/${N}_letter_boxes.json" --letter-markers off \
    --sizes-csv "$RUN/image/${N}_letter_sizes.csv" --out-dir "$RUN/image" --force
echo "IMAGE (inksweep preview): $RUN/ink/preview_d+01.png"
echo "IMAGE (release script on inksweep maps): $RUN/image/$N.png"
echo "ensemble map: $RUN/ink/ensemble.tif"
durable_copy "$RUN" "pherc0343/run_$MODE"
