#!/bin/bash
# run_inksweep_reference.sh - the PHerc. 343 reference chain, exactly as its author ran it, on inksweep's render:
# villa ink inference per model and depth (the release's run_depth_sweep.sh / run_ink.sh), on the plain render and on a
# zero-padded copy (pad_render.py, 40 rows x 200 columns), merge_tile_phase.py, ensemble_maps.py and
# make_submission_image.py. Runs in its own venv ($ROOT/venv-inksweep-ref: villa at the release's commit).
#
#   DMIN=-15 DMAX=15 P_VILLA=2 P_HEC=2 GPUS=0 CPUS=0-11 [REF_NAME=reference] bash node/run_inksweep_reference.sh
#
# Needs the render of node/run_inksweep_343.sh (step 3). Writes $DATA/pherc0343/$REF_NAME/{maps_plain,maps_padded,
# maps,images,image,timing.json} (REF_NAME defaults to "reference", the folder node/run_inksweep_compare.sh reads; give
# another name, e.g. reference_p4, to time other P_VILLA / P_HEC / CPUS settings beside it). Prints the wall time of the
# sweep (the reference side of the speedup).
set -euo pipefail
source "$(dirname "$0")/common.sh"
log_to "run_inksweep_${REF_NAME:-reference}"
require_ready
command -v gcc > /dev/null || echo "WARNING: no C compiler: villa infer compiles the model with torch.compile by default"
N=concat_w047-w048_R5B2_z9500-11000
W=$DATA/pherc0343
Z=$W/${N}_66.zarr
REF=$W/${REF_NAME:-reference}
DMIN=${DMIN:--15}; DMAX=${DMAX:-15}
export P_VILLA=${P_VILLA:-2} P_HEC=${P_HEC:-2}
[[ -e $Z/0/.zarray ]] || { echo "render missing ($Z): run node/run_inksweep_343.sh first"; exit 3; }
mkdir -p "$REF" "$W/tmp"
[[ -d $W/${N}_66_padded.zarr ]] || "$RPY" "$WINNER/ink/scripts/pad_render.py" "$Z" "$W/${N}_66_padded.zarr" \
    --pad-rows 40 --pad-cols 200
export CKPT_DIR=$REFCKPT PY=$RPY HECATE_PY=$REFCKPT/hecate.py N_LAYERS=66 NATIVE_UM=8.64 TMPDIR=$W/tmp
t0=$(date +%s)
pinned bash "$WINNER/ink/scripts/run_depth_sweep.sh" "$Z" "$REF/maps_plain" "$DMIN" "$DMAX"
t1=$(date +%s)
pinned bash "$WINNER/ink/scripts/run_depth_sweep.sh" "$W/${N}_66_padded.zarr" "$REF/maps_padded" "$DMIN" "$DMAX"
t2=$(date +%s)
"$RPY" "$WINNER/ink/scripts/merge_tile_phase.py" --plain "$REF/maps_plain" --plain-name "${N}_66" \
    --padded "$REF/maps_padded" --padded-name "${N}_66_padded" --node-source "$W/mesh/${N}_node_source.npz" \
    --out "$REF/maps" --out-name "$N"
t3=$(date +%s)
mkdir -p "$REF/images" "$REF/image"
(cd "$REF/images" && "$RPY" "$WINNER/ink/scripts/ensemble_maps.py" --surface "$Z" --name "$N" --maps "$REF/maps" \
    --out . "--stat-depths=$DMIN..$DMAX" --write-depths 1)
t4=$(date +%s)
M=$REF/maps
"$RPY" "$WINNER/ink/scripts/make_submission_image.py" --mesh "$W/mesh/$N.tifxyz" \
    --umbilicus "$WINNER/inputs/umbilicus_PHerc0343.json" --ct "$Z" --overlay-style dimmed-ct --caption compact \
    --area 380,105,6021,2560 --max-cm2 12 --ct-layer d+1 \
    --ink "$M/${N}__dnative__L66s26_reverse.tif" "$M/${N}__hecate__L66zc33.50_reverse.png" \
          "$M/${N}__hybrid_3d2d-seed42__L66s26_reverse.tif" "$M/${N}__hybrid_3d2d-seed43__L66s26_reverse.tif" \
    --ensemble --ink-stats "$REF/images/${N}__stats.json" \
    --rows "$WINNER/inputs/${N}_rows.json" --letters "$WINNER/inputs/${N}_letter_boxes.json" --letter-markers off \
    --sizes-csv "$REF/image/${N}_letter_sizes.csv" --out-dir "$REF/image" --force
t5=$(date +%s)
"$PY" - "$REF" "$t0" "$t1" "$t2" "$t3" "$t4" "$t5" "$DMIN" "$DMAX" <<'EOF'
import json, sys, csv, os
ref, t0, t1, t2, t3, t4, t5, dmin, dmax = sys.argv[1], *map(int, sys.argv[2:8]), int(sys.argv[8]), int(sys.argv[9])
def jobs(p):
    rows = [r for r in csv.reader(open(p), delimiter="\t") if len(r) >= 4]
    return dict(jobs=len(rows), failed=sum(r[3] != "0" for r in rows), job_seconds=sum(float(r[2]) for r in rows))
t = dict(depths=[dmin, dmax], p_villa=int(os.environ["P_VILLA"]), p_hec=int(os.environ["P_HEC"]),
         sweep_plain_s=t1 - t0, sweep_padded_s=t2 - t1, sweep_wall_s=t2 - t0, merge_s=t3 - t2, ensemble_s=t4 - t3,
         image_s=t5 - t4, total_s=t5 - t0,
         plain=jobs(f"{ref}/maps_plain/logs/timings.tsv"), padded=jobs(f"{ref}/maps_padded/logs/timings.tsv"))
json.dump(t, open(f"{ref}/timing.json", "w"), indent=1)
print(f"REFERENCE: sweep wall {t['sweep_wall_s']} s (plain {t['sweep_plain_s']} s + padded {t['sweep_padded_s']} s), "
      f"total {t['total_s']} s; failed jobs {t['plain']['failed'] + t['padded']['failed']}")
EOF
echo "reference image: $REF/image/$N.png"
durable_copy "$REF" "pherc0343/$(basename "$REF")"
