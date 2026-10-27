#!/bin/bash
# run_inksweep_compare.sh - equivalence of inksweep with the reference chain on the same PHerc. 343 render.
#
#   DMIN=-15 DMAX=15 bash node/run_inksweep_compare.sh      (after run_inksweep_343.sh [MODE=exact and MODE=fast]
#                                                            and run_inksweep_reference.sh)
#
# For each inksweep run found ($DATA/pherc0343/run_exact, run_fast) every merged map (4 models x depths x 2
# orientations) is compared with the reference map in the reference's own 8-bit units: Pearson r and the largest
# absolute difference. Acceptance, printed per mode:
#   exact: every map r >= 0.9999 and max |diff| <= 1/255
#   fast:  every map r >= 0.999
# Also compared: the recipe's statistics (stretch, mean, SD, ensemble stretch), its grey images and the final image.
set -euo pipefail
source "$(dirname "$0")/common.sh"
log_to run_inksweep_compare
require_ready
N=concat_w047-w048_R5B2_z9500-11000
W=$DATA/pherc0343
REF=$W/reference
DMIN=${DMIN:--15}; DMAX=${DMAX:-15}
[[ -d $REF/maps ]] || { echo "reference maps missing: run node/run_inksweep_reference.sh"; exit 3; }
rc=0
for MODE in exact fast; do
  RUN=$W/run_$MODE
  [[ -f $RUN/ink.json ]] || { echo "(no $MODE run: MODE=$MODE bash node/run_inksweep_343.sh)"; continue; }
  if [[ $MODE == exact ]]; then RMIN=0.9999; MAXD=$("$PY" -c "print(1/255)"); else RMIN=0.999; MAXD=1.0; fi
  echo "== $MODE"
  if "$IS" compare "$RUN" "$REF/maps" --name "$N" "--depths=$DMIN..$DMAX" --r-min "$RMIN" --max-diff "$MAXD" \
        --report "$RUN/compare_reference.json" --ref-stats "$REF/images/${N}__stats.json" \
        --ref-grey "$REF/images"; then
    echo "ACCEPTANCE $MODE maps: PASS"
  else
    echo "ACCEPTANCE $MODE maps: FAIL"; rc=1
  fi
  "$PY" - "$RUN" "$REF" "$N" <<'EOF'
import json, sys
from pathlib import Path
import cv2, numpy as np
run, ref, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rep = json.load(open(run / "compare_reference.json"))
st = rep.get("stats", {})
for m, v in st.items():
    if m in ("ensemble", "ct"):
        continue
    print(f"stats {m:8s} " + " ".join(f"{k} {d['ours']:.5f}/{d['reference']:.5f}" for k, d in v.items() if d))
e = st.get("ensemble", {})
print("ensemble stretch", {k: (round(d["ours"], 5), round(d["reference"], 5)) for k, d in e.items() if d})
g = rep.get("grey_images", {}).get("images", [])
if g:
    eq = [x["frac_equal"] for x in g if "frac_equal" in x]
    print(f"grey images: {len(eq)} compared, min share of equal pixels {min(eq):.6f}" if eq else "grey images: none")
a, b = run / "image" / f"{n}.png", ref / "image" / f"{n}.png"
if a.is_file() and b.is_file():
    x, y = cv2.imread(str(a)), cv2.imread(str(b))
    if x.shape == y.shape:
        d = np.abs(x.astype(int) - y.astype(int))
        print(f"final image: share of equal pixels {float((d == 0).mean()):.6f}, max diff {int(d.max())}")
    else:
        print("final image: shapes differ", x.shape, y.shape)
EOF
  durable_copy "$RUN" "pherc0343/run_$MODE"
done
exit $rc
