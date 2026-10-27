#!/bin/bash
# run_inksweep_bench.sh - speed of the reference chain against inksweep exact and fast mode, same GPU, same work;
# inksweep's GPU memory.
#
#   GPUS=0 CPUS=0-11 [STEPS="1 2 3 4 5"] [MEM_CAP_GB=22] bash node/run_inksweep_bench.sh
#
# 1 full sweep (PHerc. 343: 4 models x 31 depths x 2 orientations x 2 tile phases): the wall time of
#   run_inksweep_reference.sh (timing.json, every $DATA/pherc0343/reference*/) against the wall time of
#   run_inksweep_343.sh in exact and fast mode (ink.json); all must have run on the same GPU type with the same pinning.
# 2 per-map throughput on a 2048 x 2048 crop at d = 0 (reading side, one phase): inksweep bench with exact, fast and
#   fast + torch.compile; seconds per map and tiles per second per reader.
# 3 where the time goes, on the same crop at d = -1..+1: inksweep bench --profile (read, schedule, normalise, forward,
#   blend, storage per map; CUDA-synchronised, so shares count, not the profiled tiles/s)
# 4 Hecate in fast mode at batches 32 (the default), 64, 128 and 256 on the crop, unprofiled: is it launch-bound?
# 5 GPU memory on the full render at d = +1 with both tile phases: the peak per reader in exact mode, fast mode and
#   fast mode with --batch 16, then the same capped at MEM_CAP_GB GiB (--max-gpu-mem-gb: a 24 GB card emulated).
# STEPS selects steps (default: all).
# Prints: SPEEDUP exact / fast against each reference, the per-reader tables, and the memory summary (peak reserved
# GiB per reader and configuration; "fits" / "OUT OF MEMORY" under the cap).
set -euo pipefail
source "$(dirname "$0")/common.sh"
log_to run_inksweep_bench
require_ready
N=concat_w047-w048_R5B2_z9500-11000
W=$DATA/pherc0343
B=$W/bench
READERS=dnative,hecate,ink9_s42,ink9_s43
STEPS=${STEPS:-1 2 3 4 5}
CROP=(--rows 256:2304 --cols 2048:4096)
MEM_CAP_GB=${MEM_CAP_GB:-22}
mkdir -p "$B"
want() { [[ " $STEPS " == *" $1 "* ]]; }

if want 1; then
echo "-- 1 full sweep against the reference"
"$PY" - "$W" <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
# every reference run: $W/reference (the author's defaults) and others timed beside it (REF_NAME=reference_*)
refs = {p.parent.name: json.load(open(p)) for p in sorted(w.glob("reference*/timing.json"))}
rows = []
for mode in ("exact", "fast"):
    p = w / f"run_{mode}" / "ink.json"
    if p.is_file():
        j = json.load(open(p))
        maps = sum(1 for m in j["maps"] if not m.get("skipped"))
        tps = {r: round(v["tiles"] / max(v["seconds"], 1e-9)) for r, v in j["timings"]["per_reader"].items()}
        rows.append((mode, j["timings"]["sweep_wall_s"], j["timings"].get("total_s"), maps, j["software"].get("gpu"),
                     tps))
for name, ref in refs.items():
    print(f"{name}: sweep wall {ref['sweep_wall_s']} s (plain {ref['sweep_plain_s']} s + padded "
          f"{ref['sweep_padded_s']} s) for depths {ref['depths']} (P_VILLA={ref['p_villa']}, P_HEC={ref['p_hec']}), "
          f"total with merge/ensemble/image {ref['total_s']} s, failed jobs "
          f"{ref['plain']['failed'] + ref['padded']['failed']}")
for mode, sweep, total, maps, gpu, tps in rows:
    print(f"inksweep {mode}: sweep wall {sweep:.0f} s, total {total:.0f} s, {maps} maps, gpu {gpu}, tiles/s {tps}")
    for name, ref in refs.items():
        print(f"  SPEEDUP vs {name}: sweep {ref['sweep_wall_s'] / sweep:.2f}x, total {ref['total_s'] / total:.2f}x")
json.dump(dict(reference=refs.get("reference"), references=refs,
               inksweep=[dict(mode=m, sweep_wall_s=s, total_s=t, maps=n, gpu=g, tiles_per_s=tp)
                         for m, s, t, n, g, tp in rows]), open(w / "bench" / "full_sweep.json", "w"), indent=1)
EOF
fi

if want 2; then
echo "-- 2 per-map throughput on a crop"
pinned "$IS" bench "$W/sv" --out "$B/crop" --readers "$READERS" --depths 0 "${CROP[@]}" \
    --configs exact,fast,fast-compile --ckpt-dir "$CKPT" --no-download
fi

if want 3; then
echo "-- 3 stage profile on the crop (CUDA-synchronised timers)"
pinned "$IS" bench "$W/sv" --out "$B/profile" --readers "$READERS" --depths=-1..1 "${CROP[@]}" \
    --configs exact,fast,fast-compile --profile --ckpt-dir "$CKPT" --no-download
fi

if want 4; then
echo "-- 4 Hecate, fast mode, larger batches (unprofiled)"
for bs in 32 64 128 256; do
  pinned "$IS" bench "$W/sv" --out "$B/hecate_b$bs" --readers hecate --depths=-1..1 "${CROP[@]}" \
      --configs fast,fast-compile --batch "$bs" --ckpt-dir "$CKPT" --no-download
done
"$PY" - "$B" <<'EOF'
import json, sys
from pathlib import Path
b = Path(sys.argv[1])
print("| Hecate batch | fast tiles/s | fast-compile tiles/s | peak reserved GiB (fast) |")
print("|---|---|---|---|")
for bs in (32, 64, 128, 256):
    p = b / f"hecate_b{bs}" / "bench.json"
    if not p.is_file():
        continue
    cells = {e["label"]: e["readers"].get("hecate", {}) for e in json.load(open(p))["configs"]}
    f, c = cells.get("fast", {}), cells.get("fast-compile", {})
    tps = lambda r: "out of memory" if r.get("oom") else (f"{r['tiles_per_s']:.0f}" if r.get("tiles_per_s") else "-")
    peak = f.get("gpu_peak_reserved_mb")
    print(f"| {bs} | {tps(f)} | {tps(c)} | {peak / 1024:.1f} |" if peak else f"| {bs} | {tps(f)} | {tps(c)} | - |")
EOF
fi

if want 5; then
echo "-- 5 GPU memory: full render, d = +1, both tile phases; then capped at $MEM_CAP_GB GiB (a 24 GB card emulated)"
MEMARGS=(--readers "$READERS" --depths 1 --phase 0,0 --phase 40,200 --ckpt-dir "$CKPT" --no-download)
pinned "$IS" bench "$W/sv" --out "$B/mem" --configs exact,fast "${MEMARGS[@]}"
pinned "$IS" bench "$W/sv" --out "$B/mem_b16" --configs fast --batch 16 "${MEMARGS[@]}"
pinned "$IS" bench "$W/sv" --out "$B/mem_cap" --configs exact,fast --max-gpu-mem-gb "$MEM_CAP_GB" "${MEMARGS[@]}"
pinned "$IS" bench "$W/sv" --out "$B/mem_cap_b16" --configs fast --batch 16 --max-gpu-mem-gb "$MEM_CAP_GB" \
    "${MEMARGS[@]}"
"$PY" - "$B" "$MEM_CAP_GB" <<'EOF'
import json, sys
from pathlib import Path
b, cap = Path(sys.argv[1]), sys.argv[2]
def load(name):
    p = b / name / "bench.json"
    return json.load(open(p)) if p.is_file() else {"configs": []}
def rows(report, suffix=""):
    out = {}
    for e in report["configs"]:
        for reader, r in e.get("readers", {}).items():
            out[(reader, e["label"] + suffix)] = r
    return out
free = {**rows(load("mem")), **rows(load("mem_b16"), " --batch 16")}
capped = {**rows(load("mem_cap")), **rows(load("mem_cap_b16"), " --batch 16")}
v = load("mem").get("shape")
print(f"surface volume on the GPU: {v} = {v[0] * v[1] * v[2] / 2**30:.2f} GiB" if v else "no memory run found")
print(f"| reader | configuration | peak reserved GiB | peak allocated GiB | under a {cap} GiB cap |")
print("|---|---|---|---|---|")
for key in sorted(free):
    r, c = free[key], capped.get(key, {})
    res, alloc = r.get("gpu_peak_reserved_mb"), r.get("gpu_peak_allocated_mb")
    fits = "OUT OF MEMORY" if c.get("oom") else ("fits" if c else "not run")
    print(f"| {key[0]} | {key[1]} | {res / 1024 if res else float('nan'):.2f} | "
          f"{alloc / 1024 if alloc else float('nan'):.2f} | {fits} |")
EOF
fi
durable_copy "$B" pherc0343/bench
