# Validation

All runs on one H100 80 GB with 12 CPU cores, the PHerc. 343 segment `concat_w047-w048_R5B2_z9500-11000` rendered by
`inksweep render` (66 layers), the reference chain run by `node/run_inksweep_reference.sh` on the same render.
Commands, in order: `node/setup_inksweep.sh`, `node/run_inksweep_343.sh` (MODE=exact, then MODE=fast),
`node/run_inksweep_reference.sh`, `node/run_inksweep_compare.sh`, `node/run_inksweep_bench.sh`, and for the renderer
`node/run_inksweep_vc3d_compare.sh` (needs a VC3D build, `VC3D_APPRUN=...`).

## Equivalence with the reference chain

Every map: 4 models × 31 depths × 2 orientations, tile phases merged. Differences are in the reference's own 8-bit
units. "Pass" is the strict criterion (exact: r ≥ 0.9999 and every pixel within 1/255; fast: r ≥ 0.999).

| Model | Mode | Maps | Min r | Max abs diff | Share of equal pixels (min) | Pass |
|---|---|---|---|---|---|---|
| dnative | exact | 62 | 0.9999961 | 1/255 | 0.985 | 62/62 |
| hecate | exact | 62 | 1.0000000 | 0/255 | 1.000 (bit-identical) | 62/62 |
| ink9_s42 | exact | 62 | 0.9999917 | 4/255 | 0.975 | 42/62 |
| ink9_s43 | exact | 62 | 0.9999939 | 3/255 | 0.981 | 15/62 |
| all four | fast | 248 | 0.9994812 | 37/255 (Hecate) | 0.878 (Hecate; 0.978 for the villa models) | 248/248 |

The exact-mode residue on the two `ink_9um` checkpoints is the reference's own `torch.compile` rounding, not an
inksweep difference: on one such map (`ink9_s42`, d = −10, plain tile phase), villa's own eager run (`--no-compile`)
differs from its compiled run by the same amount (r 0.999992, at most 4/255, 97.6 % of pixels equal), and inksweep's
exact map is **bit-identical to villa's eager run** (0/255, every pixel, both orientations).

Recipe statistics, exact mode against the reference: the per-model p1/p99.5 stretch values are equal, the per-model
means and SDs agree to 4 decimals, the ensemble stretch agrees to 3 decimals (p1 −0.7182 vs −0.7181, p99.5 3.8637 vs
3.8636), and the final image (the release's own `make_submission_image.py` on inksweep's maps) has 99.5 % equal
pixels (fast mode: 98.0 %).

## Speed

The PHerc. 343 recipe, 248 maps (4 models × 31 depths × 2 orientations, the two tile phases merged), on one H100
80 GB. The reference at the author's defaults and both inksweep runs used the same 12 CPU cores; other jobs ran on the
node's other GPUs and cores during all runs. Wall times come from `node/run_inksweep_reference.sh` (`timing.json`) and
`node/run_inksweep_343.sh` (`ink.json`), side by side in `node/run_inksweep_bench.sh`; tiles per second are inksweep's
own per-reader counters (`ink.json`). The reference has no per-tile counter.

| Pipeline | Sweep wall time | Total (+ merge, ensemble, image) | Speedup (sweep, total) | Tiles/s: dnative, hecate, ink9_s42, ink9_s43 |
|---|---|---|---|---|
| reference: the release scripts at the author's defaults (`P_VILLA=2 P_HEC=2`) | 10329 s (plain phase 4798 s + padded phase 5531 s; no failed job) | 10431 s | 1× | n/a |
| reference with more parallelism (`P_VILLA=4 P_HEC=4`, 40 CPU cores) | 7773 s | 7875 s | 1.33×, 1.32× | n/a |
| inksweep exact | 5254 s | 5290 s | 1.97×, 1.97× | 335, 178, 334, 326 |
| inksweep fast (`--compile`) | 1828 s | 1864 s | **5.65×, 5.60×** | 540, 512, 1212, 1215 |

Given 3.3× more CPU cores (the second row), the reference is still 4.3× slower than fast mode. The second row is
`P_VILLA=4 P_HEC=4 CPUS=<40 cores> REF_NAME=reference_p4 bash node/run_inksweep_reference.sh`.

**Eight GPUs.** The same fast run with `--gpus 0,1,2,3,4,5,6,7` on one 8 × H100 node (48 CPU cores; only light CPU
jobs ran on the node's other cores): **577 s sweep, 613 s total**, about ten minutes; the sweep is 17.9× faster than
the reference's on one GPU. Scaling is 3.2× over one GPU, not 8×: each worker compiles its own models, and the 62
Hecate maps, the slowest, make the tail of the shared job queue. The command is step 4 of `node/run_inksweep_343.sh`
with `--gpus` added and its own `--out`.

**Where the time goes.** `inksweep bench --profile` splits every map's time into `read` (the depth window: gather or
resampling, and the tiles cut from it), `schedule` (tile grid, empty and needed tiles), `normalise` (the reader's
preprocessing), `forward` (network and postprocessing), `blend` (tile weighting and accumulation, the division, the
phase merge) and `storage` (copy to the host, calibration, writing), with a CUDA synchronisation at every stage
boundary. Profiling is off by default; without it the timers are no-ops and the sweep runs the same kernels in the same
order. In fast mode Hecate takes 1141 s of the 1828 s sweep (512 tiles/s, against about 1210 for the two `ink_9um`
checkpoints and 540 for dnative). Shares of each map's time on the 2048 × 2048 crop (`node/run_inksweep_bench.sh`,
steps 3 and 4):

| Reader, mode | read | schedule | normalise | forward | blend | storage | forward ms/call |
|---|---|---|---|---|---|---|---|
| ink9_s42, fast | 0 % | 0 % | 24 % | 56 % | 0 % | 18 % | 26.9 |
| hecate, fast | 0 % | 0 % | 0 % | 99 % | 0 % | 1 % | 99.8 |
| ink9_s42, exact | 0 % | 0 % | 7 % | 86 % | 1 % | 5 % | 9.0 |
| hecate, exact | 1 % | 0 % | 0 % | 98 % | 1 % | 1 % | 21.2 |

Hecate in fast mode at batch 32 (the default), 64, 128 and 256 tiles: 314, 317, 318 and 317 tiles/s (with
`--compile`: 602, 609, 618 and 599). Its forward pass is compute-bound, so larger batches do not help; `--compile`
doubles it, and the full fast sweep above already uses it.

## GPU memory

Peak memory per reader and mode, and what fits a 24 GB card, are in the guide's [GPU memory](guide.md#gpu-memory)
section (`node/run_inksweep_bench.sh`, step 5): peak 6.3 GiB in exact mode and 12.2 GiB in fast mode, both under a
22 GiB cap.

## Rendering: inksweep render against VC3D

The recipe's renders were made with VC3D 41701aa (GPL-3.0), from an AppImage of which the release's `ink/Dockerfile`
says: the "latest" release URL its author downloaded it from "no longer serves this build" (the villa releases list no
build of 41701aa today). A pinned MIT renderer makes the recipe reproducible from its meshes alone.
`node/run_inksweep_vc3d_compare.sh` renders the PHerc. 343 segment with a VC3D build given by `VC3D_APPRUN` (the
release's own `mesh/render_66_layers.sh` command line: 66 slices, slice step 1, no `--surface-interpolation`) on a
local level-0 copy of the CT chunks `inksweep render` read, so that both renderers sample the same bytes, and compares
the two renders layer by layer with `inksweep compare-render`. With the current release build, VC3D 322cb08
(24 Oct 2026):

| VC3D 322cb08 against `inksweep render` | all 66 layers (705 million voxels) | d = +1 (the image's layer) |
|---|---|---|
| Pearson r (the lowest layer) | 0.999893 (d = −15) | 0.999899 |
| largest difference (grey levels) | 1 | 1 |
| voxels equal / within one grey level | 50.0 % / 100 % | 50.0 % / 100 % |
| pixels whose valid mask agrees (the lowest layer) | 0.999995 | 0.999997 |

Where the two differ, VC3D 322cb08 is exactly one grey level lower, at half of the voxels: a rounding convention, not
a geometric difference (the ±2 px shift search matches best at (0, 0)). `inksweep render` truncates like the VC3D
build its renderer was measured against when it was ported. VC3D is only run as an external program by that script;
nothing of it is in this repository.

## Comparing readers

These tables establish equivalence, not which reader is better. To compare readers on labelled segments, use a paired
spatial block bootstrap of the AUROC difference (as in ChullMotors/ink-nullkit): today's held-out sets have only a few
dozen independent 256 px blocks, so differences below about 0.03 AUROC cannot be resolved.
