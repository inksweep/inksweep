# Guide

1. [Install](#install)
2. [Use it on your own segment](#use-it-on-your-own-segment)
3. [Commands](#commands)
4. [Exact and fast mode](#exact-and-fast-mode)
5. [GPU memory](#gpu-memory)
6. [Inputs and outputs](#inputs-and-outputs)
7. [Rendering without VC3D](#rendering-without-vc3d)
8. [FAQ](#faq)

## Install

The [quick start](../README.md#quick-start) installs inksweep on one Linux machine with an NVIDIA GPU. You need git and
a **final** CPython 3.14: the organisers' `vesuvius` package, which builds the villa-family networks, requires Python
3.14, and torch's wheels crash on 3.14 pre-releases (uv older than 0.9 picks a pre-release for `--python 3.14`).

**vesuvius from a sparse checkout.** Installing `vesuvius` from git clones the whole villa repository. This fetches only
the two folders inksweep uses, as `node/setup_inksweep.sh` does; the `canonical_2um` reader needs such a checkout (it
imports villa's `ink-detection/optimized_inference`; `INKSWEEP_VILLA_DIR` points it to one):

```bash
git clone --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git villa
git -C villa sparse-checkout set --cone vesuvius ink-detection
git -C villa checkout 0fb38c45b488e192566ea83ae36369cb6db92652
pip install --no-deps -e villa/vesuvius
```

inksweep's own Hecate and v8-in adapters do not import `vesuvius`; they load the model code their authors publish
(pinned and sha256-checked, see [Readers](readers.md)).

**On a GPU node, by script.** From the repository deployed at `$ROOT/src/inksweep` (CUDA 12.9 driver or newer):

```bash
bash node/setup_inksweep.sh        # idempotent; writes $ROOT/venv-inksweep/READY
```

It needs uv >= 0.9 (with an older uv it installs a pinned uv under `$ROOT/tools` by itself) and installs CPython
3.14.10 (a venv with a pre-release interpreter is recreated), `torch==2.13.0+cu129` from the PyTorch index, the pinned
packages of `node/requirements.txt`, ScrollPrize/villa's `vesuvius` package at `0fb38c45` (`--no-deps`; only
`vesuvius/` and `ink-detection/` are checked out), inksweep, and every checkpoint with its sha256 checked. On Linux
x86_64 it installs OpenCV's manylinux2014 wheel (pinned by sha256): the manylinux_2_28 wheels bundle an OpenSSL whose
FIPS self-test aborts `import cv2` on FIPS-mode kernels. It also builds the reference pipeline's own venv and checkout
for the comparison scripts. To reproduce the node's environment by hand, install the pinned packages of
`node/requirements.txt` after torch, then inksweep from a clone with `pip install --no-deps -e .`.

**Checking an installation.** `inksweep doctor [--ckpt-dir DIR] [--readers LIST] [--out DIR]` prints one line per
check, with a `fix:` line under every warning and failure:

| Check | What it verifies |
|---|---|
| `python` | a final CPython release (torch's wheels crash on pre-releases); 3.14 for the `vesuvius` package |
| `packages` | inksweep's dependencies are installed (from their metadata) |
| `torch`, `cuda` | torch imports, has a CUDA build and a matching torchvision; every visible GPU with its total and free memory |
| `opencv` | `import cv2` works; on a FIPS-mode kernel the manylinux_2_28 wheels abort, and the fix installs the manylinux2014 wheel |
| `vesuvius` | the organisers' package imports with every module the villa-family readers use, and which villa commit it comes from |
| `checkpoint` | every pinned file of the readers is in `--ckpt-dir`, with its size and sha256 (`--no-hash`: sizes only) |
| `disk` | room for the checkpoints still to download, and at least `--min-free-gb` (default 50) for the outputs |

torch, cv2 and vesuvius are imported in subprocesses, so a crash is reported instead of ending the check. The exit
status is 1 when a check failed; `--json` prints the checks for a bug report.

## Use it on your own segment

1. **Inputs.** A tifxyz mesh (a folder with `x.tif`, `y.tif`, `z.tif` and `meta.json`, as VC3D writes it), the
   scroll's OME-Zarr volume (its URL on the open-data bucket, or a local copy), the volume's voxel size from its
   `metadata.json` (never from a file name), and the scroll axis as JSON (`{"control_points": [{"x": …, "y": …,
   "z": …}, …]}` in level-0 voxels).
2. **Plan the render.** This prints the render size, the share of mesh normals pointing away from the axis and the
   amount of CT to fetch, then stops:

   ```bash
   inksweep render MESH.tifxyz --volume VOLUME_URL --cache ct_cache --out sv --layers 66 \
       --normal-sign auto --umbilicus axis.json --voxel-um VOXEL_UM --plan-only
   ```

3. **Render** with the same command without `--plan-only`. `--normal-sign auto` stacks the layers so that +d points
   away from the scroll axis, and refuses a mesh whose normals point both ways (`--force` overrides). Without an axis
   file, `--normal-sign keep` stacks along the mesh normal as VC3D does by default; `surface.json` records the choice.
4. **Check the setup:** `inksweep doctor --ckpt-dir ckpt --out .` (every line `ok`).
5. **A first look** in fast mode at a few depths, then the full recipe:

   ```bash
   inksweep run sv --out quick --recipe pherc0343 --ckpt-dir ckpt --mode fast --depths=-3..3 --write-depths=-1..2
   inksweep run sv --out results --recipe pherc0343 --ckpt-dir ckpt          # d = -15..+15, exact mode
   ```

6. **Read the results** in `results/`: `ink/preview_d±NN.png` (the ensemble in yellow over the dimmed CT, 1 cm scale
   bar, at each `--write-depths` depth), `ink/ensemble.tif`, each reader's maps `ink/<reader>/d±NN.tif` (and `blind/`,
   the other orientation, as a control), `ink.json` (everything the run did).
   `inksweep ensemble results --surface sv --write-depths=-2..3` redraws the images at other depths without running
   the models again.
7. **Other readers:** `--readers readerv3,d9v4c,dnative,hecate,v8in,ink9_s43_50-75k` (the `best9um` list);
   `inksweep readers` lists them all, and your own model fits in as a [plugin](readers.md#writing-a-reader-plugin).
8. **A render you already have** (a VC3D zarr, a folder of `00.tif`, `01.tif`, …, or a multi-page TIFF) works as
   input with `--voxel-um VOXEL_UM`; layer i is read as depth d = i − L//2. If it is stacked towards the scroll axis,
   pass `--reading forward` (the depth labels then count inward), or render it again with `--normal-sign auto`.
   `inksweep compare-render YOURS.zarr sv` shows how it differs from inksweep's render of the same mesh.
9. **Tile phases:** `--phase 0,0 --phase 40,200 --phase-merge mean` averages two tile grids, so that a letter's maps do
   not depend on where it falls in the grid; the recipe's `select` merge is specific to its joined segment.

## Commands

```bash
inksweep readers                                          # built-in readers and recipes
inksweep fetch --readers dnative,hecate,ink9_s42,ink9_s43 --ckpt-dir ckpt

# the PHerc. 343 recipe on a surface volume (directory with layers.zarr + surface.json, a zarr render, or 00.tif ...)
inksweep run SURFACE --out OUT --recipe pherc0343 --ckpt-dir ckpt

# ... with the recipe's two tile phases, merged per half by the segment's node-source map
inksweep run SURFACE --out OUT --recipe pherc0343 --ckpt-dir ckpt \
    --phase 0,0 --phase 40,200 --phase-merge select --phase-mask npz:concat_node_source.npz:1,4

# other readers, fast mode, four GPUs, image at d = +1
inksweep run SURFACE --out OUT --readers readerv3,d9v4c,dnative,hecate,v8in,ink9_s43_50-75k \
    --depths=-15..15 --orientations both --mode fast --compile --gpus 0,1,2,3 --display-depth 1

inksweep compare OUT REFERENCE_MAPS --name SEGMENT_NAME --depths=-15..15   # equivalence report
inksweep bench SURFACE --out B --readers dnative,hecate --depths 0 --rows 0:2048 --cols 0:2048 --configs exact,fast
inksweep bench SURFACE --out B --readers hecate --depths=-1..1 --configs fast --profile   # time per stage
inksweep depth-mean OUT --model hecate --offsets -1,0,1                     # Hecate's +-1 depth ensemble

# two renders of one mesh, layer by layer (e.g. VC3D's against inksweep's); the CT cache as a local OME-Zarr
inksweep compare-render VC3D_RENDER.zarr SURFACE --names vc3d,inksweep --max-shift 2 --report cmp.json
inksweep export-ct --volume VOLUME_URL --cache ct_cache --out ct_level0.zarr
```

Write negative depth ranges as `--depths=-15..15` (with `=`), so that the shell argument is not read as an option.

Runs are resumable: each map has a marker (`ink/<reader>/.done/`); re-running skips finished maps and refuses to mix
maps made with other settings. `--gpus` starts one process per GPU on a shared job queue.

## Exact and fast mode

**Exact** (default) reproduces the reference numerics. Where the reference command is deterministic the maps are
bit-identical; elsewhere they differ only by the GPU's own run-to-run noise.

- Precision is the reference's: the villa readers run under fp16 autocast (villa's `--amp-dtype auto` takes the
  training precision from the checkpoint; every released checkpoint says fp16), Hecate in float32 with PyTorch's
  default TF32 policy for convolutions, v8-in and the canonical model under fp16 autocast. `--precision fp32` forces
  strict float32 everywhere instead.
- Batch size and batch composition of the reference command (villa 4, Hecate 4, v8-in 16): the reference's tile
  schedule is cut into batches and empty tiles are dropped within each batch as the reference drops them, so every
  forward pass has the reference's shape. cuDNN without autotuning, eager execution (`--cuda-graphs` replays the same
  kernels with less CPU overhead). Tiles are uploaded and normalised in large groups, so the GPU never waits for the
  host between the small batches.
- Tiles are added to the blend sums one by one in grid order, the reference's per-pixel addition order.
- Hecate's input resampling and the resize of its maps run in OpenCV, exactly as `hecate_prep.py` and
  `ensemble_maps.py` do.

The reference itself is not bit-reproducible: its author measured r 0.99999 with at most 1/255 difference between
GPU types, villa compiles the model with `torch.compile` by default, and cuDNN may choose other algorithms. The
acceptance for exact mode is therefore r ≥ 0.9999 and max |difference| ≤ 1/255 for every map.

**Fast** uses TF32, the readers' training autocast dtype (bf16 when they have none), cuDNN autotuning, channels-last
weights, large batches, vectorised blending, GPU resampling, and optionally `--compile` (torch.compile) and
`--cuda-graphs`. Acceptance: r ≥ 0.999 for every map.

Both modes run, with a `select` phase merge, only the tiles that touch the pixels the merge takes from each phase.
This halves the tile-phase cost and changes no output pixel.

## GPU memory

A sweep keeps the whole surface volume on the GPU (L × H × W bytes, `surface.volume_mb` in `ink.json`), one reader
at a time with its weights, and that reader's working buffers: the tiles being normalised (groups of up to 512 MiB of
float32 input), the forward batch, and two float32 blend sums of the canvas size. `ink.json` records the peak per
map and per reader (`timings.per_reader.<reader>.gpu_peak_reserved_mb`, what PyTorch's caching allocator held, and
`gpu_peak_allocated_mb`, what tensors used); `inksweep bench` prints it. The CUDA context comes on top.

Peak reserved memory on the PHerc. 343 render at d = +1, both tile phases (`node/run_inksweep_bench.sh`, step 5):

| Reader | exact | fast (default batch) | fast `--batch 16` |
|---|---|---|---|
| dnative | 5.1 GiB | 6.5 GiB | 5.1 GiB |
| hecate | 6.3 GiB | 12.2 GiB | 9.9 GiB |
| ink9_s42 | 6.1 GiB | 6.0 GiB | 6.1 GiB |
| ink9_s43 | 6.1 GiB | 6.0 GiB | 6.1 GiB |

**24 GB cards.** Exact mode uses the reference's batches (villa 4, Hecate 4): every reader fits under a 22 GiB cap
(peak 6.3 GiB). Fast mode's default batches are 64 tiles for the 128 px villa readers (16 for Reader V3's 384 px
tiles), 32 for Hecate and v8-in, 4 for the canonical 2 µm model: the four recipe readers fit (peak 12.2 GiB, Hecate);
with `--batch 16`: peak 9.9 GiB. The surface volume itself takes 1.06 GiB here (66 × 2560 × 6720). Larger segments
need more memory for the volume and the blend sums. Before a long run on your own render, check with
`inksweep bench SURFACE --out B --depths 1 --configs exact,fast --max-gpu-mem-gb 22`: a reader that does not fit is
listed as out of memory.

**`--max-gpu-mem-gb G`** (`run` and `bench`) caps this process's CUDA caching allocator at G GiB on each GPU
(`torch.cuda.set_per_process_memory_fraction`, set before anything is loaded). It is a test aid to emulate a smaller
card: the CUDA context and memory that libraries allocate outside PyTorch's allocator are not counted, so leave a
margin (22 for a 24 GB card). A run that needs more stops with CUDA's out-of-memory error; `inksweep bench` records it
per reader and goes on.

## Inputs and outputs

Input: a uint8 surface volume `(L, H, W)` where layer `i` sits at depth `d = i − L//2` voxels along the surface
normal, positive away from the scroll axis. On such renders the readers' **reading side** is the `reverse` orientation
(the released readers read inward); `--reading forward` changes that for renders stacked the other way. The voxel size
comes from `surface.json` (`voxel_size_um`) or `--voxel-um`, never from a file name.

Output directory:

| Path | Content |
|---|---|
| `ink/<reader>/d±NN.tif` | float32 [0, 1], render grid: the reader's calibrated probability, reading side |
| `ink/<reader>/blind/d±NN.tif` | the same for the other orientation (the blind control) |
| `ink/<reader>/raw/<side>/d±NN.tif` | the blended probability before calibration, on the reader's grid (`inksweep compare` uses it) |
| `ink/ensemble.tif` | the stretched ensemble at the display depth, float32 [0, 1] |
| `ink/ensemble/d±NN.tif`, `ink/ensemble/blind/` | the mean of per-model z-scores at each written depth |
| `ink/preview_d±NN.png` | ensemble in yellow over the dimmed CT, reading orientation, 1 cm scale bar |
| `ink/grey/*.png`, `ink/stats.json` | the recipe's grey images and statistics, under the recipe's names |
| `ink/valid.tif` | pixels with CT data in every layer |
| `ink.json` | readers, checkpoints (sha256), every setting, timings and GPU memory peaks per map and reader, software versions |

**Map values.** The villa readers were trained with label smoothing 0.5, so their outputs sit between 0.25 and 0.75;
their maps are `clip((p − 0.25) / 0.5, 0, 1)` (the model-card rescale). With `reference_storage` on (default) this is
computed from the reference's own 8-bit value, `clip((trunc(255 p) − 64) / 128, 0, 1)`, which is exactly what the
recipe's ensemble reads; Hecate maps are `round(255 p) / 255` on its 9.6 µm grid, resized bilinearly to the render
grid. `--no-reference-storage` keeps full precision.

## Rendering without VC3D

```bash
inksweep render MESH.tifxyz --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/<SAMPLE>/volumes/<VOL>.zarr \
    --cache ct_cache --out SURFACE --layers 66 --normal-sign auto --umbilicus umbilicus.json --voxel-um 8.64
```

A port of EconLearn's clean-room renderer (see `src/inksweep/render.py`), which reproduces `vc_render_tifxyz`'s
documented behaviour (linear mode): pixel `(r, c)` at grid position `((r + 0.5) s, (c + 0.5) s)`, normal
`(P[r, c+1] − P[r, c−1]) × (P[r+1, c] − P[r−1, c])` at the nearest node, layer `i` at `(i − (n − 1)/2)` voxels along
it, trilinear sampling, truncation. Against VC3D's current release build every voxel is within one grey level
([Validation](validation.md#rendering-inksweep-render-against-vc3d)).

Layer order and `--flip-normals`: without flipping, layer index increases along the mesh normal (as
`vc_render_tifxyz` without `--flip-normals`, the PHerc. 343 render); `--normal-sign flip` stacks along −N (as
`--flip-normals`). The surface-volume contract wants +d away from the scroll axis, so `--normal-sign auto` measures the
share of outward normals against the scroll's umbilicus and flips when it is below one half, and refuses meshes whose
normals point both ways. The decision and the measured share are written to `surface.json`. CT chunks are fetched
once into a disk cache; a missing chunk is never cached as zeros, and a chunk missing where the volume's coarse level
shows data stops the render.

`inksweep compare-render A B` works on any two renders of one mesh (zarr, surface-volume folder or TIFF stack): it
matches layers by depth label `d = i − L//2`, prints one row per depth, and with `--max-shift N` also correlates the
d = +1 layer at every offset up to N pixels, which exposes a pixel-centre convention difference. `inksweep export-ct`
writes the CT chunks of an `inksweep render` cache as a local OME-Zarr (hard links for the uncompressed open-data
volumes), for programs that read the CT from disk.

## FAQ

**`import cv2` aborts ("core dumped") on a FIPS-mode kernel.** When `/proc/sys/crypto/fips_enabled` reads 1, OpenCV's
manylinux_2_28 wheels abort the process as they are imported: they bundle an OpenSSL whose FIPS self-test fails.
OpenCV's manylinux2014 wheel does not (Linux x86_64; the wheel `node/setup_inksweep.sh` pins):

```bash
pip install --no-deps --force-reinstall "opencv-python-headless @ https://files.pythonhosted.org/packages/2b/97/8170e9819764c47e436c130d3ff6cfb73b58f923eae9d3a03d8982b04aec/opencv_python_headless-5.0.0.93-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=09a872a157c1376ab922a69bbf22f9a95bcc7b658a9d8b436a60212b02b2eeb4"
```

`inksweep doctor` imports cv2 in a subprocess, so it reports the abort and prints this command.

**torch crashes with a segmentation fault on Python 3.14.** The interpreter is probably a 3.14 pre-release, on which
torch's wheels crash: `python -c "import sys; print(sys.version_info.releaselevel)"` must print `final`. uv older than
0.9 installs a pre-release for `--python 3.14`; update uv (or install CPython from python.org) and recreate the venv.
`inksweep doctor` checks the release level and imports torch in a subprocess.

**How much GPU memory does it need? I have a 24 GB card.** The surface volume stays on the GPU next to one reader at a
time; see [GPU memory](#gpu-memory) for the measured peaks and the `--batch` to use. `--max-gpu-mem-gb 22` on a larger
card shows whether a run fits before you start it.

**Does it run on Windows or macOS?** Untested. inksweep is developed and validated on Linux x86_64 with NVIDIA GPUs.
On Windows, WSL2 with CUDA is the likely route; `inksweep render`'s worker pool forks processes (`--procs 1` does
without). Reports are welcome.

**Which side of the papyrus do the maps show?** The reading side: under the surface-volume contract (+d away from the
scroll axis) that is the `reverse` orientation, as in the recipe. The other orientation is written to
`ink/<reader>/blind/` as a control (see [Inputs and outputs](#inputs-and-outputs)).
