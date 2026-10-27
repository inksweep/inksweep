# inksweep

**GPU-resident ink-detection depth sweeps for Herculaneum scroll surface volumes.**

inksweep runs the recipe that won the PHerc. 343 First Letters prize (Erwin Nieuwlaar, October 2026) and reproduces
its maps: four models at every depth d = −15..+15, two tile phases and both orientations, in one process per GPU. The
other public 9 µm readers and the canonical 2 µm model run behind the same plugin contract, and `inksweep render`
turns tifxyz meshes into surface volumes without VC3D.

On the PHerc. 343 recipe (248 maps):

- **5.65× faster** than the reference chain on one H100: 30 min in fast mode against 2 h 52 min (exact mode: 1 h 28
  min, 1.97×). About 10 min on 8 × H100.
- **The same maps.** In exact mode every map is within 4/255 of the reference, the size of the reference's own
  `torch.compile` noise (bit-identical to villa's eager run where checked); in fast mode every map has r ≥ 0.9994.
- **The same render without VC3D:** every voxel within one grey level of VC3D 322cb08.
- **Fits a 24 GB card:** peak 6.3 GiB in exact mode, 12.2 GiB in fast mode.

Every number comes from a script in `node/`; see [Validation](https://github.com/inksweep/inksweep/blob/main/docs/validation.md).

**Why it is faster.** The reference chain starts one Python process per model and depth, compiles the model in each,
reads every tile from disk, normalises it on the CPU and writes every map as an 8-bit file. inksweep loads the surface
volume onto the GPU once, keeps each model loaded for all its maps, normalises tiles on the GPU (bit-identical to the
CPU code) and writes no intermediate file.

## Quick start

One Linux machine with an NVIDIA GPU, git and a **final** CPython 3.14: the organisers' `vesuvius` package needs 3.14,
and torch crashes on 3.14 pre-releases (which uv older than 0.9 picks).

```bash
uv venv --seed --python 3.14 ~/venvs/inksweep && source ~/venvs/inksweep/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129    # the build for your CUDA driver
pip install "inksweep[villa]"
pip install --no-deps "vesuvius @ git+https://github.com/ScrollPrize/villa.git@0fb38c45b488e192566ea83ae36369cb6db92652#subdirectory=vesuvius"

inksweep fetch --readers dnative,hecate,ink9_s42,ink9_s43 --ckpt-dir ckpt   # the recipe's checkpoints, 1.07 GB
inksweep doctor --ckpt-dir ckpt                                             # one line per check, with fixes

# the PHerc. 343 segment from the winning release, rendered from the open-data bucket
git clone https://github.com/Nieuwlaar/pherc343-first-letters && git -C pherc343-first-letters checkout e8370f1f
REL=pherc343-first-letters N=concat_w047-w048_R5B2_z9500-11000
VOL=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr
inksweep render $REL/outputs/$N.tifxyz --volume $VOL --cache ct_cache --out sv --layers 66 \
    --normal-sign keep --umbilicus $REL/inputs/umbilicus_PHerc0343.json --voxel-um 8.64

# the recipe (add --mode fast --compile for fast mode), then its image at d = +1
inksweep run sv --out run --name $N --recipe pherc0343 --ckpt-dir ckpt --area 380,105,6021,2560
xdg-open run/ink/preview_d+01.png
```

For your own segment, the [guide](https://github.com/inksweep/inksweep/blob/main/docs/guide.md#use-it-on-your-own-segment)
covers rendering against the scroll axis, a first look in fast mode, other readers and renders you already have.

## Commands

| Command | What it does |
|---|---|
| `inksweep run` | the depth sweep, ensemble and images, with everything the run did in `ink.json`; resumable, `--gpus` for several GPUs |
| `inksweep render` | tifxyz mesh + OME-Zarr volume → surface volume, without VC3D |
| `inksweep compare` | each map against the reference chain's |
| `inksweep compare-render` | two renders of one mesh, layer by layer |
| `inksweep bench` | wall time, tiles per second and GPU memory per reader and mode (`--profile`: per stage) |
| `inksweep doctor` | checks the environment and prints a fix for every problem |

Also `readers`, `fetch`, `ensemble`, `depth-mean`, `export-ct` and `export-reference`.

## Documentation

- [Guide](https://github.com/inksweep/inksweep/blob/main/docs/guide.md): install, your own segment, commands, exact
  and fast mode, GPU memory, inputs and outputs, rendering, FAQ.
- [Readers](https://github.com/inksweep/inksweep/blob/main/docs/readers.md): the built-in readers with their authors,
  pinned weights and licences, and how to add your own.
- [The PHerc. 343 recipe](https://github.com/inksweep/inksweep/blob/main/docs/recipe.md): each step of the reference
  and how inksweep runs it, and the choices where the reference was ambiguous.
- [Validation](https://github.com/inksweep/inksweep/blob/main/docs/validation.md): equivalence, speed, memory and
  renderer measurements, with the scripts that produce them.

## Development

```bash
pip install -e ".[test]" && pytest -q        # CPU only: CUDA and vesuvius tests skip without them
```

CI runs these tests on Python 3.12 and 3.14; a `v*` tag equal to `__version__` publishes the release to PyPI. The GPU
validation is the `node/` scripts. Changes are listed in
[CHANGELOG.md](https://github.com/inksweep/inksweep/blob/main/CHANGELOG.md).

## Credits and licence

inksweep is MIT-licensed (`LICENSE`); third-party portions are listed with their licences in `NOTICE`. It builds on:

- E. Nieuwlaar, *PHerc. 343: First Letters* (2026), github.com/Nieuwlaar/pherc343-first-letters, and
  *ink9um-dense-native*, huggingface.co/Nieuwlaar/ink9um-dense-native.
- Vesuvius Challenge, ScrollPrize/villa (ink detection, `ink_9um`, Hecate, canonical 2 µm), github.com/ScrollPrize/villa.
- D. Russo, Reader v2 and Reader V3, huggingface.co/domenicor046.
- TAUIL Abd Elilah, d9v4C, github.com/TAUIL-Abd-Elilah/9um-reader-letter-shape.
- Y. Nader, v8-in, huggingface.co/YoussefMoNader/ink-8um-v8in.
- J. Wallis (EconLearn), *First Letters workflow on known text*, github.com/EconLearn/vesuvius-first-letters-stage-loss
  (clean-room renderer, exact GPU normalisation, v8-in stitching fix).
- M. Bulloni, v8in-12gb (geometric depth order), github.com/Bullo27/v8in-12gb.
- ArcheyChen, vesuvius-eligible-scouting (Hecate resampling and ±1 depth ensemble), github.com/ArcheyChen.
- C. Hull, ink-nullkit (paired block bootstrap), github.com/ChullMotors/ink-nullkit.

Scan data and everything derived from it (renders, maps, images) are under the Vesuvius Challenge data licence
(CC BY-NC 4.0). No scan data, map or model weight is included in this repository.
