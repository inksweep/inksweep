# Changelog

All notable changes to inksweep. Versions follow [semantic versioning](https://semver.org/).

## 0.1.0

First release.

- The sweep engine: the surface volume on the GPU once, every reader at every depth, tile phase and orientation in
  one process per GPU; **exact** mode (the reference numerics: precision, batch composition, accumulation order,
  OpenCV resampling) and **fast** mode (TF32, autotuning, large batches, optional `torch.compile`); resumable maps;
  several GPUs on one job queue.
- Readers behind one plugin contract (`ReaderSpec`): `ink_9um` and its checkpoint ensembles, dense_native, the
  seed-42 soup, Reader v2 and V3, d9v4C, Hecate 9.6 µm and 2.4 µm, v8-in (stitching fixed; the released stitching as
  `v8in_released`), the canonical 2 µm model; every checkpoint pinned by revision and sha256.
- The PHerc. 343 recipe (`--recipe pherc0343`): two tile phases merged per half, the ensemble, its statistics and
  images, the preview at d = +1.
- `inksweep render`: a clean-room MIT renderer of tifxyz meshes into surface volumes (port of EconLearn's), with the
  normal sign checked against the scroll's umbilicus; `compare-render` and `export-ct` to compare it with VC3D.
- `compare` and `export-reference` (equivalence with the reference chain's maps), `bench` (time, tiles per second and
  GPU memory per reader; `--profile` per stage; `--max-gpu-mem-gb` to emulate a smaller card), `doctor` (environment
  checks with a fix for every problem), `fetch`, `ensemble`, `depth-mean`.
- Node scripts for the PHerc. 343 run, the reference chain on the same render, the equivalence report, the speed and
  memory benchmark and the VC3D comparison.
- CI (CPU tests on Python 3.12 and 3.14) and a release workflow that publishes to PyPI with trusted publishing.
