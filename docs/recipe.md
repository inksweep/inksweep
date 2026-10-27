# The PHerc. 343 recipe

The reference is [Nieuwlaar/pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters) at commit
`e8370f1f` (MIT), whose ink maps were made with ScrollPrize/villa at `0fb38c45`.

| Step | Reference (release script) | inksweep |
|---|---|---|
| Surface volume | 66 layers, layer k at k − 32.5 voxels along the mesh normal, d = k − 33, + away from the scroll axis | same contract; `inksweep render` makes one without VC3D |
| Models | dense_native (step 16 000), `ink_9um` seeds 42 and 43 (step 75 000), Hecate 9.6 µm | `--recipe pherc0343` (`dnative, hecate, ink9_s42, ink9_s43`) |
| Depths | `run_depth_sweep.sh`: d = −15..+15, one process per model and depth | one process per GPU, every depth |
| villa window | `run_ink.sh`: layers s..s+16, s = 33 + d − 8; `--direction both` | `window_indices(L, d, 17)`, both orientations |
| Hecate window | `hecate_prep.py`: 16 planes at 9.6 µm centred on layer 32.5 + d, linear in depth, `INTER_AREA` in plane | `resample.py` (transcription; OpenCV in exact mode) |
| Tiling | villa: 128 px tiles, stride 64, Hann blend floored at 0.001; Hecate: 64 px, stride 32, same window | `tiling.py`, identical grids and windows |
| Normalisation | villa: per-tile robust median/MAD on the CPU; Hecate: `/255` | `readers/norm.py`: the same values on the GPU, bit for bit |
| Tile phases | `pad_render.py` (40 rows, 200 columns of zeros) and `merge_tile_phase.py` (w047 half from the padded run) | virtual padding; `--phase 0,0 --phase 40,200 --phase-merge select --phase-mask npz:..._node_source.npz` |
| Storage | villa `trunc(255 p)` TIFF; Hecate `round(255 p)` PNG on the 9.6 µm grid | emulated (`reference_storage`), raw float maps kept too |
| Ensemble | `ensemble_maps.py`: per-model p1/p99.5 stretch over all depths, mean of per-model z-scores | `ensemble.py` (transcription), the same grey images and `stats.json` |
| Image | `make_submission_image.py` at d = +1, dimmed CT, ink in yellow, 1 cm scale bar | `ink/preview_d+01.png`; the node script also runs the release's own image script on inksweep's maps |

The [quick start](../README.md#quick-start) runs one tile phase. The release also merges a second phase on the w047
half of this segment; the flags are under [Commands](guide.md#commands), and `node/run_inksweep_343.sh` builds the
node-source map they need with the release's `mesh/build_concat.py`. Everything a run did (readers, checkpoints with
sha256, settings, timings, software versions) is in its `ink.json`.

## Choices where the reference was ambiguous

1. **Exact means the reference's precision, not float32.** The PHerc. 343 maps were made under fp16 autocast (villa
   `--amp-dtype auto` reads `mixed_precision: fp16` from the checkpoints), Hecate in float32. Exact mode does the same;
   `--precision fp32` is available.
2. **Eager instead of torch.compile.** villa `infer` compiles the model by default (the recipe did not pass
   `--no-compile`); exact mode runs eagerly and deterministically. Exact mode is bit-identical to villa run with
   `--no-compile`; the compiled reference differs from both by at most a few 8-bit steps
   ([Validation](validation.md#equivalence-with-the-reference-chain)).
3. **Reading side.** Under the contract (+d outward) the reading side is the `reverse` orientation, as in the recipe
   (its `_reverse` maps are the reading side).
4. **Tile phases.** The recipe's second phase exists for one segment (the w047 half is shifted by its concatenation);
   inksweep reproduces it with a node-source mask and also offers `--phase-merge mean` for other surfaces.
5. **v8-in stitching** is fixed by default; `v8in_released` reproduces the released maps.
6. **Hecate ±1 depth mean** (a community improvement, not in the recipe) is a post-step (`inksweep depth-mean`).
7. **Canonical 2 µm**: no reference command is published for this checkpoint; inksweep uses the settings of the
   community runs (centred 62 layers, 256 px tiles, stride 128, villa #1765/#1766).
8. **villa loads weights permissively** (`strict=False`); inksweep stops on any missing weight.
