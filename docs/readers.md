# Readers

All weights are fetched from their authors' repositories at pinned revisions and checked against their sha256
(`src/inksweep/readers/checkpoints.py`). Licences are those the authors published.

| Reader | Model, authors | Weights (pinned) | Licence | Window / tile / stride / µm | Recipe notes |
|---|---|---|---|---|---|
| `ink9_s42`, `ink9_s43`, `ink9_s{42,43}_{010..075}k` | `ink_9um` hybrid 3D-stem / 2D U-Net, Vesuvius Challenge | `scrollprize/ink_9um@7109667e` | MIT | 17 / 128 / 64 / ~9 | villa `infer`, robust MAD, fp16, (p − 0.25)/0.5 |
| `ink9_s43_50-75k` | checkpoint ensemble of seed 43 steps 50k, 60k, 75k | as above | MIT | 17 / 128 / 64 / ~9 | probability mean per tile (villa `--ensemble-checkpoint`) |
| `ink9_soup42` | weight average of seed-42 steps 40k–75k, E. Nieuwlaar | `Nieuwlaar/ink9um-dense-native@b894a049` | MIT | 17 / 128 / 64 / ~9 | single-trajectory soup (never average across seeds) |
| `dnative` | dense_native step 16 000, E. Nieuwlaar | `Nieuwlaar/ink9um-dense-native@b894a049` (safetensors + training config) | MIT | 17 / 128 / 64 / ~9 | wrapped in memory as `rebuild_dense_native_pth.py` does |
| `readerv2` | Reader v2, D. Russo | `domenicor046/reader-v2@72633d8a` | MIT | 17 / 128 / 64 / ~9 | drop-in villa checkpoint |
| `readerv3` | Reader V3, D. Russo | `domenicor046/reader-v3@aee1d64b` | MIT | 17 / 384 / 192 / ~9 | trained on 384 px tiles |
| `d9v4c`, `d9v4c_seed1` | d9v4C shape-loss reader, TAUIL Abd Elilah | GitHub release v1.0 of `9um-reader-letter-shape` | MIT | 17 / 128 / 64 / ~9 | forward depth order on their renders |
| `hecate` | Hecate 9.6 µm, Vesuvius Challenge | `scrollprize/hecate@9cb86e50` (+ `hecate.py`, sha256-checked) | MIT | 16 / 64 / 32 / 9.6 | resampled to 9.6 µm in 3 axes (8.64 µm: `INTER_AREA`, 9.362 µm: `INTER_LINEAR`); `depth-mean` gives the ±1 ensemble |
| `hecate_2.4` | Hecate 2.4 µm | as above | MIT | 64 / 256 / 128 / 2.4 | `/200`, depth margin 1 |
| `v8in` | v8-in ResNet3D-50, Y. Nader | `YoussefMoNader/ink-8um-v8in@d89166b4` (+ `ink8um`, sha256-checked) | MIT | 24 / 64 / 21 / ~8 | tiles inside the closed coverage mask, clip 200, **stitching fixed** (`v8in_released` keeps the released division by tile count) |
| `canonical_2um` | canonical ResNet-152 3D, Vesuvius Challenge | `scrollprize/ink_canonical_2um@075855bc` | MIT | 62 (centred) / 256 / 128 / 2.4 | villa `optimized_inference` recipe; for 2.4 µm inputs |

Recipes: `pherc0343` = `dnative, hecate, ink9_s42, ink9_s43` (in the order of the recipe's float32 ensemble sum);
`best9um` = `readerv3, d9v4c, dnative, hecate, v8in, ink9_s43_50-75k`.

## Writing a reader plugin

A plugin is a module that defines `READER = inksweep.readers.ReaderSpec(...)`:

```python
from inksweep.readers import ReaderSpec

READER = ReaderSpec(
    name="myreader", window_layers=17, voxel_size_um=9.36, tile=128, stride=64,
    load=lambda checkpoint_path, device: build_my_model(checkpoint_path).to(device).eval(),
    preprocess=lambda stack_uint8: my_normalise(stack_uint8),       # [W, tile, tile] uint8 -> model input
    forward=lambda module, batch: module(batch),                    # batch = stacked preprocess outputs
    postprocess=lambda out: out.float().sigmoid(),                  # -> probability [B, tile, tile] or [B, 1, ...]
)
```

```bash
inksweep run SURFACE --out OUT --plugin myreader.py --readers myreader --checkpoint myreader=/path/weights.pth
```

The engine owns the depth window (`s = L//2 + d − W//2`, model depth order applied), the tile grid, padding, blending,
resampling and writing; `forward` runs under the mode's precision context, `postprocess` outside it. Optional fields
(`batched`, `blend`, `grid`, `stitch`, `min_weight`, `zero_empty_columns`, `resample_tolerance`, `calibration`,
`reference_storage`, `amp_dtype`, `reference_batch`, ...) are documented in `src/inksweep/readers/spec.py`; the built-in
adapters use them to reproduce their reference implementations.
