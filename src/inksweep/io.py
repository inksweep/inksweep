"""Reading surface volumes and writing results.

Accepted inputs (:func:`open_surface`):

* a **surface-volume directory** (the surface-volume contract): ``layers.zarr`` (uint8 ``(L, H, W)``; an array, or a group
  holding array ``"0"``), ``surface.json`` (its ``voxel_size_um`` is used) and ``mesh.tifxyz/``;
* a **zarr** render: an array, or a group with array ``"0"`` (OME-Zarr level 0, as VC3D and the PHerc. 343 scripts
  write it);
* a **TIFF-stack directory**: ``00.tif, 01.tif, ...`` (numerically sorted, as ``vc_render_tifxyz`` writes them), or a
  directory with such a ``layers/`` folder; uint16 layers are reduced to 8 bit with ``>> 8`` (as v8-in reads them);
* a **multi-page TIFF** ``(L, H, W)``.

Layer ``i`` is taken to sit at depth ``d = i - L // 2`` (the contract); nothing is reordered on reading.

Writers are atomic (temporary file, then rename), so a killed run never leaves a truncated map behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

LAYER_FILE = re.compile(r"^(\d+)\.(tif|tiff)$", re.IGNORECASE)
VOXEL_KEYS = ("voxel_size_um", "voxel_um", "pixel_size_um", "voxel_size")


# ----------------------------------------------------------------------------------------------- inputs
@dataclass
class SurfaceVolume:
    """A uint8 ``(L, H, W)`` surface volume, read lazily."""

    path: Path
    name: str
    layers: Any
    kind: str
    voxel_size_um: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.layers.shape)  # type: ignore[return-value]

    @property
    def n_layers(self) -> int:
        return self.shape[0]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def width(self) -> int:
        return self.shape[2]

    def read(self, rows: Optional[tuple[int, int]] = None, cols: Optional[tuple[int, int]] = None) -> np.ndarray:
        """The whole volume (or a row / column crop) as a C-contiguous uint8 numpy array."""
        r0, r1 = rows or (0, self.height)
        c0, c1 = cols or (0, self.width)
        if isinstance(self.layers, np.ndarray):
            out = self.layers[:, r0:r1, c0:c1]
        else:
            out = np.asarray(self.layers[:, r0:r1, c0:c1])
        return np.ascontiguousarray(to_uint8(out))

    def iter_row_blocks(self, block: int = 256) -> Iterator[tuple[int, np.ndarray]]:
        """``(row0, uint8 block (L, rows, W))`` over the whole volume."""
        for r0 in range(0, self.height, block):
            yield r0, self.read(rows=(r0, min(self.height, r0 + block)))

    def valid_mask(self, block: int = 256) -> np.ndarray:
        """Pixels with CT data in every layer (``all layers > 0``), the PHerc. 343 recipe's valid rule."""
        valid = np.zeros((self.height, self.width), bool)
        for r0, blk in self.iter_row_blocks(block):
            valid[r0:r0 + blk.shape[1]] = (blk > 0).all(0)
        return valid


def to_uint8(a: np.ndarray) -> np.ndarray:
    """uint8 as is; uint16 reduced with ``>> 8``; anything else refused."""
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    raise TypeError(f"surface volumes must be uint8 (or uint16), got {a.dtype}")


def _open_zarr(path: Path) -> Any:
    import zarr

    node = zarr.open(str(path), mode="r")
    if hasattr(node, "shape"):
        return node
    try:
        return node["0"]
    except KeyError as exc:
        raise ValueError(f"{path}: zarr group without array '0'") from exc


def _layer_files(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and LAYER_FILE.match(p.name)]
    return sorted(files, key=lambda p: int(LAYER_FILE.match(p.name).group(1)))  # type: ignore[union-attr]


class TiffStack:
    """Lazy ``(L, H, W)`` view of numbered layer TIFFs (each read whole on first use and cached)."""

    def __init__(self, files: Sequence[Path]) -> None:
        import tifffile

        self.files = list(files)
        first = tifffile.imread(self.files[0])
        if first.ndim == 3:
            first = first[..., 0]
        self.shape = (len(self.files),) + tuple(int(v) for v in first.shape)
        self.dtype = first.dtype
        self._cache: dict[int, np.ndarray] = {0: to_uint8(first)}

    def _layer(self, i: int) -> np.ndarray:
        import tifffile

        if i not in self._cache:
            a = tifffile.imread(self.files[i])
            if a.ndim == 3:
                a = a[..., 0]
            if a.shape != self.shape[1:]:
                raise ValueError(f"{self.files[i]}: shape {a.shape} differs from {self.shape[1:]}")
            self._cache[i] = to_uint8(a)
        return self._cache[i]

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        zsel = key[0]
        rest = key[1:]
        idx = range(self.shape[0])[zsel] if isinstance(zsel, slice) else zsel
        if isinstance(idx, (int, np.integer)):
            return self._layer(int(idx))[rest] if rest else self._layer(int(idx))
        return np.stack([self._layer(int(i))[rest] if rest else self._layer(int(i)) for i in idx])


def read_surface_json(folder: Path) -> dict:
    p = folder / "surface.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def voxel_size_from_meta(meta: dict) -> Optional[float]:
    for key in VOXEL_KEYS:
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                pass
    return None


def open_surface(path: str | os.PathLike, voxel_size_um: Optional[float] = None,
                 name: Optional[str] = None) -> SurfaceVolume:
    """Open any supported surface volume (see module docstring). ``voxel_size_um`` overrides the metadata."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    meta: dict = {}
    if p.is_dir() and (p / "layers.zarr").exists():
        meta = read_surface_json(p)
        layers, kind = _open_zarr(p / "layers.zarr"), "contract"
    elif p.is_dir() and (p.suffix == ".zarr" or (p / ".zgroup").exists() or (p / ".zarray").exists()
                         or (p / "zarr.json").exists()):
        layers, kind = _open_zarr(p), "zarr"
    elif p.is_dir() and _layer_files(p):
        layers, kind = TiffStack(_layer_files(p)), "tiffdir"
    elif p.is_dir() and (p / "layers").is_dir() and _layer_files(p / "layers"):
        layers, kind = TiffStack(_layer_files(p / "layers")), "tiffdir"
        meta = read_surface_json(p)
    elif p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        layers, kind = tifffile.imread(p), "tiff"
        if layers.ndim != 3:
            raise ValueError(f"{p}: expected a multi-page (L, H, W) TIFF, got shape {layers.shape}")
        layers = to_uint8(layers)
    else:
        raise ValueError(f"{p}: not a surface volume (layers.zarr dir, zarr, TIFF-stack dir or multi-page TIFF)")
    if len(layers.shape) != 3:
        raise ValueError(f"{p}: expected (L, H, W), got {tuple(layers.shape)}")
    vox = voxel_size_um if voxel_size_um is not None else voxel_size_from_meta(meta)
    stem = name or (p.name[:-5] if p.name.endswith(".zarr") else p.name)
    return SurfaceVolume(path=p, name=stem, layers=layers, kind=kind, voxel_size_um=vox, meta=meta)


# ----------------------------------------------------------------------------------------------- tile-phase masks
def load_phase_mask(spec: str, shape: tuple[int, int]) -> np.ndarray:
    """Boolean render-grid mask for a two-phase ``select`` merge (True = take the second phase).

    * ``npz:PATH[:VALUES[:NODE_PX]]``: a node-source file as written by ``build_concat.py`` (array ``source`` per
      mesh node); pixels of nodes whose source is in ``VALUES`` (default ``1,4``: the w047 half and its transition
      strip in the PHerc. 343 segment) are True; one node covers ``NODE_PX`` x ``NODE_PX`` pixels (default 20);
    * ``npy:PATH``: a boolean (or 0/1) array of the render shape;
    * ``tif:PATH``: non-zero pixels of an image of the render shape.
    """
    kind, _, rest = spec.partition(":")
    h, w = (int(v) for v in shape)
    if kind == "npz":
        parts = rest.split(":")
        path = parts[0]
        values = tuple(int(v) for v in parts[1].split(",")) if len(parts) > 1 and parts[1] else (1, 4)
        node_px = int(parts[2]) if len(parts) > 2 else 20
        source = np.load(path)["source"]
        px = np.repeat(np.repeat(source, node_px, 0), node_px, 1)
        mask = np.isin(px, values)
        out = np.zeros((h, w), bool)
        hh, ww = min(h, mask.shape[0]), min(w, mask.shape[1])
        out[:hh, :ww] = mask[:hh, :ww]
        return out
    if kind == "npy":
        mask = np.load(rest).astype(bool)
    elif kind == "tif":
        import tifffile

        mask = tifffile.imread(rest) != 0
    else:
        raise ValueError(f"unknown phase-mask spec {spec!r} (npz:, npy: or tif:)")
    if mask.shape != (h, w):
        raise ValueError(f"phase mask {mask.shape} != render {(h, w)}")
    return mask


# ----------------------------------------------------------------------------------------------- outputs
def _atomic(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.part{path.suffix}")


def write_tif(path: str | os.PathLike, array: np.ndarray, compression: Optional[str] = "zlib") -> Path:
    """Atomic TIFF write (float32 maps: zlib-compressed)."""
    import tifffile

    path = Path(path)
    tmp = _atomic(path)
    tifffile.imwrite(tmp, np.ascontiguousarray(array), compression=compression)
    os.replace(tmp, path)
    return path


def read_tif(path: str | os.PathLike) -> np.ndarray:
    import tifffile

    return tifffile.imread(str(path))


def write_png(path: str | os.PathLike, image: np.ndarray, level: int = 3) -> Path:
    """Atomic PNG write with OpenCV (grey or BGR uint8)."""
    import cv2

    path = Path(path)
    tmp = _atomic(path)
    if not cv2.imwrite(str(tmp), np.ascontiguousarray(image), [cv2.IMWRITE_PNG_COMPRESSION, int(level)]):
        raise OSError(f"could not write {path}")
    os.replace(tmp, path)
    return path


def write_json(path: str | os.PathLike, obj: Any) -> Path:
    path = Path(path)
    tmp = _atomic(path)
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=_json_default) + "\n")
    os.replace(tmp, path)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def sha256_file(path: str | os.PathLike, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_zarr_layers(path: str | os.PathLike, layers: np.ndarray, attrs: Optional[dict] = None,
                      chunks: tuple[int, int] = (128, 128)) -> Path:
    """Write a uint8 ``(L, H, W)`` stack as a zarr v2 group with array ``"0"`` (level 0 only), Blosc lz4.

    This layout is read by inksweep, by villa's ``infer`` and by the PHerc. 343 scripts (which open group ``"0"``).
    """
    import numcodecs
    import shutil
    import zarr

    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    layers = np.ascontiguousarray(layers, dtype=np.uint8)
    n, h, w = layers.shape
    chunk = (n, min(chunks[0], h), min(chunks[1], w))
    comp = numcodecs.Blosc(cname="lz4", clevel=3, shuffle=1)
    major = int(zarr.__version__.split(".")[0])
    if major >= 3:
        g = zarr.open_group(str(tmp), mode="w", zarr_format=2)
        arr = g.create_array("0", shape=layers.shape, chunks=chunk, dtype="uint8", fill_value=0,
                             compressors=comp, chunk_key_encoding={"name": "v2", "separator": "/"})
    else:
        g = zarr.open_group(str(tmp), mode="w")
        arr = g.create_dataset("0", shape=layers.shape, chunks=chunk, dtype="uint8", fill_value=0, compressor=comp,
                               dimension_separator="/")
    arr[:] = layers
    if attrs:
        g.attrs.update(json.loads(json.dumps(attrs, default=_json_default)))
    if path.exists():
        shutil.rmtree(path)
    os.rename(tmp, path)
    return path
