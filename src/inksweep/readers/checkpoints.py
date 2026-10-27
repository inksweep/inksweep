"""Pinned checkpoints and model code: URLs at fixed revisions, sha256, download and verification.

Every file is fetched anonymously over HTTPS from a pinned revision (a Hugging Face commit or a GitHub release) and
checked against its sha256 before use; a mismatch is an error, never a warning. The sha256 values are the Git LFS
object ids published by Hugging Face (or the release notes for GitHub assets), so they can be checked independently.
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .. import __version__

HF = "https://huggingface.co"
INK9_REV = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"            # scrollprize/ink_9um
HECATE_REV = "9cb86e500e944b11a06a7020403cde5dffb5bcb2"          # scrollprize/hecate
DENSE_REV = "b894a0493ab2291cc8dc3247020c0d7cb0967a58"           # Nieuwlaar/ink9um-dense-native
RV2_REV = "72633d8aa6d40e9e737ccea4c9928fbc624dab45"             # domenicor046/reader-v2
RV3_REV = "aee1d64bced4eb35546348f3d276b9e2a0b99c46"             # domenicor046/reader-v3
V8IN_REV = "d89166b41a3f5fad7749b3d7c0fdd1bd3695d844"            # YoussefMoNader/ink-8um-v8in
CANON_REV = "075855bc69317ef6febf39a0d9d687b27d2b7c29"           # scrollprize/ink_canonical_2um
D9V4C_RELEASE = "https://github.com/TAUIL-Abd-Elilah/9um-reader-letter-shape/releases/download/v1.0"


@dataclass(frozen=True)
class Checkpoint:
    """One pinned file."""

    key: str
    url: str
    sha256: str
    filename: str
    size: Optional[int] = None
    source: str = ""
    licence: str = "MIT"


def _hf(repo: str, rev: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{rev}/{path}"


_INK9_SHA = {
    (42, 10000): "5d0896899092f312299c988de5861a5b6d3669112ee2a4976e2ecf44d4fe3664",
    (42, 20000): "c72023e13ea466fe1e579ed012eb209b909d79b501b7e1839a27c58ee6f5c6e4",
    (42, 30000): "ade09944cd02f02fb9a25a135b6a9097366e47813abb18d91ddc4da919535d86",
    (42, 40000): "f1e2101df567a431d3f39d17503316483084bace64cba55ef9671e5ea24ea76a",
    (42, 50000): "c16c75db4c6ea75b929c4db82437098d0a4ca0f3779c77e3aabc228f52ea2238",
    (42, 60000): "63cad1fb689a575662b6cc367635ec968ec488b219af12aff0497781488070ab",
    (42, 75000): "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab",
    (43, 10000): "2c08d9e1f833f345cfb7ab2543069829d96e918a668d263099a0972dc5dceeeb",
    (43, 20000): "b89b2ebcf49d71712c6c38631eec2306966c1602e80aa6f32a1d6123b0bdf1c9",
    (43, 30000): "4d9b6487b7c6b4f2a1d5a7b57cba37c8b0ccee259c74d9b6b04008cc52597b4a",
    (43, 40000): "e0b9ec7e5e9cb7caf6c9689bd35159285bb82c629f63e5ec83cb220e8d659978",
    (43, 50000): "c3c55191dad99849696c980aa7abaf6305514a91d6b8ba898e1e0dbb50eb3ea4",
    (43, 60000): "bf229faf754da3f1fc3026a3f9a9649341aeb3feb7bc89099cad31c55525d270",
    (43, 75000): "2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f",
}
INK9_STEPS = (10000, 20000, 30000, 40000, 50000, 60000, 75000)


def _build() -> dict[str, Checkpoint]:
    reg: dict[str, Checkpoint] = {}

    def add(c: Checkpoint) -> None:
        reg[c.key] = c

    for (seed, step), sha in _INK9_SHA.items():
        path = f"hybrid_3d2d-seed{seed}/step-{step:06d}.pth"
        add(Checkpoint(f"ink9_s{seed}_{step // 1000:03d}k", _hf("scrollprize/ink_9um", INK9_REV, path), sha,
                       f"ink_9um/{path}", 138360039 if seed == 42 else 138360231,
                       f"huggingface.co/scrollprize/ink_9um@{INK9_REV[:8]}"))
    add(Checkpoint("dense_native_016000",
                   _hf("Nieuwlaar/ink9um-dense-native", DENSE_REV, "weights/dense_native_016000.safetensors"),
                   "d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d",
                   "dense_native/dense_native_016000.safetensors", 272767904,
                   f"huggingface.co/Nieuwlaar/ink9um-dense-native@{DENSE_REV[:8]}"))
    add(Checkpoint("dense_native_config",
                   _hf("Nieuwlaar/ink9um-dense-native", DENSE_REV, "training/configs/train_dense_native.json"),
                   "5d8459030b36cd3980a03610c684365a7407d497173eff433efdb37f75840c86",
                   "dense_native/train_dense_native.json", 11823,
                   f"huggingface.co/Nieuwlaar/ink9um-dense-native@{DENSE_REV[:8]}"))
    add(Checkpoint("soup42_last4",
                   _hf("Nieuwlaar/ink9um-dense-native", DENSE_REV, "weights/soup42_last4.safetensors"),
                   "ab3f1bdefa53733f4d4c46da6c7713c142bc8c703df5098fe8641757d764849c",
                   "dense_native/soup42_last4.safetensors", 272767904,
                   f"huggingface.co/Nieuwlaar/ink9um-dense-native@{DENSE_REV[:8]}"))
    add(Checkpoint("readerv2_040000", _hf("domenicor046/reader-v2", RV2_REV, "reader-v2-step040000.pth"),
                   "654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6",
                   "reader_v2/reader-v2-step040000.pth", 138382373,
                   f"huggingface.co/domenicor046/reader-v2@{RV2_REV[:8]}"))
    add(Checkpoint("readerv3", _hf("domenicor046/reader-v3", RV3_REV, "reader-v3.pth"),
                   "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a",
                   "reader_v3/reader-v3.pth", 138379659, f"huggingface.co/domenicor046/reader-v3@{RV3_REV[:8]}"))
    add(Checkpoint("d9v4c_006000", f"{D9V4C_RELEASE}/d9v4C_ft-006000.pth",
                   "39360873b5379d28e2ed61c6fdf477a2bef7247ab51b95d4aea6b542757f0ac0",
                   "d9v4c/d9v4C_ft-006000.pth", None, "github.com/TAUIL-Abd-Elilah/9um-reader-letter-shape v1.0"))
    add(Checkpoint("d9v4c_seed1_006000", f"{D9V4C_RELEASE}/d9v4C_seed1_ft-006000.pth",
                   "bde42b134714cb0ed6851744272216cc40ed77095e0c6939cf7ee1fbc0ac59d5",
                   "d9v4c/d9v4C_seed1_ft-006000.pth", None,
                   "github.com/TAUIL-Abd-Elilah/9um-reader-letter-shape v1.0"))
    add(Checkpoint("hecate_9.6um", _hf("scrollprize/hecate", HECATE_REV, "hecate_9.6um.pth"),
                   "809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d",
                   "hecate/hecate_9.6um.pth", 522576086, f"huggingface.co/scrollprize/hecate@{HECATE_REV[:8]}"))
    add(Checkpoint("hecate_2.4um", _hf("scrollprize/hecate", HECATE_REV, "hecate_2.4um.pth"),
                   "fa2de3827ebd247ff2d33df1b6f899ea79d5269036cf16765997c62f0e9988c7",
                   "hecate/hecate_2.4um.pth", 522656918, f"huggingface.co/scrollprize/hecate@{HECATE_REV[:8]}"))
    add(Checkpoint("hecate_code", _hf("scrollprize/hecate", HECATE_REV, "hecate.py"),
                   "c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59",
                   "hecate/hecate.py", 22462, f"huggingface.co/scrollprize/hecate@{HECATE_REV[:8]}"))
    for name, sha, size in (("model.safetensors", "3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd",
                             333532180),
                            ("config.json", "30f46a55d61dc3f4e9bf7e9ca261c1486575f1ce05a27f28d09f10ee57597018", 182),
                            ("ink8um/__init__.py", "7e1c7a9867614e3819a7556520f5f9769631f9fe4214f75d2710afc6479bae3b",
                             298),
                            ("ink8um/modeling.py", "0656550dc10556ad09f5b6c21b36c97aa97a5696f71ef4b384eada124095c109",
                             7978),
                            ("ink8um/inference.py", "bca4a2ba12e46f65df810b4f54d37a8b42ec9923e56f2a3c6acb3e3ffd984f38",
                             9002)):
        add(Checkpoint(f"v8in/{name}", _hf("YoussefMoNader/ink-8um-v8in", V8IN_REV, name), sha, f"v8in/{name}", size,
                       f"huggingface.co/YoussefMoNader/ink-8um-v8in@{V8IN_REV[:8]}"))
    add(Checkpoint("canonical_2um",
                   _hf("scrollprize/ink_canonical_2um", CANON_REV, "r152_3ddec_v2_l5_epoch13.ckpt"),
                   "36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0",
                   "canonical_2um/r152_3ddec_v2_l5_epoch13.ckpt", 1549556213,
                   f"huggingface.co/scrollprize/ink_canonical_2um@{CANON_REV[:8]}"))
    return reg


REGISTRY: dict[str, Checkpoint] = _build()
V8IN_FILES = tuple(k for k in REGISTRY if k.startswith("v8in/"))


def sha256_file(path: str | os.PathLike, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def local_path(key: str, cache_dir: str | os.PathLike) -> Path:
    return Path(cache_dir) / REGISTRY[key].filename


def _download(url: str, dest: Path, retries: int = 5, timeout: float = 60.0) -> None:
    """Resumable HTTPS GET into ``dest.part``, then rename. Anonymous; nothing is uploaded."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": f"inksweep/{__version__}"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(part, dest)
            return
        except Exception as exc:  # network errors: retry with back-off
            last = exc
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last}")


def fetch(key: str, cache_dir: str | os.PathLike, download: bool = True) -> Path:
    """Path of a verified local copy of ``key`` (downloading it if allowed). Raises on a sha256 mismatch."""
    if key not in REGISTRY:
        raise KeyError(f"unknown checkpoint {key!r}; known: {sorted(REGISTRY)}")
    c = REGISTRY[key]
    path = local_path(key, cache_dir)
    if path.is_file():
        got = sha256_file(path)
        if got == c.sha256:
            return path
        raise RuntimeError(f"{path}: sha256 {got} != pinned {c.sha256}; delete it to re-download")
    if not download:
        raise FileNotFoundError(f"{key}: {path} missing (run `inksweep fetch` or allow downloads)")
    _download(c.url, path)
    got = sha256_file(path)
    if got != c.sha256:
        bad = path.with_name(path.name + ".bad")
        os.replace(path, bad)
        raise RuntimeError(f"{key}: downloaded file has sha256 {got}, pinned {c.sha256} (kept as {bad})")
    return path


def fetch_all(keys: Iterable[str], cache_dir: str | os.PathLike, download: bool = True) -> dict[str, Path]:
    return {k: fetch(k, cache_dir, download) for k in keys}


def record(keys: Iterable[str]) -> list[dict]:
    """ink.json entries for the checkpoints used."""
    out = []
    for k in keys:
        c = REGISTRY[k]
        out.append(dict(key=k, sha256=c.sha256, url=c.url, source=c.source, licence=c.licence))
    return out
