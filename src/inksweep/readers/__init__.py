"""Readers: the :class:`ReaderSpec` contract, the built-in adapters and plugin loading.

A plugin is a Python module (dotted name or ``.py`` file) that defines ``READER: inksweep.readers.ReaderSpec``; pass it
with ``inksweep run --plugin path/to/reader.py`` and its checkpoint with ``--checkpoint NAME=PATH``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable

from .spec import ReaderSpec, as_probability_batch

__all__ = ["ReaderSpec", "as_probability_batch", "builtin_readers", "get_reader", "load_plugin", "RECIPES"]

#: Reader lists of named recipes. ``pherc0343``: the four models of the PHerc. 343 First Letters image, in the order
#: of its ensemble (that order fixes the float32 summation of the z-scores).
RECIPES: dict[str, list[str]] = {
    "pherc0343": ["dnative", "hecate", "ink9_s42", "ink9_s43"],
    "best9um": ["readerv3", "d9v4c", "dnative", "hecate", "v8in", "ink9_s43_50-75k"],
}

_FACTORIES: list[Callable[[], dict[str, ReaderSpec]]] = []


def _factories() -> list[Callable[[], dict[str, ReaderSpec]]]:
    if not _FACTORIES:
        from . import canonical, hecate, v8in, villa

        _FACTORIES.extend([villa.builtin_readers, hecate.builtin_readers, v8in.builtin_readers,
                           canonical.builtin_readers])
    return _FACTORIES


def builtin_readers() -> dict[str, ReaderSpec]:
    """Every built-in reader by name (building a spec is cheap: no weights are loaded)."""
    out: dict[str, ReaderSpec] = {}
    for factory in _factories():
        out.update(factory())
    return out


def get_reader(name: str) -> ReaderSpec:
    readers = builtin_readers()
    if name not in readers:
        raise KeyError(f"unknown reader {name!r}; built-in: {', '.join(sorted(readers))}")
    return readers[name]


def load_plugin(target: str) -> ReaderSpec:
    """Import a plugin module (``pkg.module`` or ``/path/reader.py``) and return its ``READER``."""
    if target.endswith(".py") or Path(target).is_file():
        path = Path(target).resolve()
        name = f"inksweep_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    reader = getattr(module, "READER", None)
    if not isinstance(reader, ReaderSpec):
        raise TypeError(f"{target}: READER must be an inksweep.readers.ReaderSpec, got {type(reader).__name__}")
    return reader
