"""inksweep: GPU-resident ink-detection depth sweeps for Herculaneum scroll surface volumes.

One process loads the surface volume onto the GPU once and runs every reader at every depth, tile phase and
orientation, reproducing the reference recipes exactly (``mode="exact"``) or as fast as possible (``mode="fast"``).
See docs/readers.md for the readers, their authors and licences.
"""
__version__ = "0.1.0"

__all__ = ["__version__"]
