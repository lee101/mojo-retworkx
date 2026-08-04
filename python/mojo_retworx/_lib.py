"""ctypes access to the single Mojo graph-kernel shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-retworkx.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mrx_dijkstra": ([I] * 10, I),
    "mrx_floyd_warshall": ([I] * 5, None),
    "mrx_toposort": ([I] * 6, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    sources = [os.path.join(ROOT, "src", "capi.mojo")]
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
        return LIB
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], cwd=ROOT, text=True, capture_output=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _loaded


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def addr(values: np.ndarray) -> int:
    if not isinstance(values, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if values.dtype not in (np.dtype(np.int64), np.dtype(np.float64)):
        raise TypeError(f"FFI buffers must be int64 or float64, got {values.dtype}")
    if not values.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    return values.ctypes.data
