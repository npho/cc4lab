"""Shared helpers for Milestone 1 spike scripts."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results" / "spike"


def write_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, default=_default))
    print(f"wrote {path}")
    return path


def _default(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (np.ndarray, jax.Array)):
        return np.asarray(o).tolist()
    raise TypeError(type(o))


def bitwise_equal(a, b) -> bool:
    """True iff two pytrees have identical structure, dtypes, shapes and bytes."""
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    if ta != tb:
        return False
    for x, y in zip(la, lb):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False
    return True


def differing_leaves(a, b) -> list[str]:
    """Paths of leaves whose bytes differ (for diagnostics)."""
    out = []
    for (path, x), (_, y) in zip(jax.tree_util.tree_leaves_with_path(a), jax.tree_util.tree_leaves_with_path(b)):
        if np.asarray(x).tobytes() != np.asarray(y).tobytes():
            out.append(jax.tree_util.keystr(path))
    return out
