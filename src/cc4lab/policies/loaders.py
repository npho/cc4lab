"""Load trained jaxborg policy checkpoints as cc4lab policies.

jaxborg's trainer (``scripts/train/algorithms/ippo_jax.py``, run from a
scratch clone; see README) writes ``model_<tag>.safetensors`` plus a
``recipe_<tag>.yaml`` sidecar. ``load_policy`` needs only those two files and the
installed jaxborg package, not jaxborg's repo tree.

Only the ``shared`` architecture (jaxborg's default recipe) is supported.
``separate`` and ``per_agent`` take different inputs and are not wired up.

Sampling differs from the trainer in one deliberate way: the trainer draws one
key for the whole flattened batch, while cc4lab policies take one key per agent
(see ``trivial.py``) so that swapping one agent's policy leaves the others'
random streams untouched. The action distribution is identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxborg.checkpoint import load_jax_params, read_sidecar
from jaxborg.policies import make_jax_policy

Policy = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def checkpoint_info(model_path: str | Path) -> dict[str, Any]:
    """Provenance of a checkpoint, from its recipe sidecar, to record next to results."""
    side = read_sidecar(model_path)
    return {
        "recipe": side.get("meta", {}).get("name"),
        "arch": side["arch"],
        "algorithm": side.get("algorithm"),
        **side.get("run", {}),
    }


def load_policy(model_path: str | Path) -> Policy:
    """Return ``policy(keys (N_BLUE, 2), obs (N_BLUE, OBS_DIM), masks (N_BLUE, ACTION_DIM)) -> (N_BLUE,) int32``."""
    arch = read_sidecar(model_path)["arch"]
    if arch["name"] != "shared":
        raise NotImplementedError(f"arch {arch['name']!r} is not supported; only 'shared' is")
    params, action_dim = load_jax_params(model_path)
    network = make_jax_policy(
        arch["name"],
        action_dim=action_dim,
        hidden_dim=arch["hidden_dim"],
        hidden_layers=arch["hidden_layers"],
        activation=arch["activation"],
    )

    def policy(keys: jax.Array, obs: jax.Array, masks: jax.Array) -> jax.Array:
        pi, _ = network.apply(params, obs, masks.astype(jnp.float32))
        return jax.vmap(jax.random.categorical)(keys, pi.logits).astype(jnp.int32)

    return policy
