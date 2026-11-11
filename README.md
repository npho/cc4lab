# cc4lab

Counterfactual responsibility attribution among CAGE Challenge 4 blue agents, built on jaxborg.

cc4lab computes exact Shapley values over the 5 blue defenders in CC4 by
branching jaxborg episodes mid-trajectory and replaying them under all 32
coalitions (agents outside the coalition follow a reference policy), with
common random numbers across coalitions.

## Pinned jaxborg

| | |
|---|---|
| Upstream | https://github.com/ITM-Kitware/jaxborg |
| Commit | `b85bc324851808708a597b2bba3a84994ffff7d9` (2026-05-15) |
| Location | `../jaxborg` (sibling checkout, editable path dependency, read-only) |

`describe_backend()` prints the commit actually in use (with `+dirty` if the
checkout was modified) at the top of every script.

## Setup

```bash
git clone https://github.com/ITM-Kitware/jaxborg ../jaxborg
git -C ../jaxborg checkout b85bc324851808708a597b2bba3a84994ffff7d9

uv sync --extra cpu     # macOS / CPU
uv sync --extra cuda    # NVIDIA Linux (unagi)
```

Pick exactly one extra; plain `uv sync` can silently fall back to CPU on a GPU
host. Python is pinned to 3.11 to match jaxborg.

`pyproject.toml` pins every package to the version in jaxborg's own
`uv.lock`, because a path dependency does not pass its lockfile on and a
fresh resolve drifts (see `NOTES.md`). After bumping the jaxborg commit:

```bash
uv run python scripts/pin_jaxborg_lock.py && uv lock
```

## Spikes (Milestone 1)

```bash
uv run python scripts/spike/benchmark.py      # throughput vs batch size
uv run python scripts/spike/branching.py      # branch determinism + action timers
uv run python scripts/spike/variance.py       # seed count K, CRN vs independent
uv run python scripts/spike/reward_check.py   # team vs per-agent reward, per-zone terms
```

Each writes to `results/spike/` (gitignored). Findings are logged in `NOTES.md`.

## GPU runs on unagi

- Install with `--extra cuda` from the same lockfile; never copy the venv.
- Confirm `describe_backend()` lists a GPU device.
- GPUs are shared: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or cap with
  `XLA_PYTHON_CLIENT_MEM_FRACTION`) and retune batch size for 12 GB.
- Rerun `branching.py` on GPU; CPU determinism results do not carry over.
