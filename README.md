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
| Installed as | uv git dependency (`[tool.uv.sources]` in `pyproject.toml`), read-only upstream |

The `rev` in `pyproject.toml` is the single source of truth for the pin;
`uv.lock` records the resolved commit too. `describe_backend()` prints the
commit actually in use at the top of every script (read from the install
metadata; `+dirty` if a local checkout override has uncommitted changes).

## Setup

```bash
uv sync --extra cpu     # macOS / CPU
uv sync --extra cuda    # NVIDIA Linux (unagi)
```

No separate jaxborg clone is needed: uv fetches the pinned commit from GitHub.

Pick exactly one extra; plain `uv sync` can silently fall back to CPU on a GPU
host. Python is pinned to 3.11 to match jaxborg.

`pyproject.toml` pins every package to the version in jaxborg's own
`uv.lock`, because a dependency does not pass its lockfile on and a fresh
resolve drifts (see `NOTES.md`). To bump jaxborg, change `rev` in
`[tool.uv.sources]` to the new full SHA, then regenerate the pins (needs
network; the script fetches jaxborg's `uv.lock` at that rev):

```bash
uv run python scripts/pin_jaxborg_lock.py && uv lock
```

Also update the commit and date in the table above.

## Working with jaxborg's source

The installed package contains only `src/jaxborg`. Its `recipes/`, `scripts/`
(training, eval) and docs are not installed, and `jaxborg.recipe` looks for
`recipes/` relative to the package, so recipes cannot be loaded from a git
install. cc4lab never needs them to run; two cases do:

- **Reading jaxborg code or docs, or training with its recipes as-is.** Use a
  scratch clone at the pinned commit, outside this repo, and load the resulting
  checkpoints into cc4lab via `jaxborg.checkpoint` / `jaxborg.policies` (see
  `policies/loaders.py`; loading a checkpoint trained this way is untested).
  Record the commit and recipe name next to each checkpoint.

  ```bash
  git clone https://github.com/ITM-Kitware/jaxborg ~/src/jaxborg-ref
  git -C ~/src/jaxborg-ref checkout b85bc324851808708a597b2bba3a84994ffff7d9
  ```

- **Trying a local change to jaxborg** (jaxborg is read-only upstream; describe
  any change that turns out to be necessary in `NOTES.md`). Layer a local
  checkout over the pinned install without touching `pyproject.toml` or
  `uv.lock`:

  ```bash
  uv run --with-editable ../jaxborg python scripts/spike/branching.py
  ```

  `describe_backend()` then reports that checkout's HEAD, with `+dirty` if it
  has uncommitted changes.

If cc4lab ends up leaning on jaxborg's repo tree, moving to a git submodule
(`git submodule add` at the pinned SHA, `path = "./jaxborg"` in
`[tool.uv.sources]`, point `scripts/pin_jaxborg_lock.py` at the submodule, and
restore the `git -C` commit lookup in `describe_backend()`) is a one-commit change.

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
