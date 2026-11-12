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

- **Reading jaxborg code or docs, or training with its recipes as-is.** Keep a
  scratch clone at the pinned commit, outside this repo:

  ```bash
  git clone https://github.com/ITM-Kitware/jaxborg ~/src/jaxborg-ref
  git -C ~/src/jaxborg-ref checkout b85bc324851808708a597b2bba3a84994ffff7d9
  ```

  Train from cc4lab's environment with the clone's trainer script and an
  **absolute** recipe path (recipes are not found by name from a git install).
  Verified on CPU (macOS):

  ```bash
  JAXBORG_EXP_DIR=~/cc4lab-exp uv run python ~/src/jaxborg-ref/scripts/train/algorithms/ippo_jax.py \
      --recipe ~/src/jaxborg-ref/recipes/default.yaml --total-timesteps 500000 --num-envs 8 --tag default_seed42
  ```

  This writes `model_<tag>.safetensors` and `recipe_<tag>.yaml` under
  `$JAXBORG_EXP_DIR/ippo_jax/<tag>/`. MLflow also drops run artifacts in
  `./mlruns` of the launch directory (gitignored here), regardless of `JAXBORG_EXP_DIR`. Measured on CPU: ~74 s one-time compile,
  then ~350 steps/s at 8 envs, so 500k steps is roughly 25 min and the
  recipe's 3M about 2.5 h (48 envs, the recipe default, not measured). The
  recipe notes say returns plateau by ~500k steps.

  Load and check a checkpoint with `cc4lab.policies.loaders.load_policy`, or
  play it (several checkpoints at once are fine) against the baselines (random, sleep, firewall rulebook):

  ```bash
  uv run python scripts/eval_checkpoint.py ~/cc4lab-exp/ippo_jax/default_seed42/model_default_seed42.safetensors
  ```

  Only the `shared` architecture is supported by the loader. The sidecar's
  `git_commit` is the commit of the directory the trainer was launched from
  (not necessarily jaxborg), so record the jaxborg pin yourself next to results.

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

## Attribution (Milestone 2, in progress)

`cc4lab.attribution` computes exact Shapley values (`shapley_values`) and the
leave-one-out baseline (`loo_values`) from the 32 coalition values. Coalition
`c` contains agent `i` iff bit `i` of `c` is set (index 0 = all reference
policy, 31 = all actual); `coalition_masks()` gives the masks in that order.
Run the tests (no GPU needed) with:

```bash
uv run pytest
```

`cc4lab.policies.rulebook.firewall_rulebook` is the reference agent (first rule:
firewall matches the mission phase's communication policy) and
`cc4lab.rules.firewall_violations` flags agents that leave a mismatch unaddressed.
`cc4lab.rollout.make_coalition_returns_fn` evaluates all coalitions over shared
random keys (`batch_mode`: `map` for CPU, `vmap` for GPU; `cost_mode`: `raw` or
`busy_corrected`, see `NOTES.md`). End to end for one branched episode:

```bash
uv run python scripts/attribute_branch.py MODEL.safetensors --branch-t 250 --keys 4
```

It prints Shapley values and the leave-one-out baseline with standard errors and
checks efficiency; results are written to `results/attribution/` (gitignored).
`--reference` (`sleep` or `rulebook`) and `--cost-mode` expose the open design choices.
See `NOTES.md`, "Coalition rollouts and the first end-to-end attribution".

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
