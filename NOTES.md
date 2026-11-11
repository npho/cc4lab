# NOTES

Findings log. Every **[unverified]** claim from `HANDOFF.md` is resolved here.
jaxborg paths are relative to `../jaxborg` at commit `b85bc32`.

## Milestone 0: scaffold

### Install

- jaxborg extras are `cpu` (`jax[cpu]>=0.7.1`) and `cuda` (`jax[cuda12]>=0.7.1`).
  cc4lab mirrors them as `cpu`/`cuda` and declares them conflicting.
  **Confirmed** (`pyproject.toml` `[project.optional-dependencies]`).
- jaxborg pins Python 3.11.13 (`.python-version`). On 3.14, `ray==2.38.0`
  (via cyborg) has no wheel, so cc4lab pins `==3.11.*` too.
- **A path dependency does not pass on its lockfile.** A fresh `uv lock` in
  cc4lab resolved different versions from jaxborg's `uv.lock`:
  - `jaxmarl` (declared as `branch = "main"`) moved from `63226c5` to `d4b185d`.
    The newer `MultiAgentEnv.__init__` sets `self.agents = ["agent_0", ...]`
    after `FsmRedCC4Env` set it, so `FsmRedCC4Env.reset` raised
    `KeyError: 'agent_0'`.
  - `mujoco-mjx` 3.1.3 → 4.0.1, which made `import jaxmarl` fail on macOS
    (`libmujoco.3.1.3.dylib` not loaded).
  - Also `flax` 0.10.4 → 0.12.8, `optax`, `chex`, `jax` 0.10.0 → 0.10.2.
  - Fix: `scripts/pin_jaxborg_lock.py` copies every locked version from
    `../jaxborg/uv.lock` into `[tool.uv] constraint-dependencies`. Git URLs
    go to `override-dependencies`. jaxborg's own overrides are repeated by
    hand, because uv honours overrides only in the root project.
    jaxborg's `[tool.uv.sources]` (cyborg, jaxmarl git) *are* honoured
    through the path dependency.
  - After the fix, `jax 0.10.0`, `flax 0.10.4`, `mujoco-mjx 3.1.3` and the rest
    match jaxborg's lock exactly.
- **Upstream bug: jaxborg's `cuda` extra installs CPU-only JAX.**
  jaxborg's `override-dependencies = ["jax>=0.7.1", ...]` replaces the
  whole `jax[cuda12]>=0.7.1` requirement, extras included. jaxborg's own
  `uv.lock` has no `jax-cuda12-plugin`, so `uv sync --extra cuda` in jaxborg
  gives `CpuDevice` on GPU hosts. Reproduced in a two-line probe project:
  adding the override drops `jax-cuda12-plugin` from the lock. cc4lab works
  around it by requiring `jax-cuda12-plugin[with-cuda]==0.10.0` (Linux only)
  in its own `cuda` extra. This is worth reporting upstream (Nam's call). The
  unagi `jax.devices()` check is still required.
- **[unverified] install via `uv sync --extra cpu` / `--extra cuda`: confirmed
  for `cpu`; `cuda` was corrected as above.** It was locked but not run on a
  GPU here.
- No jaxborg files were modified.

### Interfaces the code depends on

- Env: `jaxborg.parity.fsm_red_env.FsmRedCC4Env` (built through
  `jaxborg.evaluation.jax_env_factory.make_jax_env(CC4_STOCK)`). This is the
  blue-only env; red actions come from the internal FSM selector.
  `ScenarioEnv` (`src/jaxborg/env.py`) takes red actions as inputs and is not
  used directly.
- cc4lab calls `step_env`, not `step`: `step` auto-resets on `done` and
  builds a fresh reset state every step, which we never need.
- **[unverified] obs dim 210, action dim 242, agents 0–3 zero-padded with
  masking: confirmed.** Obs = 1 (mission phase) + up to 3 subnet blocks × 59
  + 4 × 8 message floats (`src/jaxborg/observations.py`). Agents 0–3 fill one
  subnet block, then messages, then zeros. Action = 2 + 4 × 48 host-slot
  actions + 2 × 24 traffic slots = 242 (`actions/action_defs.py`). The
  mask is static per topology and ignores busy state (`actions/masking.py`).
- **[unverified] one shared actor-critic for all 5 agents: confirmed as the
  default** (`recipes/default.yaml` `arch.name: shared`;
  `policies/shared_actor_critic.py`). `separate` and `per_agent` archs also exist.
- **Messages are always zero.** `state.messages` is created in `state.py`
  and read in `observations.py`, but no action writes to it. The 32 message
  dims carry no information in jaxborg.
- **No trained checkpoint ships with jaxborg.** Spikes use `random_valid`.
- Action durations match the handoff: Sleep 1, Monitor 1, Analyse 2,
  Remove 3, Restore 5, Decoy 2, Block 1, Allow 1
  (`actions/encoding.py` `BLUE_ACTION_DURATIONS`).
- **Milestone 2 caveat on busy agents:** while an agent is busy, its
  submitted action is ignored (`actions/duration.py`
  `process_blue_with_duration`). But `compute_reward_breakdown` charges −1
  for every *submitted* Restore, even on busy ticks (`rewards.py:59-69`,
  intentional CybORG parity). Coalition masking changes which action is
  submitted, so it can change action cost on busy ticks with no effect on
  the env.
