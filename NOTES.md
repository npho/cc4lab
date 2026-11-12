# NOTES

Findings log. Every **[unverified]** claim from `HANDOFF.md` is resolved here.
jaxborg paths are relative to the root of the jaxborg repo at commit `b85bc32`.

## Milestone 0: scaffold

### Install

- jaxborg extras are `cpu` (`jax[cpu]>=0.7.1`) and `cuda` (`jax[cuda12]>=0.7.1`).
  cc4lab mirrors them as `cpu`/`cuda` and declares them conflicting.
  **Confirmed** (`pyproject.toml` `[project.optional-dependencies]`).
- jaxborg pins Python 3.11.13 (`.python-version`). On 3.14, `ray==2.38.0`
  (via cyborg) has no wheel, so cc4lab pins `==3.11.*` too.
- **A dependency does not pass on its lockfile.** (Found with a path dependency;
  the same holds for a git dependency.) A fresh `uv lock` in
  cc4lab resolved different versions from jaxborg's `uv.lock`:
  - `jaxmarl` (declared as `branch = "main"`) moved from `63226c5` to `d4b185d`.
    The newer `MultiAgentEnv.__init__` sets `self.agents = ["agent_0", ...]`
    after `FsmRedCC4Env` set it, so `FsmRedCC4Env.reset` raised
    `KeyError: 'agent_0'`.
  - `mujoco-mjx` 3.1.3 → 4.0.1, which made `import jaxmarl` fail on macOS
    (`libmujoco.3.1.3.dylib` not loaded).
  - Also `flax` 0.10.4 → 0.12.8, `optax`, `chex`, `jax` 0.10.0 → 0.10.2.
  - Fix: `scripts/pin_jaxborg_lock.py` copies every locked version from
    jaxborg's `uv.lock` (fetched at the pinned rev) into `[tool.uv] constraint-dependencies`. Git URLs
    go to `override-dependencies`. jaxborg's own overrides are repeated by
    hand, because uv honours overrides only in the root project.
    jaxborg's `[tool.uv.sources]` (cyborg, jaxmarl git) *are* honoured
    through the path dependency. After the move to a git source, `uv lock`
    changed no package versions; the root overrides carry those URLs anyway,
    so this does not show whether uv still reads jaxborg's own sources.
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
- **Dependency layout: git source, not a sibling path.** jaxborg is a uv git
  dependency pinned by `rev` in `pyproject.toml` (previously an editable
  `../jaxborg` path). Reasons: the pin lives in code and `uv.lock` instead of
  README prose, and no second clone is needed on a fresh machine or on unagi.
  Trade-off: only `src/jaxborg` is installed, so jaxborg's `recipes/` and
  `scripts/` need a separate clone (see README, "Working with jaxborg's
  source"). Switching to a submodule later is a one-commit change.

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

## Milestone 1: feasibility spike

All runs: macOS, CPU backend (`cpu:0`), jax 0.10.0, jaxborg `b85bc32`,
blue policy `random_valid`. Raw output is in `results/spike/`, which is gitignored;
rerun with `uv run python scripts/spike/<name>.py`.

### 1. Throughput (`benchmark.py`)

500-step rollouts. steps/s excludes compile and reset.

| engine | batch | compile (s) | run (s) | steps/s |
|---|---|---|---|---|
| jaxborg, no vmap | 1 | 14.5 | 0.76 | **657** |
| jaxborg, vmap | 1 | 334.9 | 5.82 | 86 |
| jaxborg, vmap | 32 | 42.1 | 26.10 | 613 |
| jaxborg, vmap | 1024 | 49.7 | 1815.96 | 282 |
| CybORG `EnterpriseMAE`, 1 process | 1 | (setup 0.09) | 6.74 | 74 |

- **On CPU, vmap hurts.** Under `vmap`, `lax.cond` becomes `select`, so every
  branch of jaxborg's action and red-FSM dispatch runs for every env. One
  episode without vmap runs ~8× faster than vmapped batch 1. Batch 1024
  gets *less* total throughput than batch 32. Compile time also blows up
  (335 s vmapped vs 14.5 s unbatched). The no-vmap row is in
  `benchmark_novmap.json`.
- On CPU, the efficient layout is `lax.map` (sequential, real branches)
  inside each process, plus process-level parallelism. GPU is untested here.
  There, vmap should still win (jaxborg's README reports 2,512 steps/s
  at 1024 envs on an A6000), and unagi must measure it.
- **[unverified] "CybORG ~7 steps/s per process": corrected.** On this
  machine, one CybORG process with random-valid actions runs **74 steps/s**.
  The ~7 figure was 332/48 from jaxborg's PPO training runs, which include
  learner overhead and 48-way contention. jaxborg without vmap is still ~9×
  faster per process. The handoff's case for jaxborg holds (branching
  matters more than speed), but the speed gap on CPU is ~9×, not ~100×.

### 2. Branch determinism (`branching.py`): **PASS**

At t = 100 and t = 250, all of these hold bitwise (structure, dtype, shape, bytes):
- the prefix equals the first t steps of the full run;
- two continuations from a copied state with the same key give identical
  actions, rewards, and final `EnvState`;
- the branched continuation **replays the unbranched run** exactly,
  including the final state. The key schedule depends only on
  `(episode_key, state.time)`, so the copy loses nothing;
- a different key changes actions and rewards. The first divergence is at
  the branch step.

In-progress timers live in `SimulatorState` (`src/jaxborg/state.py`):
`blue_pending_ticks`, `blue_pending_action` (per blue agent), and
`red_pending_ticks`, `red_pending_action`, `red_pending_key` (per red agent).
The logic is in `actions/duration.py` `process_blue_with_duration`. The copy
(`jax.tree.map(jnp.copy, state)`) carries them. Timer probe at t = 100:
`blue_pending_ticks = [1,1,1,0,4]`. Changing a busy agent's submitted
action (Sleep vs BlockTraffic) leaves the next state bitwise identical.
Changing the idle agent's action changes `blocked_zones` and
`blue_pending_action`.

### 3. Seed count K (`variance.py`)

Per-episode σ of total return (100 full episodes):
- independent topologies: σ = **966** (mean −5105);
- one fixed topology, different episode keys: σ = **630** (mean −4298).

σ of D = V(all actual) − V(agent 0 → always-Sleep), from t = 250 to 500,
256 keys at each of 4 branch states (distinct topologies and prefixes):

| base | mean D | σ_D CRN | σ_D indep | corr(V_all, V_swap) CRN |
|---|---|---|---|---|
| 0 | 485.7 | 155.9 | 713.0 | 0.944 |
| 1 | 79.6 | 237.8 | 443.6 | 0.735 |
| 2 | 191.4 | 222.7 | 426.0 | 0.723 |
| 3 | 71.9 | 167.9 | 499.6 | 0.894 |
| pooled | | **199.2** | **533.0** | |

CRN cuts variance by **7.2×**. Recommended K for 2·σ_D/√K ≤ 50:

| | pooled σ | worst base |
|---|---|---|
| CRN | **K = 64** | K = 91 |
| independent | K = 455 | K = 814 |

Caveats: the "actual" policy is random-valid, not trained. A trained policy
is less noisy per action but may make agent-0 effects larger, so K must be
rechecked once a real policy exists. Here agent 0 → Sleep *raises* return
in every base, because random blue Restores cost −1 each.

Cost at K = 64: one branch point is 32 coalitions × 64 × 250 steps =
512k steps, about 13 min on one CPU core at 657 steps/s.

### 4. Reward structure (`reward_check.py`): **PASS**

- **One shared team reward.** All 5 blue agents get the same scalar every
  step (spread 0.0). `env.py:491-497` assigns one value to every blue
  agent. `fsm_red_env.py`'s post-step update touches only FSM fields.
- **Per-zone components are recoverable** from state, with no jaxborg changes.
  `const.phase_rewards[mission_phase]` gives the (subnet, [LWF, ASF, RIA])
  weights, and `green_lwf_this_step`, `green_asf_this_step`,
  `red_impact_attempted` give the per-host flags. Grouping by
  `const.host_subnet` (masked by `host_active`) rebuilds `info`'s
  `reward_lwf/asf/ria` and the total with max error 0.0. This is the same
  formula as `rewards.py` `compute_reward_breakdown`.
- Action cost is per agent (−1 per submitted Restore), so it can be
  attributed directly. Episode example: [−73, −78, −79, −54, −92].
- Subnet ownership (`const.blue_agent_subnets`): blue_0 RESTRICTED_ZONE_A,
  blue_1 OPERATIONAL_ZONE_A, blue_2 RESTRICTED_ZONE_B, blue_3
  OPERATIONAL_ZONE_B, blue_4 ADMIN/OFFICE/PUBLIC_ACCESS. No one owns
  CONTRACTOR_NETWORK or INTERNET.

## Spike verdict

**Go**, with one design change for CPU.

- **Throughput:** jaxborg runs 657 steps/s per CPU process without vmap
  (613 at vmap 32, 282 at vmap 1024). CybORG runs 74. GPU is unmeasured.
- **Branching: PASS** at t = 100 and 250. Continuations are bitwise
  reproducible and replay the unbranched run. Timers are carried in
  `SimulatorState.{blue,red}_pending_*`.
- **Recommended K = 64 under CRN** (91 for the worst branch state; 455
  without CRN). Use CRN.
- **Reward:** one shared team scalar. Per-subnet LWF/ASF/RIA and per-agent
  Restore cost can be rebuilt exactly from state.
- **[unverified] claims:**
  - obs dim 210, action dim 242, agents 0–3 zero-padded with masking: **confirmed**.
  - one shared actor-critic for all 5 agents: **confirmed** as the
    default arch (others exist).
  - `uv sync --extra cpu`: **confirmed**. `uv sync --extra cuda`:
    **corrected**, because jaxborg's override drops the CUDA plugin. cc4lab adds
    `jax-cuda12-plugin` itself; still unchecked on a GPU host.
  - CybORG ~7 steps/s per process: **corrected** to 74 measured
    (single process, random policy).
- **Change for Milestone 2:** the handoff's "32×K rollouts in one vmap call"
  is right for GPU but ~8× slow on CPU. `rollout.py` should take a
  `batch_mode` (`vmap` | `map`) so each backend can use its fast path.

## Training and checkpoint loading (after the git-source move)

- **Training runs on CPU with jaxborg's own trainer**, launched from cc4lab's
  environment: `ippo_jax.py` from a scratch clone plus an absolute recipe path
  (`recipe.py` resolves `recipes/` relative to the package, so name lookup fails
  under a git install). 12k steps, 8 envs: compile 73.7 s, then 357 and 342
  steps/s (learner included). Extrapolation, not measured: 500k steps about 25 min,
  3M about 2.5 h. 48 envs (recipe default) may differ in both compile and speed.
- **Checkpoints load into cc4lab from the installed package alone**
  (`policies/loaders.py`: `safetensors` file + `recipe_<tag>.yaml` sidecar). Tested
  with a unit test (matches a manual `apply` plus per-key `categorical`, respects masks,
  each agent uses only its own key) and with a real 12k-step checkpoint played
  for full episodes (`scripts/eval_checkpoint.py`). Shared arch only.
- The smoke checkpoint is near random-level (mean return -5069 vs -4508
  random_valid, n=4, noisy), as expected at 12k steps. No claim about trained-policy
  quality yet.
- **Sidecar caveat:** `write_sidecar` records `git rev-parse HEAD` of the launch
  directory, which is cc4lab's commit when training runs from here, not jaxborg's.
- The loader samples with one key per agent; the trainer uses one key per flattened batch.
  Same distribution, different random streams.

## Milestone 2 progress

- **Done: exact Shapley and leave-one-out** (`attribution/shapley.py`, `attribution/loo.py`),
  tested without any simulator (`tests/test_attribution.py`, 14 tests). Shapley is
  checked against closed forms (two-player asymmetric, additive, symmetric,
  3-player majority), the null-player and efficiency axioms, and an independent
  brute force over all 120 player orderings. Both take `values` of shape
  `(32, *batch)` and return `(5, *batch)`; they are jit-friendly.
- **Coalition index convention** (documented in `attribution/__init__.py`; the rollout
  code must use it): coalition `c` contains agent `i` iff bit `i` of `c` is set, so index 0 is
  the empty coalition (all reference) and 31 is the full coalition (all actual).
  `coalition_masks()` returns the `(32, 5)` bool masks in index order.
- Leave-one-out misses redundancy: if any 2 of 3 agents suffice, every agent
  scores 0, while Shapley gives each 1/3 (tested).
- **Done: rulebook reference agent and violation flags** (`policies/rulebook.py`,
  `rules.py`; `tests/test_rulebook.py`, 4 tests). First rule only: the firewall
  matches the mission phase's communication policy for each subnet the agent observes.
  - It is a pure function of `obs` and `masks`. The obs holds, per observed subnet,
    a one-hot, a `blocked` row and a `comms` row in CybORG subnet order
    (`observations.py`). `comms_policy` is "True = not connected" (`state.py:43`);
    `blocked_zones[dst, src]` is what Block/AllowTraffic write (`actions/blue_traffic.py`).
    A mismatch is fixed with BlockTrafficZone (policy blocked, firewall open) or
    AllowTrafficZone (the reverse). Action indices follow `encode_blue_action`:
    the destination slot is the observation block slot, and the source offset skips the destination.
  - **Trap: the comms-policy diagonal is `True` for every subnet** (a subnet toward
    itself is "not connected"). The rule must exclude `src == dst`, or it would try to
    "fix" a nonexistent link forever.
  - It fixes one mismatch per step (first in slot, column order), only if the action is
    valid under the mask, else it submits `default_action` (Sleep; `make_firewall_rulebook`
    takes another, e.g. Monitor). Sleep vs Monitor is not decided; it does not change
    this rule's behaviour but may matter once more rules exist.
  - Starting states are far out of compliance (e.g. agent 1 has 7 mismatches in phase 0),
    and from a fresh episode the rulebook reaches full compliance within 25 steps and
    re-adapts after a phase change (tested). The first-action test uses jaxborg's own
    encoder as an independent oracle; a Block/Allow swap mutation makes it fail.
  - **The rulebook never emits Restore**, so the busy-agent action-cost caveat above
    does not apply to it: a busy agent re-submitting a Block/Allow is ignored and costs nothing.
  - `rules.firewall_violations(obs, actions, masks=None, busy=None)` returns
    `firewall_mismatch` (state: the obs shows a disagreement) and `firewall_unaddressed`
    (mismatch present and the chosen action fixes none). Observations do not say whether
    an agent is busy, so pass `busy = blue_pending_ticks > 0` to exempt busy agents.
    Any fixing action counts as addressing; it does not demand the rulebook's order.
- **Not started, deferred by Nam (2026-11-11):**
  1. Train a real blue policy (500k steps, ~25 min on this CPU) and re-measure K.
  2. Busy-tick action-cost semantics for coalition masking (see "Milestone 2 caveat
     on busy agents" above). Decide before `rollout.py`; proposed default is to count
     the raw submitted cost and keep it switchable.
  3. How branch points (which failures, which step) are chosen; ties into the
     ex-ante vs ex-post decision.
  Also remaining for Milestone 2: coalition-masked `rollout.py` with the
  `batch_mode` (`vmap` | `map`) switch.

