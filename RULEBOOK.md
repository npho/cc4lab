# RULEBOOK: specification of the reference policy

Status: **specification, not yet implemented.** Written 2026-11-23 for a contributor who has not
seen the design discussion. Everything here was checked against jaxborg `b85bc32` (the pinned
commit) by reading its source; items I could not check are marked **[verify]**.

## 1. Why this exists

In cc4lab, every agent outside a coalition follows the **reference policy**
(`rollout.make_coalition_returns_fn`). Every Shapley value is therefore a value *relative to the
reference*: "agent i versus the rulebook". Today the reference is a one-rule firewall agent that
scores the same as always-Sleep (-6262 vs -6260, n = 32; `NOTES.md`, "First trained policy and
reference comparison"), so every attribution currently means "versus doing nothing" and cannot
support a claim of breach or negligence.

This file specifies a reference policy that is (a) grounded in published incident-response
practice rather than ad hoc, (b) simple enough to explain in a paper, and (c) JAX-traceable.
Implementing it closes these items in `HANDOFF.md`:

| HANDOFF item | Resolved here |
|---|---|
| Open decision 2: reference policy needs more rules | §4, §5 |
| Open decision 3: Sleep vs Monitor as the no-op; `cost_mode` default | §5.4, §6.3 |
| Suggested next step 4: extend the rulebook and re-check the reference comparison | §6 |
| Planted-negligence hooks (justified vs unjustified action) | §7 |

Not resolved here (the team's call): which failures and windows to analyse, the ex-ante/ex-post
reading, longer training, GPU runs.

**Non-goals.** No learned component and no LLM in the reference. No tuning of rule parameters to make
attribution results look a certain way (§6.2). Not a strong defender: it only has to be an explainable
standard.

## 2. Grounding in prior work

Primary citation: **NIST SP 800-61 Rev. 3**, *Incident Response Recommendations and Considerations
for Cybersecurity Risk Management: A CSF 2.0 Community Profile* (April 2025),
<https://csrc.nist.gov/pubs/sp/800/61/r3/final>. It replaced Rev. 2 (2012, *Computer Security
Incident Handling Guide*), whose four phases were preparation, detection and analysis, containment /
eradication / recovery, and post-incident activity. Rev. 3 reorganises incident response around the
six functions of the NIST Cybersecurity Framework 2.0 (Govern, Identify, Protect, Detect, Respond,
Recover). One citation therefore covers both vocabularies. Secondary: NIST CSF 2.0 (CSWP 29, 2024).

What we take from it is an **ordering and a set of roles**, not a specification of actions. Neither
document says "remove a host that shows a malicious process"; that translation into CC4 actions is
ours. In the paper, say the rulebook is *informed by* NIST SP 800-61 Rev. 3, never *compliant with* it.
Function-level mapping only; add category/subcategory IDs after reading the Rev. 3 tables (**[verify]**,
I did not check them).

| NIST function / role | CC4 mechanism | Rule |
|---|---|---|
| Protect: segmentation, enforce the approved baseline | `BlockTrafficZone` / `AllowTrafficZone` against the phase's communication policy | R1 Firewall (exists) |
| Detect: continuous monitoring | Monitor runs **automatically every step** in jaxborg and fills the observation's per-host flags (§3) | no action needed; idle choice in §5.4 |
| Respond, analyse | jaxborg's `Analyse` is a **no-op** in the flat observation (§3) | **omitted**, see §3 |
| Respond, contain | Block inbound links to a subnet with a detected compromise | R3 Contain (optional) |
| Respond, eradicate | `Remove` on a host with a detected malicious process | R2 Remove |
| Recover: rebuild when eradication cannot be trusted | `Restore` | R4 Restore (optional) |

## 3. What the simulator exposes (read, not assumed)

Paths are relative to jaxborg `src/jaxborg/`.

- **Monitor is automatic.** `env.py:132` calls `apply_blue_monitor` for every blue agent at the end of
  every step. The explicit `Monitor` action (`actions/__init__.py`, branch 1) runs the same function a
  second time, which ingests and ages events again. Whether that changes outcomes is **[verify]**: do not
  assume Monitor is what makes detection work. It is an ablation (§6.3).
- **Observation flags.** Per observed subnet, `observations.py` emits a block of
  `[one-hot(9), blocked(9), comms(9), malicious_processes(H), network_connections(H)]`, `H = 16`
  (`OBS_VECTOR_HOSTS_PER_SUBNET`; 1 + 3 x (27 + 32) + 32 message bits = 210 matches `OBS_DIM`).
  `malicious_processes[k]` is `host_exploit_detected | old_host_exploit_detected` for host slot k: a
  detection of a process-creation event, with two-cycle persistence. It also fires on green false
  positives (`blue_monitor.py` comment), so **R2 will sometimes act on a benign host**.
- **Host slot = action slot.** The `k`-th host flag of the `r`-th observed subnet corresponds to action
  `BLUE_{REMOVE,RESTORE}_START + r * H + k` (`encoding.py::_slot_to_global_host`,
  `masking.py`). Derive the offsets from the constants in `actions/action_defs.py`; do not hard-code
  them. (With current constants Remove starts at 50 and Restore at 98, but treat that as a check.)
- **Analyse does nothing.** `actions/blue_analyse.py` returns the state unchanged: the flat observation
  never receives Analyse's output. Emitting it would only make the agent busy for 2 ticks. The NIST
  "analysis" step is realised by reading the flags, so there is no Analyse rule.
- **Remove cannot clear privileged sessions.** `actions/blue_remove.py` removes only PIDs that Monitor
  recorded (`blue_suspicious_pids`) and skips sessions whose PID is privileged
  (root/SYSTEM, after red's privilege escalation). For such a host the flag persists after Remove.
  This is the in-simulator reason Restore exists, and the reason a stateless Remove rule can loop (§5.2).
- **Restore** wipes the host (`actions/blue_restore.py`): sessions, malware, decoys and services reset.
  It takes 5 ticks and costs -1 **per submitted action, including while the agent is busy**
  (`rewards.py:59-69`), so a naive repeat-while-flagged Restore pays up to -5 for one restore under
  `cost_mode=raw`.
- **Durations (ticks):** Sleep 1, Monitor 1, Analyse 2, Remove 3, Restore 5, Decoy 2, Block 1, Allow 1.
  An action submitted while busy is ignored; the **mask is static and not busy-aware**
  (`masking.py` docstring).
- **The policy interface is stateless:** `policy(keys (N_BLUE,2), obs (N_BLUE,OBS_DIM), masks
  (N_BLUE,ACTION_DIM)) -> actions (N_BLUE,)` (`policies/trivial.py`). The policy cannot see whether an
  agent is busy or what it did last step. Per-agent `keys` exist so swapping one agent's policy leaves
  the others' random streams untouched; **using them is allowed** (the old rulebook ignored them).

## 4. The design in one paragraph

A per-agent priority list, evaluated each step from that agent's own observation. **R1 Firewall**
enforces the *effective* communication policy (the phase's policy, plus links that Containment has
temporarily closed). **R2 Remove** picks, uniformly at random, one flagged host it can act on and
removes it. **R3 Contain** and **R4 Restore** are optional and off in the primary configuration.
When no rule fires the agent takes the **idle** action. Everything is a pure function of
`(key, obs, mask)`.

## 5. Specification

### 5.1 Configuration

```python
@dataclass(frozen=True)
class RulebookConfig:
    firewall: bool = True
    remove: bool = False
    contain: bool = False
    contain_sources: tuple[int, ...] = (INTERNET, CONTRACTOR_NETWORK)  # absolute jaxborg subnet ids
    restore_prob: float = 0.0          # 0 disables R4
    idle: str = "sleep"                # "sleep" | "monitor"
    priority: str = "protect_first"    # "protect_first" | "respond_first" (ablation only)

def make_rulebook(cfg: RulebookConfig) -> Policy: ...
```

`firewall_rulebook` and `make_firewall_rulebook(default_action)` stay importable and must return
**identical actions** to `make_rulebook(RulebookConfig(idle=...))` (regression test, §8).
Shape is the contributor's call; the content above is what must be expressible.

### 5.2 Rules

**R1 Firewall** (exists; `firewall_candidates`). Trigger: some real (observed-slot, non-diagonal) cell
has `blocked != effective_comms` and the fixing action is valid under `masks`. Action: Block if
`effective_comms` says blocked, else Allow; first cell in (slot, column) order. This is deterministic
and **converges** (each fix changes one cell and takes 1 tick), so it cannot starve other rules for
long: the existing tests reach compliance within 25 steps and re-adapt after a phase change.
`effective_comms = comms | contained`, where `contained[dst, src]` is true iff `cfg.contain` and the
observed subnet `dst` has at least one flagged host and `src` is in `cfg.contain_sources`.
Implementing containment as an override of the *target* (not as a separate block action) is deliberate:
a separate action would be undone by R1 on the next step (oscillation), and with the override the links
reopen on their own once the flags clear.

**R2 Remove** (NIST: eradicate). Trigger: at least one host slot with `malicious_processes == 1` where
`masks[REMOVE_START + r*H + k] > 0`. Action: `Remove` on one such slot **chosen uniformly at random
with the agent's key** (e.g. `jax.random.categorical` over `where(flagged, 0, -inf)`).
Why random, not first-in-order: a host owned by a privileged red session keeps its flag, and a
first-in-order rule would Remove it forever and starve every flagged host behind it.

**R4 Restore** (NIST: recover; optional). Trigger: same flagged-and-valid set as R2, and a Bernoulli
draw with probability `cfg.restore_prob` succeeds (second key split). Action: `Restore` on a slot
chosen as in R2; this *replaces* R2's Remove for that step. Rationale: a stateless policy cannot know
that Remove already failed on a host. Escalating with a fixed probability gives a geometric number of
Remove attempts before a Restore, so hosts whose flag clears after one Remove are rarely restored and
hosts stuck behind privileged sessions eventually are. **Not usable under `cost_mode=raw`** (§3, repeat
charges while busy); require `cost_mode=busy_corrected` when `restore_prob > 0`, and make
`make_rulebook` or the scripts refuse the combination.

**R3 Contain** (NIST: contain; optional). Realised only through `effective_comms` in R1. Default
`contain_sources = {INTERNET, CONTRACTOR_NETWORK}`: the external-facing subnets, my proposal, **[verify]**
against red's entry points in `jaxborg/scenarios`. Total isolation (all sources) is a stress setting.
Expect Contain to be net-**negative** under this reward: blocking an open link that the policy wants
open cuts green users off (ASF). That is the point of measuring it (§7).

### 5.3 Priority

`protect_first` (default): R1, then R2/R4, then idle. R1 is first because it converges, so it cannot
starve incident response for long, while R2 can loop on a stuck host. `respond_first` exists only as an
ablation and can starve R1; report it as such.

### 5.4 Idle action

`sleep` by default. `monitor` is the ablation (§6.3). Each agent submits exactly one action per step, so
the rulebook returns one action per agent.

## 6. Experiment protocol

### 6.1 The ladder

Measure each configuration with `scripts/eval_checkpoint.py` (extend it, §9) on the same episodes as
the existing baselines:

| Rung | Config | Notes |
|---|---|---|
| L0 | `always_sleep` | exists, -6260 |
| L1 | firewall only | exists, -6262 |
| L2 | L1 + `remove=True` | candidate primary reference |
| L3 | L2 + `idle="monitor"` | isolates the explicit-Monitor effect |
| L4 | L2 + `contain=True` | expected harmful; report it |
| L5 | L2 + `restore_prob=0.25`, `cost_mode=busy_corrected` | strongest, sensitivity analysis |

`restore_prob = 0.25` is a starting value, not a tuned one. Also run `respond_first` once on L2.

### 6.2 Seeds: decide on one set, report on another

The choices below are data-driven, so make them on **selection seeds 1000-1031** and report on the
project's evaluation seeds **0-31**. `eval_checkpoint.py` currently uses seeds from 0; add a
`--seed-start` argument. Do not tune anything else (rule triggers, `contain_sources`, `restore_prob`)
on any seeds; if you change them, say so in `NOTES.md` and re-run the whole ladder.

### 6.3 Decision rules (so the choices are not ad hoc)

Use paired differences per episode (same seeds, same topologies), mean +/- SE.
1. **Idle action:** Sleep unless Monitor beats it by more than 2 paired SE on the selection seeds.
2. **Primary reference:** L2 (with the idle action from rule 1). Promote L5 only if the team wants a
   stronger standard; either way run attribution under L1, the primary, and L5 and report all three, so
   the paper shows how much the attribution depends on the reference.
3. **`cost_mode`:** `raw` for every configuration without Restore (the rulebook never emits Restore when
   `restore_prob = 0`, so it makes no difference there); `busy_corrected` whenever `restore_prob > 0`.
4. If the primary reference **beats the trained policies** (-2091 to -2683), keep it and say so:
   attribution then means "versus a competent baseline" and negative values become expected. Do not
   weaken the rulebook to avoid this.

## 7. Violation flags and planted-negligence hooks

Extend `rules.py` (all jit/vmap friendly, busy-exempt like `firewall_violations`) with:

- `incident_unaddressed`: some host slot is flagged and valid, the agent is not busy, and its chosen
  action is neither Remove nor Restore on a flagged host. This is "ignored a detected compromise".
- `restore_unjustified`: the agent submitted Restore on a host slot whose flag is **not** set.
- `contain_justified` (optional): a Block that `traffic_action_kind` calls `violating` while the
  subnet has a flagged host.

These make the project's justified/negligent pairs observable from `obs` alone: a Restore on an
Operational Zone A host in phase 2A is justified when its flag is set and unjustified when it is not.
The flags use only observable state, so they can be computed for trained agents too, which is what
lets you ask whether a trained agent's negative Shapley value coincides with rule violations.

## 8. Tests (follow `tests/test_rulebook.py`)

Use jaxborg's own `encode_blue_action` as the independent oracle, as the existing tests do, and make
each test fail under an obvious mutation (swap Remove/Restore, drop the mask check, drop the flag).

1. R2 picks the flagged slot; respects an invalid mask (falls through to idle); no flag means no Remove.
2. R2 fairness: with 3 flagged hosts, every one is selected over a few hundred keys.
3. Containment: with a flagged host, `effective_comms` blocks `contain_sources`, R1 never emits an
   Allow for a contained link, and it re-allows once the flag clears. Run a short rollout to confirm
   there is no Block/Allow oscillation.
4. R4: `restore_prob=0` never emits Restore; `restore_prob=1` with a flag always does; never without a flag.
5. `idle` switch returns Monitor or Sleep when nothing fires.
6. Regression: `firewall_rulebook` equals `make_rulebook(firewall only)` on random observations.
7. The new flags against hand-built cases, including the busy exemption.
8. Integration: a ladder rung inside `make_coalition_returns_fn` still satisfies Shapley efficiency, and
   the full/empty coalitions still equal the plain single-policy rollouts exactly.

## 9. Work breakdown

0. **Reconcile the existing work in progress first.** On 2026-11-23 the main checkout had uncommitted
   edits (`rulebook.py`: `firewall_cells` refactor; `rules.py`: `traffic_action_kind`, `TRAFFIC_KINDS`;
   `tests/test_rulebook.py`; `scripts/action_type_swap.py`, `scripts/block_compliance.py`). `rules.py`
   there contains duplicated imports, two definitions of `traffic_action_kind` and of `TRAFFIC_KINDS`, and
   the second definition unpacks `firewall_cells` in the wrong order. Keep one, fix the order, run the
   tests, commit; this spec builds on `firewall_cells`.
1. `policies/rulebook.py`: `RulebookConfig`, `make_rulebook`; keep the old names as aliases.
2. `rules.py`: the §7 flags.
3. Tests, §8.
4. `scripts/eval_checkpoint.py`: add the ladder rungs as baselines and `--seed-start`.
   `scripts/attribute_branch.py`: let `--reference` take a rung name or a config.
5. Run the ladder (§6) on both seed sets; log the table, the decisions and their evidence in
   `NOTES.md`; update the README, and mark the reference-policy items in `HANDOFF.md` (a local, untracked file; never
commit it) as resolved.

## 10. Done when

- `uv run pytest` passes, including §8.
- The ladder table (mean +/- SE on seeds 0-31, plus paired differences) is in `NOTES.md`, with the idle
  and primary-reference decisions and the selection-seed evidence.
- `attribute_branch.py --reference <rung>` runs end to end for L1, the primary and L5, and efficiency holds.
- One short paragraph in `NOTES.md` states the reference in the paper's terms: what each rule is, what
  NIST role it plays, and that it is informed by, not compliant with, NIST SP 800-61 Rev. 3.

## 11. Known unknowns

- Whether explicit `Monitor` differs from `Sleep` at all, given automatic Monitor (§6.3 measures it).
- False-positive removes on benign hosts cost 3 busy ticks; the size of that effect is unmeasured.
- Whether `INTERNET` / `CONTRACTOR_NETWORK` are the right `contain_sources` **[verify]**.
- Outcomes are one topology family and one set of trained policies; the reference-sensitivity table
  (L1 / primary / L5) is the guard against over-reading any single reference.
