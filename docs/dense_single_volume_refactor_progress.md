# Dense Single-Volume EM Refactor Progress

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)  
Current phase: C1 — introduce data contracts behind compatibility APIs
Last updated: 2026-09-08

## Status board

| Component | Status | Current result / next action |
|---|---|---|
| C0 Baseline and guardrails | IN PROGRESS | Inventory and existing same-HEAD K=1 artifact recorded. Next: make a fresh paired, hardware-attributed baseline before a performance-sensitive source edit. |
| C1 Data contracts | IN PROGRESS | Stable local result, composed request types, typed adapter, and first K=1 caller migration committed. Next: validate GPU parity, then migrate the K-class caller family. |
| C2 Policy/environment boundary | NOT STARTED | Classify and centralize 206 resolved `RECOVAR_*` reads. |
| C3 Diagnostics extraction | NOT STARTED | Depends on C1/C2 typed seams. |
| C4 Exact-local engine | NOT STARTED | Migrate host request first, JIT PyTree boundary second. |
| C5 Sparse pass 2 | NOT STARTED | Split 19,436-line module and break significance import cycle. |
| C6 Dense/global engine | NOT STARTED | Stabilize request/result and orchestration stages. |
| C7 Iteration controller | NOT STARTED | Decompose 5,564-line loop after engine boundaries stabilize. |
| C8 K-class/replay routing | NOT STARTED | Consume typed engine/controller boundaries. |
| C9 Duplicate/compatibility cleanup | NOT STARTED | Requires migrated production callers and evidence. |
| C10 Final acceptance | NOT STARTED | Focused, CPU guard, Slurm GPU parity, full K=1, K-class, and performance gates. |

## Baseline inventory

Recorded at HEAD `1e2f229b3e0e8edaec029d2b604937f692148578` on branch
`double_parity_refactor` with no tracked changes.

| Measure | Baseline |
|---|---:|
| Python files | 56 |
| Lines | 67,999 |
| Functions | 1,124 |
| Functions with >=10 args | 125 |
| Functions with >=20 args | 38 |
| Calls with >=15 args | 133 |
| Calls with >=20 args | 81 |
| Resolved `RECOVAR_*` environment reads | 206 names |
| Direct import cycles | 1 (`significance` <-> `sparse_pass2_bucketed`) |

Largest boundaries:

- `_run_relion_iteration_loop`: 5,564 lines;
- `compute_pass2_stats_sparse_bucketed`: 61 args / 4,155 lines;
- `compute_k_class_pass2_stats_sparse_fused`: 46 args / 3,704 lines;
- `run_local_em_exact`: 58 args / 3,348 lines;
- `run_local_bucket_big_jit`: 98 args / 843 lines;
- `_run_local_search_iteration`: 71 args / 393 lines;
- `_score_half_dense`: 58 args / 617 lines;
- `_score_half_local`: 56 args / 697 lines.

## Quality reference

Existing artifact:
`/home/ry295/palmer_scratch/tmp/recovar_em_test_regenerated`

Its `benchmark_ledger.json` records:

- commit `1e2f229b3e0e8edaec029d2b604937f692148578`;
- 13 completed numbered iterations;
- final-all-data path ran;
- final merged correlation versus RELION `0.9985789461317439`;
- final merged FSC-AUC versus RELION `0.995855447698812`;
- elapsed time `1184.1040608882904 s`;
- current sizes `46, 46, 72, 70, 70, 70, 70, 70, 70, 72, 72, 72, 72`.

This artifact is the initial quality reference. A fresh same-allocation control
is still required for performance claims because the ledger lacks a complete
GPU identity and paired timing context.

## Validation log

| Date | Scope | Command or artifact | Result |
|---|---|---|---|
| 2026-09-08 | Provenance | `git rev-parse HEAD`, branch/status/diff checks | HEAD and branch recorded; tracked tree clean; parity ancestors present. |
| 2026-09-08 | Static inventory | AST scan of `recovar/em/dense_single_volume/**/*.py` | Hotspots, argument/call counts, environment surface, and import cycle recorded in plan. |
| 2026-09-08 | Existing full K=1 reference | `/home/ry295/palmer_scratch/tmp/recovar_em_test_regenerated/benchmark_ledger.json` | Same-HEAD correlation/FSC-AUC/trajectory/runtime baseline recorded; no new GPU job submitted. |
| 2026-09-08 | Plan review | Naming and commit-discipline review | Adopted `ExecutionSettings` and `RefinementInputs`; added small, descriptive, independently revertible commit requirements. |
| 2026-09-08 | Local result contract | `pixi run python -m pytest tests/unit/test_local_em_types.py -q` | 17/17 passed; all legacy optional tuple shapes round-trip and malformed shapes fail closed. |
| 2026-09-08 | Composed request types | Same focused test file | 20/20 passed; request groups are immutable and capture-driven profile shape is preserved. |
| 2026-09-08 | Typed request adapter | Same focused test file | 21/21 passed; all 52 legacy keyword parameters are covered by exact name/value mapping. |
| 2026-09-08 | First production caller | `module load FFTW/3.3.10-GCC-12.2.0; pixi run python -m pytest tests/unit/test_refine_relion_mode.py -k 'run_local_search_iteration' -q` | 10 passed, 364 deselected. An initial run without the FFTW module failed four tests before the migrated path because the existing RELION binding could not load `libfftw3.so.3`. |
| 2026-09-08 | Local-search merge guards | `module load FFTW/3.3.10-GCC-12.2.0; pixi run python -m pytest tests/unit/test_dense_iteration_loop_merge_guards.py -q` | 27/27 passed. |
| 2026-09-08 | CPU EM fast guard | `pixi run test-em-fast-guard` | 16/16 passed after the result seam, after hook-required formatting, and after the K=1 caller migration. |

## Decision log

### 2026-09-08 — preserve existing typed foundations

Extend `RefinementOptions`, `DensePrecisionPolicy`, `HalfScoreResult`,
`PerHalfOutputs`, and existing result/layout types rather than introducing a
parallel framework.

### 2026-09-08 — no algebra-only consolidation

Numerically distinct RELION/JAX, dense/sparse, active/compact, and launch-order
variants remain explicit until focused tests establish the required
equivalence. Structural cleanup does not change the selected algorithm.

### 2026-09-08 — diagnostics have three classes

Classify diagnostics as passive, shadow, or invasive. Passive capture must not
select production outputs. Shadow arithmetic may add work but cannot replace
outputs. Invasive target-only/stop/force-split/state-swap behavior is explicitly
quarantined and cannot support normal quality or performance claims.

### 2026-09-08 — first implementation slice is host-side local request/result

Introduce a typed local-engine request/result behind the current compatibility
API, migrate production callers, and leave numerical code and the large JIT
signature unchanged. This establishes a safe seam before C3/C4.

### 2026-09-08 — naming and commit granularity

Use `ExecutionSettings` for batch, cache, bucket, fusion, and microbatch
settings. Use `RefinementInputs` for immutable datasets, initial model state,
and base grids; this is more specific than `RefinementData`, which could be
confused with particle data or carried iteration state.

Every accepted commit must contain one understandable structural change, have
a descriptive imperative message, include its focused tests/documentation, and
be independently revertible. Large mechanical moves and interface migrations
belong in separate commits, and caller families should migrate incrementally.

### 2026-09-08 — local-engine contract and first caller migration

Hypothesis: the host-side exact-local API can gain stable named results and a
composed request without changing the numerical engine, JIT boundary, selected
flags, or legacy monkeypatch surface.

Files changed: `local_em_types.py`, `local_em_engine.py`,
`local_search_iteration.py`, and `tests/unit/test_local_em_types.py`.

Algorithmic invariants protected:

- `run_local_em_exact` remains the only numerical implementation and retains
  its public signature and historical tuple result;
- the typed adapter forwards all 52 keyword parameters, with exact defaults;
- K=1 local search still invokes `iteration_loop.run_local_em_exact`, preserving
  existing monkeypatch and debug interception behavior;
- no kernel, array order, dtype, reduction, JIT signature, or environment
  lookup changed.

Focused tests and exact results: 21/21 local contract tests, 10/10 selected
local-search tests, and 27/27 merge/source guards passed. CPU fast guard: 16/16
passed. GPU/Slurm: pending first candidate submission. Quality and performance:
not yet measured for this slice; the existing reference remains correlation
`0.9985789461317439`, FSC-AUC `0.995855447698812`, and `1184.1040608882904 s`.
The adapter adds only frozen host objects around one engine invocation and does
not alter compiled work; empirical GPU confirmation is still required before
accepting the migrated production path.

Provenance: implementation started from `bc0e2954cc3b4e4ade20d4bb0a6b90e17c91735b`.
At `596d1c898b2d340d4a2f433bd70c5704e465a3f9`, the tracked tree was clean
(`e3b0c442...` diff SHA-256), all required parity ancestors were present, and
the unchanged pre-existing untracked manifest contained 1,427 files.

Commits:

- `d48f9659` — `style: format local EM engine` (isolated hook-required
  mechanical formatting; diff SHA-256 `13976cf1...`);
- `176fd3e4` — `em: add stable local-engine result contract` (diff SHA-256
  `917998e7...`);
- `0a1f967e` — `em: define cohesive local-engine request types` (diff SHA-256
  `fcf88aaa...`);
- `df425b0c` — `em: adapt typed local requests to exact engine` (diff SHA-256
  `d467098d...`);
- `596d1c89` — `em: migrate K=1 local search to typed engine request` (diff
  SHA-256 `083ecb51...`).

Decision: provisionally accepted pending the GPU K=1 parity run. Next action:
submit the user-provided multi-iteration command on Slurm `gpu`, compare final
correlation/FSC-AUC/runtime to the reference, and only then migrate K-class
callers. Open risk: actual GPU end-to-end behavior and timing are not measured
yet; the CPU evidence exercises the adapter and caller plumbing but cannot
replace that gate.

## Per-slice update template

Copy this block for each implementation slice:

```text
### YYYY-MM-DD — <component/slice>

Hypothesis:
Files changed:
Algorithmic invariants protected:
Focused tests and exact results:
CPU fast guard:
GPU/Slurm job IDs:
Quality artifacts and deltas:
Performance artifacts and deltas:
Compile/memory observations:
Provenance (HEAD, diff SHA-256, untracked manifest):
Commit SHA and descriptive message:
Decision: accepted / revised / reverted
Next action:
Open risks:
```

## Immediate next actions

1. Submit the user-provided K=1 multi-iteration parity command on Slurm `gpu`
   against commit `596d1c89`; preserve the existing reference output.
2. Compare final correlation and FSC-AUC, current-size/convergence trajectory,
   elapsed time, and available stage timings with the recorded reference.
3. If accepted, migrate the three direct K-class `run_local_em_exact` call sites
   as one caller-family commit with their focused tests.
4. Keep the large JIT signature unchanged until a paired baseline/candidate
   harness has been recorded on the same GPU allocation.
