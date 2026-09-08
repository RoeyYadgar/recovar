# Dense Single-Volume EM Refactor Progress

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)  
Current phase: C1 — introduce data contracts behind compatibility APIs
Last updated: 2026-09-08

## Status board

| Component | Status | Current result / next action |
|---|---|---|
| C0 Baseline and guardrails | COMPLETE FOR C1 | Inventory, focused/CPU guards, and a same-allocation A100 control/candidate run are recorded. The older absolute K=1 FSC gate remains an independent open issue. |
| C1 Data contracts | IN PROGRESS | Stable local result, composed request types, typed adapter, and the K=1 caller migration are accepted against their immediate control. Next: migrate the K-class caller family. |
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
| 2026-09-08 | GPU launcher bootstrap | Slurm `60510554`, `60510585` | Exited `127` during launcher/module bootstrap before Python import or science; excluded from quality and timing comparisons. |
| 2026-09-08 | Full K=1 candidate | Slurm `60510625` | Completed on A100-SXM4-80GB: 13 iterations, final-all-data, correlation `0.9983320461`, FSC-AUC `0.9944496194`, `841.865 s`. This unpaired result did not meet the older absolute FSC gate. |
| 2026-09-08 | Same-GPU candidate repeat | Slurm `60510827` | Candidate completed on A100-SXM4-80GB with correlation `0.9983335769`, FSC-AUC `0.9944632395`, `813.722 s`; control setup then failed before science because the detached worktree lacked the ignored RELION binding. |
| 2026-09-08 | Corrected candidate repeat | Slurm `60510947` | Candidate completed on the same physical A100 as `60510827`: correlation `0.9983333994`, FSC-AUC `0.9944556956`, `821.328 s`; control stopped before iteration 1 when fresh checkout timestamps triggered a CUDA rebuild without `nvcc`. |
| 2026-09-08 | Final paired K=1 A/B | Slurm `60511038` | Both arms completed in one allocation on the same A100 with the same pinned CUDA binary. Control/candidate FSC-AUC: `0.9944628985` / `0.9944491811`; exact-local time: `279.213` / `276.223 s`; full details below. |

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
passed.

GPU validation used the full command requested for this refactor, with a fresh
output directory for each arm. Final paired job `60511038` ran the immediate
pre-migration control `df425b0c` first and candidate `e47a128a` second in one
allocation on node `r818u09n09`, GPU
`GPU-986d5e56-846f-1864-8fc9-9ef20b0b5b88` (A100-SXM4-80GB, driver
570.211.01). Both arms used CUDA library SHA-256
`f2b8dbb0ca0b8bd1151e71652c1bbee2da93eb53a43033bb3bd3cd7f470b9c42` and
the control used the same RELION binding binary as the candidate (SHA-256
`0605b857b2c1f052911101fb4ce7ed179332f4bcae42ca89a1de6a6be281079c`).
Both tracked trees were clean (empty diff SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`) and
contained all five required parity ancestors. The candidate's preserved
untracked manifest contains the original 1,427 files; the control's two
untracked entries are input-fixture symlinks.

Artifact root:
`/home/ry295/palmer_scratch/tmp/dense_em_refactor_samegpu_final_e47a128a_vs_df425b0c`.
It contains both output ledgers, complete intermediate trajectories, process
resource records, environment/provenance manifests, and logs under `logs/`.
The launcher SHA-256 is
`a7c40272bdf5956adb50ef5e4a895a2b14f8e69dc58f66257e1e30c34bedec64`.
The run root and all earlier refactor run roots have `SAFE_TO_DELETE` markers.
Earlier attempt artifacts and logs are retained under
`/home/ry295/palmer_scratch/tmp/dense_em_refactor_e47a128a`,
`/home/ry295/palmer_scratch/tmp/dense_em_refactor_samegpu_e47a128a_vs_df425b0c`,
and
`/home/ry295/palmer_scratch/tmp/dense_em_refactor_samegpu_retry_e47a128a_vs_df425b0c`.

| Paired measure | Control `df425b0c` | Candidate `e47a128a` | Candidate delta |
|---|---:|---:|---:|
| Completed numbered iterations | 13 | 13 | same |
| Final all-data path | ran | ran | same |
| Current-size trajectory | `46,46,72,70,70,70,70,70,70,72,72,72,72` | same | same |
| Final merged FSC-AUC vs RELION | `0.9944628985` | `0.9944491811` | `-0.0000137174` |
| Final merged correlation vs RELION (diagnostic) | `0.9983339935` | `0.9983321029` | `-0.0000018907` |
| Ledger elapsed | `845.936 s` | `870.393 s` | `+2.89%` |
| Exact-local EM time | `279.213 s` | `276.223 s` | `-1.07%` |
| External process wall time | `897.06 s` | `946.83 s` | `+5.55%` |
| Peak RSS | `11,519,084 KiB` | `11,529,036 KiB` | `+0.086%` |

The 310-key result schemas match. Schedule, resolution, finalization, and final
sampling policy fields match exactly. The maximum per-iteration FSC-state gap
was `1.2222e-4`, below the `1.6429e-4` gap observed between two unchanged
candidate runs. Across all iterations the pair differed in 7 significant-count
rows, 8 best-rotation rows, and 3 best-translation rows; an unchanged-candidate
repeat differed in 9, 7, and 5 rows respectively. The three candidate FSC-AUC
values span `0.9944491811` to `0.9944632395`, so the paired quality delta lies
inside native repeat variability.

The ledger and external wall totals are mixed rather than a demonstrated speed
improvement: candidate total time was higher, but the exact local engine—the
only production path changed by the caller migration—was slightly faster, and
nearly all of the total delta was outside that engine. Candidate ledger times
across the completed runs span `813.722` to `870.393 s`; peak memory is
unchanged to `0.1%`. This is sufficient performance sanity for the host-only
adapter, not a formal performance benchmark.

The older artifact remains materially higher in absolute FSC-AUC
(`0.9958554477`). Both the new control and candidate are below the program's
`0.995` cross-FSC gate. Because the immediate pre-migration control reproduces
the candidate's quality and divergences begin in shared pre-local execution,
that absolute drift is not attributed to this refactor. It remains open and
prevents treating this small fixture as a new quality checkpoint.

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

Decision: accepted for C1 structural equivalence against the immediate control;
not accepted as a replacement K=1 quality checkpoint because the absolute FSC
gate is still missing. Next action: migrate the K-class caller family as one
small commit with its focused tests, retaining `run_local_em_exact` as the
compatibility/numerical implementation. Open risk: independently explain or
stabilize the old-to-current absolute FSC-AUC drift before any broad algorithmic
quality claim.

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

1. Migrate the three direct K-class `run_local_em_exact` call sites as one
   caller-family commit and preserve their monkeypatch/debug interception
   surface.
2. Run `tests/unit/test_k_class_joint_semantics.py`, the relevant merge guards,
   and the CPU EM fast guard before accepting that caller migration.
3. Keep the large JIT signature and all numerical kernels unchanged until a
   dedicated paired GPU benchmark is designed for that boundary.
4. Track the absolute FSC-AUC gap between the older artifact and both arms of
   job `60511038` separately from structural-refactor equivalence.
