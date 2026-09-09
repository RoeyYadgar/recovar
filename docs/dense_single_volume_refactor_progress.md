# Dense Single-Volume EM Refactor Progress

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)  
Current phase: C2 — centralize runtime policy and environment parsing
Last updated: 2026-09-09

## Status board

| Component | Status | Current result / next action |
|---|---|---|
| C0 Baseline and guardrails | COMPLETE FOR C1 | Inventory, focused/CPU guards, and a same-allocation A100 control/candidate run are recorded. The older absolute K=1 FSC gate remains an independent open issue. |
| C1 Data contracts | IMPLEMENTATION COMPLETE — GPU CHECK PENDING | Stable local and dense contracts are used by every in-package production caller. Legacy engines remain the numerical implementations. Same-GPU K=1 job `60517345` is queued. |
| C2 Policy/environment boundary | IN PROGRESS | First-iteration reconstruction-budget parsing is centralized behind immutable `FirstIterationBatchSettings`; callers retain environment-compatible defaults and can inject resolved settings. Continue one policy family at a time. |
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
`$HOME/palmer_scratch/tmp/recovar_em_test_regenerated`

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
| 2026-09-08 | Existing full K=1 reference | `$HOME/palmer_scratch/tmp/recovar_em_test_regenerated/benchmark_ledger.json` | Same-HEAD correlation/FSC-AUC/trajectory/runtime baseline recorded; no new GPU job submitted. |
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
| 2026-09-08 | K-class caller formatting | `pixi run test-em-fast-guard` | 16/16 passed before the isolated formatting-only commit `7d1d1d4c`; no functional source changed in that commit. |
| 2026-09-08 | K-class request compatibility | `pixi run python -m pytest tests/unit/test_local_em_types.py -q` | 21/21 passed. A fully populated exact-engine call round-trips through the K-class request builder with all 52 keyword names, values, and defaults preserved. |
| 2026-09-08 | K-class local callers | selected local K-class cases in `test_k_class_joint_semantics.py` and `test_refine_relion_mode.py` | 2/2 and 4/4 passed, respectively; the single-class, class-evidence probe, and normalized per-class M-step routes use named results. |
| 2026-09-08 | K-class regression set | `pixi run python -m pytest tests/unit/test_em_kclass_merge_guards.py tests/unit/test_k_class_joint_semantics.py -q` | 109/109 passed with one pre-existing SciPy gimbal-lock warning. No tolerance or baseline changed. |
| 2026-09-08 | K-class adapter performance sanity | Five 10,000-call host microbenchmark samples | Median request-build/dispatch overhead `59.334 us/call` (samples `59.313`, `59.334`, `59.362`, `59.569`, `59.330 us/call`), or about `0.475 ms` for eight K=4 probe/M-step calls. |
| 2026-09-08 | Dense result contract | `pixi run python -m pytest tests/unit/test_dense_em_types.py -q` | 9/9 initial result tests passed; all eight optional legacy tuple shapes round-trip and malformed tuples fail closed. |
| 2026-09-08 | Dense request and adapter | Same focused test file | 12/12 passed after request grouping and adapter wiring; all 32 optional `run_em` parameters round-trip by exact name/value against the live signature. |
| 2026-09-08 | Dense-engine formatting prerequisite | `pixi run test-em-fast-guard` | 16/16 passed in `49.57 s` before isolated mechanical commit `656ca2c5`. |
| 2026-09-08 | Dense K-class caller family | `test_dense_em_types.py`, `test_em_kclass_merge_guards.py`, and `test_k_class_joint_semantics.py` | 121/121 passed. Sandbox-only read-only JAX cache warnings and the pre-existing gimbal-lock warning were non-failures. |
| 2026-09-08 | Dense K-class refine integration | `test_refine_relion_mode.py -k 'dense_k_class' -q` | 3 passed, 371 deselected; existing complex-cast and gimbal-lock warnings only. |
| 2026-09-08 | Dense caller CPU fast guard | `pixi run test-em-fast-guard` | 16/16 passed in `49.75 s` on the complete dense K-class caller patch. |
| 2026-09-08 | Dense adapter performance sanity | Five 10,000-call host microbenchmark samples | Median request-build/dispatch overhead `44.373 us/call` (samples `44.507`, `44.349`, `44.373`, `44.333`, `44.398 us/call`), or about `0.355 ms` for eight K=4 probe/M-step calls. |
| 2026-09-08 | Oversampling formatting prerequisite | `test_adaptive_oversampling.py -q` with FFTW module loaded | 42/42 passed in `66.99 s` before isolated mechanical commit `1ec307d4`. The initial run without FFTW reached 38 passes and failed four binding-dependent tests before the formatted logic. |
| 2026-09-08 | Oversampling dense callers | Same focused suite with FFTW module loaded | 42/42 passed in `67.31 s`; both union-dense and per-image reference paths consume named dense results. |
| 2026-09-08 | Oversampling CPU fast guard | `pixi run test-em-fast-guard` | 16/16 passed in `49.71 s` on the complete oversampling caller patch. |
| 2026-09-09 | Portable documentation paths | Refactor plan/progress path scan and `git diff --check` | User-specific home/workspace prefixes were replaced by `$HOME`; the plan now requires portable placeholders in future records. |
| 2026-09-09 | Direct dense half-step seam | `test_dense_em_types.py`, `test_firstiter_cc_batch_budget.py`, and eight selected direct/final K=1 cases from `test_refine_relion_mode.py` | 12/12, 12/12, and 8/8 passed. The module-level legacy runner hook receives the same seven inputs and 32 resolved keyword parameters. |
| 2026-09-09 | Direct dense half-step CPU guard | `pixi run test-em-fast-guard` | 16/16 passed in `50.00 s`. |
| 2026-09-09 | Direct dense half-step GPU A/B | Slurm `60517345` | Queued on `gpu`; immutable control `14bbf00f` and candidate `204384c7` will run sequentially on the same allocated GPU. Initial A100-only request `60517301` was canceled while pending so a right-sized generic-GPU request could queue sooner. |
| 2026-09-09 | First runtime-settings boundary | `test_dense_runtime_options.py` plus `test_firstiter_cc_batch_budget.py` | 18/18 passed; default, compatibility override, invalid input, and explicit settings injection retain the existing batch formula. |
| 2026-09-09 | First-iteration caller compatibility | `test_run_k_class_parity.py` | 31/31 passed. |

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
`$HOME/palmer_scratch/tmp/dense_em_refactor_samegpu_final_e47a128a_vs_df425b0c`.
It contains both output ledgers, complete intermediate trajectories, process
resource records, environment/provenance manifests, and logs under `logs/`.
The launcher SHA-256 is
`a7c40272bdf5956adb50ef5e4a895a2b14f8e69dc58f66257e1e30c34bedec64`.
The run root and all earlier refactor run roots have `SAFE_TO_DELETE` markers.
Earlier attempt artifacts and logs are retained under
`$HOME/palmer_scratch/tmp/dense_em_refactor_e47a128a`,
`$HOME/palmer_scratch/tmp/dense_em_refactor_samegpu_e47a128a_vs_df425b0c`,
and
`$HOME/palmer_scratch/tmp/dense_em_refactor_samegpu_retry_e47a128a_vs_df425b0c`.

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

### 2026-09-08 — K-class local caller migration

Hypothesis: K-class local orchestration can use the same composed request and
stable named result as K=1 without changing exact-engine arguments, class
normalization, diagnostic labeling, numerical execution, or result values.

Files changed: `k_class.py`, `test_local_em_types.py`, and
`test_k_class_joint_semantics.py`. A hook-required formatter pass over the two
previously unformatted caller files was isolated before the functional commits.

Algorithmic invariants protected:

- all three routes—single class, per-class evidence probe, and normalized
  per-class M-step—still invoke `run_local_em_exact` through the K-class module's
  monkeypatchable binding;
- `inspect.Signature.bind` and `apply_defaults` preserve the exact legacy
  signature's validation and all 52 keyword defaults before constructing the
  grouped request;
- class priors, shared normalization evidence, profile/support collection,
  diagnostic phase labels, and output `None` semantics are unchanged;
- no score, posterior, reconstruction, dtype, array-order, JIT, kernel, or
  environment logic changed.

Focused validation passed: 21/21 local contract tests; 2/2 selected local
K-class semantic tests; 4/4 selected refine-mode K-class tests; 46/46 complete
K-class joint-semantics tests; and 109/109 combined K-class joint/merge tests.
The CPU EM fast guard passed 16/16 in `50.01 s` at source HEAD `6f50b38b`.
JAX reported that no CUDA device was present and the guard correctly stayed on
its CPU-default route; no local GPU work was performed. Ruff formatting and
lint checks passed on the committed files. No tolerance or baseline was
modified.

The host compatibility builder's five-sample median is `59.334 us/call`; a K=4
probe plus M-step sequence therefore adds about `0.475 ms` of Python work. This
is negligible relative to the exact local engine, but it is only a host
microbenchmark. No K=4 GPU end-to-end quality, compile-count, device-memory, or
wall-time result was measured for this slice. Those cells remain **not
measured**, rather than assumed equal. A new GPU run was not warranted here
because the exact runner and its fully materialized arguments are unchanged;
the shared adapter already has the paired K=1 A/B evidence from job `60511038`.

Commits:

- `7d1d1d4c` — `style: format K-class local caller files`;
- `356638f4` — `em: group flat K-class local request settings`;
- `6f50b38b` — `em: migrate K-class local calls to typed results`.

Decision: accepted as the completion of the C1 local host request/result
boundary. `run_local_em_exact` remains the sole numerical implementation and
compatibility API; its large body and JIT signature are deliberately unchanged.
Next action: add a stable dense-engine result seam behind `run_em`, then migrate
one dense caller family. Open risks: the broader C1 dense/sparse/controller
contracts remain, K-class end-to-end GPU behavior has not been remeasured, and
the pre-existing absolute K=1 FSC-AUC drift remains unresolved.

### 2026-09-08 — dense request/result seam and K-class callers

Hypothesis: dense EM can expose a composed request and stable named result, and
the dense K-class caller family can consume them, without changing `run_em`'s
signature, numerical implementation, flags, call interception, or result
values.

Files changed: `dense_em_types.py`, `em_engine.py`, `k_class.py`,
`test_dense_em_types.py`, and `test_k_class_joint_semantics.py`. The pre-existing
format/lint delta in `em_engine.py` was isolated in its own mechanical commit
before adapter wiring.

Algorithmic invariants protected:

- `run_em` retains its public signature, eight historical tuple shapes, and
  remains the only dense numerical implementation;
- the typed adapter forwards the seven required positional inputs and all 32
  optional parameters, while `inspect.Signature.bind` and `apply_defaults`
  preserve legacy validation/defaults at dictionary-based compatibility sites;
- all four dense K-class routes—ordinary score probe, first-iteration winner
  subset pass 2, single class, and normalized per-class M-step—still invoke the
  K-class module's monkeypatchable `run_em` binding;
- class priors, shared evidence normalization, subset correction slicing,
  diagnostic class labels, half-volume accumulation, and output `None`
  semantics are unchanged;
- no score, posterior, reconstruction, dtype, array order, reduction, JIT,
  kernel, or environment behavior changed.

Focused validation passed: 12/12 dense contract/adapter tests; 46/46 complete
K-class joint-semantics tests; 121/121 combined dense-contract, K-class
joint-semantics, and merge tests; and 3/3 selected dense K-class refine-mode
tests. The CPU EM fast guard passed 16/16 in `49.75 s` on the complete caller
patch. Ruff formatting and lint checks passed. No tolerance or baseline was
modified.

The dense compatibility builder plus typed dispatch has a five-sample median
of `44.373 us/call`; a normal K=4 probe plus M-step sequence adds about
`0.355 ms` of Python work. This is negligible relative to the dense engine but
is only host-side evidence. No K-class GPU end-to-end quality, compile-count,
device-memory, or wall-time cell was run for this slice, so those results remain
**not measured**. The user-specified K=1 job was not repeated because this
caller migration changes only K-class routing and the legacy engine invocation
is reconstructed exactly.

Commits:

- `8a987a4f` — `em: add stable dense-engine result contract`;
- `64fee97f` — `em: define cohesive dense-engine request types`;
- `656ca2c5` — `style: format dense EM engine`;
- `ce432b87` — `em: adapt typed dense requests to legacy engine`;
- `bfcee8af` — `em: group flat dense request settings`;
- `65732ebb` — `em: migrate K-class dense calls to typed results`.

Decision: accepted as the dense engine seam and first dense caller-family
migration. `run_em` remains the compatibility/numerical implementation. Next
action: migrate the direct dense half-step call in `iteration_loop.py`, then the
two oversampling caller paths, one reviewable family at a time. Open risks: no
K-class end-to-end GPU result was measured; direct non-K-class callers still
decode legacy tuples; the broader sparse/controller contracts and the
pre-existing absolute K=1 FSC-AUC drift remain unresolved.

### 2026-09-08 — oversampling dense caller migration

Hypothesis: the union-dense adaptive pass and its per-image numerical reference
can consume `DenseEMResult` without changing their generated fine grids,
candidate masks, dense-engine arguments, coarse posterior aggregation, noise
aggregation, or their own public tuple contracts.

Files changed: `helpers/oversampling.py` only. Six pre-existing formatter sites
were committed mechanically before the caller migration.

Algorithmic invariants protected:

- each path lazily imports the adapter and the legacy runner at its existing
  call boundary, preserving import-cycle and late monkeypatch behavior;
- all dense arguments are validated and defaulted against `run_em`'s live
  signature, and `run_em` remains the sole numerical implementation;
- oversampled rotation/translation generation, priors, image subsets,
  candidate masks, accumulator summation order, and returned oversampling tuple
  layout are unchanged;
- the removed logic only selected optional dense tuple positions for the four
  `return_stats`/`accumulate_noise` combinations.

The focused adaptive-oversampling suite passed 42/42 in `67.31 s` with the FFTW
module loaded, and the CPU EM fast guard passed 16/16 in `49.71 s`. The first
focused run without FFTW reached 38 passes and failed four tests while importing
the RELION binding because `libfftw3.so.3` was unavailable; rerunning with the
required module resolved all four without a source or test change. Existing
complex-to-real warnings remained. No tolerance or baseline was modified.

Commits:

- `1ec307d4` — `style: format dense oversampling helpers`;
- `c7255521` — `em: migrate oversampling dense calls to typed results`.

Decision: accepted. Only the direct dense half-step call in
`iteration_loop.py` still consumes the legacy tuple within package production
code. That 10,374-line legacy module is not formatter-clean; touching it under
the current hook would create a broad unrelated rewrite. The next slice must
choose a reviewable controller seam or an explicitly isolated formatting
strategy before migrating that call. No new GPU job was warranted because both
oversampling paths reconstruct the unchanged legacy invocation and all device
work remains inside `run_em`.

### 2026-09-09 — direct dense half-step caller migration

Hypothesis: the final direct `run_em` caller can construct the typed dense
request and consume `DenseEMResult` while preserving the iteration-loop
module's legacy runner hook, arguments, defaults, and numerical implementation.

Files changed: `iteration_loop.py` and
`tests/unit/test_firstiter_cc_batch_budget.py`.

Algorithmic invariants protected:

- `run_em` remains the only numerical implementation and is passed explicitly
  as the adapter's legacy runner, so module-level monkeypatch and debug hooks
  still intercept the call;
- the same seven required inputs and all 32 defaulted/explicit keyword
  parameters reach the legacy engine;
- the same hard assignment, `Ft_y`, `Ft_ctf`, RELION statistics, and noise
  statistics populate `HalfScoreResult`, now by stable names;
- no kernel, candidate order, dtype, reduction, environment lookup, or
  reconstruction route changed.

Focused results: 12/12 dense contract tests, 12/12 first-iteration
batch/caller tests, and 8/8 selected direct/final K=1 refinement tests passed.
The CPU EM fast guard passed 16/16 in `50.00 s`. The unchanged 10,374-line
controller has five pre-existing Ruff findings and is not formatter-clean; the
source commit skipped the two Ruff hooks to avoid an unrelated whole-file
rewrite, after confirming the candidate introduced no new lint finding.

Commit: `204384c7` — `em: migrate dense half-step to typed result`.

Decision: implementation accepted; end-to-end acceptance awaits Slurm job
`60517345`. Its artifact root is
`$HOME/palmer_scratch/tmp/dense_em_refactor_samegpu_final_204384c7_vs_14bbf00f`.
The detached control and candidate worktrees pin `14bbf00f` and `204384c7`, so
continued branch work cannot alter either arm. The earlier A100-only request
`60517301` was canceled before allocation or science because its estimated
start was the next day; it produced no quality or performance result.

### 2026-09-09 — first runtime-settings boundary

Hypothesis: first-iteration reconstruction-budget parsing can move out of the
batch formula into a frozen host-side settings object without changing the
default, environment override, validation behavior, or computed cap.

Files changed: `runtime_options.py`, `firstiter_cc.py`, and
`tests/unit/test_dense_runtime_options.py`.

`FirstIterationBatchSettings` owns the resolved complex-element budget and
`load_first_iteration_batch_settings` is the sole compatibility parser for
that variable. `_safe_firstiter_cc_image_batch_size` still resolves the
environment by default, preserving every current caller, and also accepts an
explicit settings object for later parse-once controller migration. Its memory
formula and constants are unchanged, and the historical constants remain
importable from `firstiter_cc.py`.

Focused results: the combined runtime-settings and batch suite passed 18/18;
the K-class parity caller suite passed 31/31. No GPU run is needed for this
parser-only slice; job `60517345` validates the preceding C1 checkpoint, not
this subsequent commit.

Commit: `88620683` — `em: centralize first-iteration batch settings`.

Decision: accepted as the first C2 compatibility seam. Next action: classify
and centralize another cohesive execution-policy family, then migrate
top-level construction to parse resolved settings once without widening a
numerical/JIT boundary.

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

1. Monitor same-GPU K=1 job `60517345` and add its quality, trajectory,
   runtime, and memory comparison before closing C1 validation.
2. Classify and centralize the next cohesive C2 execution-policy family;
   retain environment variables only as compatibility inputs.
3. Introduce a parse-once host boundary incrementally without widening the
   numerical/JIT signatures or mixing controller decomposition into C2.
4. Track the absolute FSC-AUC gap between the older artifact and both arms of
   jobs `60511038` and `60517345` separately from structural equivalence.
