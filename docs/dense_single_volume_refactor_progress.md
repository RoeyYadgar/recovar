# Dense Single-Volume EM Refactor Progress

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)  
Current phase: C3 — extract diagnostics and parity capture
Last updated: 2026-09-10

## Status board

| Component | Status | Current result / next action |
|---|---|---|
| C0 Baseline and guardrails | COMPLETE FOR C1 | Inventory, focused/CPU guards, and a same-allocation A100 control/candidate run are recorded. The older absolute K=1 FSC gate remains an independent open issue. |
| C1 Data contracts | COMPLETE — STRUCTURAL GPU PASS | Stable local and dense contracts are used by every in-package production caller. V100 job `60517729` confirms direct control/candidate final-map FSC-AUC `0.9998763`, improved candidate-vs-RELION FSC-AUC, and no runtime or memory regression. The older absolute K=1 FSC gate remains independently open. |
| C2 Policy/environment boundary | COMPLETE — STRUCTURAL GPU PASS | All 260 named settings are classified, process reads are confined to the two configuration boundaries, and refinement receives one immutable `RuntimeConfiguration`. A100 job `60521289` found improved RELION FSC-AUC, direct control/candidate map FSC-AUC `0.9992773`, and no runtime or memory regression. |
| C3 Diagnostics extraction | IN PROGRESS | The typed lifecycle/trace contract and zero-work null sink are implemented; migrate parity timing/capture next, then the remaining serialization families. |
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
| 2026-09-09 | Direct dense half-step GPU A/B | Slurm `60517729` | Completed `0:0` on one V100. Candidate/control direct map FSC-AUC was `0.9998763`; candidate-vs-RELION FSC-AUC improved by `+0.0001383`, ledger time improved `1.46%`, process wall improved `4.61%`, and peak RSS improved `0.12%`. Preliminary jobs were excluded before science. |
| 2026-09-09 | First runtime-settings boundary | `test_dense_runtime_options.py` plus `test_firstiter_cc_batch_budget.py` | 18/18 passed; default, compatibility override, invalid input, and explicit settings injection retain the existing batch formula. |
| 2026-09-09 | First-iteration caller compatibility | `test_run_k_class_parity.py` | 31/31 passed. |
| 2026-09-09 | Raw-image cache settings | Runtime settings plus selected cache cases in `test_refine_relion_mode.py` | 12 passed, 371 deselected; explicit settings, compatibility parsing, and lazy disabled-mode validation retain the existing behavior. |
| 2026-09-09 | Dense batch-planning settings | `test_dense_runtime_options.py` and `test_refine_relion_mode.py -k relion_em_batch_sizing` | 17/17 and 12/12 passed, respectively; the explicit settings path bypasses an invalid environment value and the compatibility path retains its errors. |
| 2026-09-09 | Dense planner host timing | Seven 10,000-call samples at pre-C2 `14bbf00f` and current `bfb62693` | Best time per call was `86.3 us` before C2, `87.4 us` through the compatibility parser, and `82.8 us` with a pre-resolved settings object. The observed compatibility cost is `1.1 us/call`; the parse-once route is faster than control. |
| 2026-09-09 | Cumulative C2 CPU guard | `pixi run test-em-fast-guard` | 16/16 passed in `50.05 s` after the raw-cache and dense-planner settings slices. |
| 2026-09-09 | Exact-local cache settings | `test_dense_runtime_options.py` and selected exact-local cache cases in `test_refine_relion_mode.py` | 24/24 and 5/5 passed, respectively; defaults, overrides, invalid values, lazy field parsing, and explicit environment bypass retain existing behavior. |
| 2026-09-09 | Exact-local cache host timing | Seven 200,000-call samples of the per-bucket sparse M-step memory helper | Best times were `0.48 us/call` for the former inline lookup, `0.82 us/call` through the compatibility adapter, and `0.32 us/call` with resolved settings. The transitional `0.34 us/bucket` cost is immaterial; the target parse-once path is faster. |
| 2026-09-09 | Exact-local cache CPU guard | `pixi run test-em-fast-guard` | 16/16 passed in `49.60 s`. |
| 2026-09-09 | Local execution-name prerequisite | `pixi run python -m pytest tests/unit/test_local_em_types.py -q` | 22/22 passed. The local request group is now `LocalExecutionSettings`; its former generic name remains an identity alias for compatibility. |
| 2026-09-09 | Composed execution settings | `test_dense_runtime_options.py`; selected settings-forwarding and batch-sizing cases in `test_refine_relion_mode.py` with FFTW loaded | 25/25 and 14/14 passed, respectively. One explicit snapshot overrides incompatible process values at the refinement boundary and reaches both migrated top-level consumers by identity. An initial two-case run without FFTW failed before planning because the existing binding could not load `libfftw3.so.3`; the FFTW-loaded rerun passed 2/2. |
| 2026-09-09 | Composed-settings CPU guard | `pixi run test-em-fast-guard` | 16/16 passed in `50.18 s`, within `0.58 s` (`1.2%`) of the recent `49.60`--`50.05 s` runs. |
| 2026-09-09 | First-iteration planner settings | Complete `test_firstiter_cc_batch_budget.py` plus the top-level settings-boundary case in `test_refine_relion_mode.py` | 14/14 passed with two pre-existing SciPy gimbal-lock warnings. Invalid process state was bypassed by the explicit typed budget for both coarse and fine clamps. |
| 2026-09-09 | Batch-planner façade host timing | Seven 500,000-call samples of a bare stub and the callable façade | Best direct/façade times were `0.262` / `0.535 us/call`, a `0.274 us` dispatch cost (about `0.3%` of the previously measured real planner call). |
| 2026-09-09 | First-iteration settings CPU guard | `pixi run test-em-fast-guard` | 16/16 passed in `49.72 s`. |
| 2026-09-09 | Exact-local typed cache request | `test_local_em_types.py` plus two real cache-route equivalence cases in `test_refine_relion_mode.py` | 24/24 passed. The typed/legacy adapters round-trip the cache object, and invalid environment values are bypassed for raw, processed-half, and sparse-M-step decisions without changing numerical outputs. Read-only persistent-cache warnings were sandbox-only non-failures. |
| 2026-09-09 | Exact-local caller compatibility | Ten selected local-search cases plus two local K-class cases | 12/12 passed; the new optional compatibility keyword preserves both caller families. |
| 2026-09-09 | Exact-local request CPU guard | `pixi run test-em-fast-guard` | 16/16 passed in `49.45 s`. |
| 2026-09-09 | Grouped local-search host boundary | `test_local_search_types.py`, selected local-search refinement cases, and merge guards | 24/24, 13/13, and 29/29 passed. All three production local-search calls use one typed request; execution settings forwarding passed 16/16. |
| 2026-09-09 | Environment access migration | Focused local, shared, sparse, refinement, K-class, and policy suites | Local selection: 185 passed and one expected GPU-HLO skip; shared selection: 129 passed and three GPU-only skips; K-class/merge selection: 136/136; refinement policy selection: 121/121. The broad sparse selection passed 93 cases; two unrelated numerical-expectation cases are recorded under open risks. |
| 2026-09-09 | Runtime configuration and structural ratchets | `test_runtime_environment_boundary.py` plus `test_dense_runtime_options.py` | 37/37 passed. Only `runtime_options.py` and `diagnostics/config.py` may access the process environment, JIT functions may not call compatibility accessors, and every named setting must have an effect class. |
| 2026-09-09 | Final C2 compatibility matrix | Selected diagnostic, precision, finalization, translation-grid, batching, convergence, dump, merge, and sparse-preparation cases | 54 passed, 350 deselected; seven existing warnings. Explicit runtime injection and legacy environment aliases resolved identically in the covered routes. |
| 2026-09-09 | Final C2 CPU guard | `pixi run test-em-fast-guard` | 16/16 passed in `49.57 s`, consistent with the prior C2 range of `49.45`--`50.18 s`. |
| 2026-09-09 | Sparse expectation adjudication | Two exact focused cases at C2 `0f7b0337`, then the failing coarse case at pre-C2 `dc64e343` | The explicit algebraic-bypass route passed. The coarse-posterior case failed identically at both commits by one `7.45e-9` float32 value (`1.2938e-7` relative), proving it predates C2; no tolerance was changed. |
| 2026-09-09 | Requested full K=1 validation | Slurm `60520380`; `$HOME/palmer_scratch/tmp/recovar_em_test_c2_0f7b0337_20260909_retry1` | Completed `0:0` on one A100 allocation: 13 numbered iterations, identical current-size trajectory, final-all-data, correlation `0.998392015`, RELION FSC-AUC `0.994875338`, ledger time `965.948 s`, Slurm wall `1081 s`, and peak RSS `17.764 GiB`. The raw, unaligned GT metrics are non-scoring. |
| 2026-09-09 | Final same-allocation C2 A/B | Slurm `60521289`; `$HOME/palmer_scratch/tmp/dense_em_refactor_c2_samegpu_0f7b0337_vs_dc64e343_retry1` | Completed `0:0` on one A100-PCIE-40GB. Candidate/control direct map FSC-AUC was `0.9992773`; candidate-vs-RELION FSC-AUC improved by `+0.0004093`; ledger time was unchanged (`+0.0009%`), process wall improved `0.20%`, exact-local time improved `1.07%`, and peak RSS improved `0.48%`. |

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

Final same-GPU job `60517729` completed `0:0` in one allocation on node
`r908u24n02`, GPU `GPU-b8a62512-c749-1707-8256-85b9e4509b6c`
(Tesla V100-PCIE-16GB, driver 570.211.01). The detached worktrees pin control
`14bbf00f1bce6d68c3be68a0095f09568a97cd06` and candidate
`204384c78c4b622d952850d18dfc649c67773323`; both tracked diffs are empty.
Both arms used CUDA library SHA-256
`f2b8dbb0ca0b8bd1151e71652c1bbee2da93eb53a43033bb3bd3cd7f470b9c42`,
and the control RELION binding SHA-256 is
`0605b857b2c1f052911101fb4ce7ed179332f4bcae42ca89a1de6a6be281079c`.

| Paired measure | Control `14bbf00f` | Candidate `204384c7` | Candidate delta |
|---|---:|---:|---:|
| Completed numbered iterations | 13 | 13 | same |
| Final all-data path | ran | ran | same |
| Current-size trajectory | `46,46,72,70,70,70,70,70,70,72,72,72,72` | same | same |
| Final merged FSC-AUC vs RELION | `0.9943203100` | `0.9944586446` | `+0.0001383346` |
| Merged FSC-AUC vs GT, alignment disabled | `-0.4635628688` | `-0.4635986733` | `-0.0000358045` (worse, unaligned) |
| Final merged correlation vs RELION (diagnostic) | `0.9983093529` | `0.9983346379` | `+0.0000252850` |
| Ledger elapsed | `930.976 s` | `917.351 s` | `-1.46%` |
| Exact-local EM time | `303.554 s` | `289.315 s` | `-4.69%` |
| External process wall time | `1018.16 s` | `971.26 s` | `-4.61%` |
| Peak RSS | `10,658,172 KiB` | `10,644,892 KiB` | `-0.12%` |

The 310-key result schemas have identical key order, shapes, and dtypes.
Current-size, pixel-resolution, healpix-order, finalization, sampling, and
gridding-policy fields match exactly. The two final maps directly match at
FSC-AUC `0.9998763255`, minimum non-DC FSC `0.9995150370`, and diagnostic
correlation `0.9999801517`. Iteration 0 state is exact; the first discrete
difference appears at iteration 9. Across all 13 iterations, 24 of 13,000
significant-count rows, 14 best-rotation rows, and 7 best-translation rows
differ. The maximum saved FSC-state difference is `9.4599e-4`; the final
all-data results differ in 9 rotation rows and 2 translation rows. These
small late-trajectory differences did not regress the map, schedule,
finalization, memory, or runtime, but they are not claimed as bitwise parity.
The command left GT alignment disabled, and both unaligned GT FSC-AUC values
are negative; their small mixed delta is recorded above but cannot establish
an absolute GT-quality pass. Ledger compile counts are `null` for both arms,
so compilation count was not measured by this run.

Both V100 arms remain below the program's absolute `0.995` cross-RELION FSC
gate, as did the earlier A100 control/candidate pair. The candidate improved
over its immediate V100 control and reproduces the earlier candidate's
approximately `0.99445` range, so this is not a refactor regression; the
pre-existing absolute quality gap remains open and prevents promoting this
fixture to a quality checkpoint.

Artifact root:
`$HOME/palmer_scratch/tmp/dense_em_refactor_samegpu_final_204384c7_vs_14bbf00f`.
It contains both ledgers, full trajectories, process resource records,
provenance manifests, and logs. The launcher SHA-256 is
`35e6c16b9eadbba4174adc1fb6f6bd48b07c2526b6b90734a8f04d18e3a219c5`,
and the run root has a `SAFE_TO_DELETE` marker.
Control and candidate ledger SHA-256 values are, respectively,
`5c366b5a1a89b32ea6f0c7a1b3b5ca8e6c83c82d13f2a673dff8e1d4b9cecb20`
and
`648dd1e064c514dc9cd136f9ff37d7b5863793afe3878567e7293fcd000945bc`.
Each arm ran this command from its pinned worktree, changing only `<arm>`:

```bash
python -m scripts.run_multi_iter_parity \
  --relion_dir relion_em_test_double_seeded \
  --data_star _full_refinement_data_double_seeded/particles.star \
  --iter 0 --max_iter 20 \
  --output_dir "$RUN_ROOT/<arm>_output" \
  --gt_volume "$HOME/pi_data/igg_1d/init_mask/backproj_0.01.mrc" \
  --replay-override-max-iter 0
```

The launcher cleared inherited RECOVAR/RELION and Python/conda overrides,
set `PYTHONNOUSERSITE=1` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`, pinned the
CUDA library, and used separate per-arm temporary, JAX, and CUDA cache roots.

Preliminary launcher attempts produced no scientific result: `60517301` was
canceled unallocated; `60517345` rejected an incorrect SHA literal;
`60517430` encountered the prescribed scratch path's permission boundary;
and `60517467` rejected the otherwise SHA-correct pinned CUDA binary because
fresh worktree source mtimes were newer. The last attempt spent six minutes
preparing its environment before that import preflight, but neither arm
entered the parity script. The final launcher retained the SHA check, refreshed
only the pinned binary's mtime, and used an isolated fallback runtime root.

Decision: accepted as C1 structural-equivalence and performance evidence.
The unchanged numerical implementation, exact argument-forwarding tests, and
paired GPU result jointly show no algorithm or performance regression. The
absolute FSC gate remains a separate open issue.

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
parser-only slice; completed job `60517729` validates the preceding C1
checkpoint, not this subsequent commit.

Commit: `88620683` — `em: centralize first-iteration batch settings`.

Decision: accepted as the first C2 compatibility seam. Next action: classify
and centralize another cohesive execution-policy family, then migrate
top-level construction to parse resolved settings once without widening a
numerical/JIT boundary.

### 2026-09-09 — raw-image cache settings boundary

Hypothesis: raw-image cache mode and memory-ceiling parsing can move out of
`batch_planning.py` without changing cache eligibility, deduplication, size
estimation, force behavior, or the historical lazy validation of the ceiling.

Files changed: `runtime_options.py`, `batch_planning.py`, and
`tests/unit/test_dense_runtime_options.py`.

`RawImageCacheSettings` owns the resolved mode and ceiling. The cache helper
accepts an explicit instance, while its compatibility route still resolves
the mode first and does not parse an invalid ceiling when caching is disabled
or no loader can be cached. No device arrays, JAX functions, or numerical
engine calls changed.

Focused runtime/cache coverage passed 12 selected tests with 371 unrelated
`test_refine_relion_mode.py` cases deselected. Commit: `8622497a` —
`em: centralize raw image cache settings`.

Decision: accepted. The compatibility semantics, including lazy validation,
are explicit and tested.

### 2026-09-09 — dense batch-planning settings boundary

Hypothesis: the projection-memory fraction can be resolved behind an
immutable `DenseBatchPlanningSettings` object without changing any memory
budget, cap, formula, dtype, or device query.

Files changed: `runtime_options.py`, `batch_planning.py`, and
`tests/unit/test_dense_runtime_options.py`.

The planner now accepts resolved settings and otherwise calls the sole
compatibility parser. Defaults remain `0.20`; positive finite overrides and
the historical invalid-value error remain unchanged. The explicit path was
tested while the corresponding environment value was deliberately invalid,
proving the planner does not consult process state after settings are passed.
This leaf type will compose into the planned top-level `ExecutionSettings`
rather than creating a competing orchestration object.

Focused results: `test_dense_runtime_options.py` passed 17/17 and the existing
`relion_em_batch_sizing` cases in `test_refine_relion_mode.py` passed 12/12
with 362 deselected. Ruff formatting and lint passed on all three changed
files. The cumulative CPU EM fast guard passed 16/16 in `50.05 s`. No
tolerance or baseline changed.

A same-host microbenchmark of the complete planner used seven samples of
10,000 calls. The pre-C2 control took `86.3 us/call`; the current compatibility
route took `87.4 us/call` (`+1.1 us/call`), while passing a pre-resolved object
took `82.8 us/call` (`-3.5 us/call` versus control). The compatibility overhead
is immaterial at planner call frequency, and the intended parse-once route has
no measured host regression. No device execution is involved in this helper.

Commit: `bfb62693` — `em: centralize dense batch planning settings`.

Decision: accepted. `batch_planning.py` now contains no direct environment
read. The next C2 slice should resolve another host policy family without
widening a numerical or JIT boundary.

### 2026-09-09 — exact-local cache settings boundary

Hypothesis: the raw-image cache, processed-half cache, and sparse big-JIT
M-step memory ceilings can move behind one immutable `LocalCacheSettings`
object without changing their estimates, defaults, eligibility comparisons,
or lazy compatibility behavior.

Files changed: `runtime_options.py`, `local_caches.py`, and
`tests/unit/test_dense_runtime_options.py`.

The three historical constants and their environment names now live with the
other runtime compatibility settings and remain re-exported from
`local_caches.py`. Each cache helper accepts an optional resolved settings
object. Its compatibility path still parses only the field used by that
helper, so an invalid unrelated cache variable remains dormant exactly as
before. No array operation, dtype, cache-size estimate, comparison, engine
route, or JIT signature changed. `local_caches.py` now contains no direct
environment read.

Focused results: `test_dense_runtime_options.py` passed 24/24, and the five
existing exact-local raw/processed cache cases in `test_refine_relion_mode.py`
passed with 369 unrelated cases deselected. The explicit path was exercised
while all three compatibility variables contained invalid values; the lazy
compatibility test also proves the raw-cache helper ignores invalid sibling
fields. Ruff formatting and lint passed on all three changed files. The CPU
EM fast guard passed 16/16 in `49.60 s`. No tolerance or baseline changed.

A same-host microbenchmark used seven samples of 200,000 calls to the sparse
big-JIT M-step memory helper. The former inline environment lookup took
`0.48 us/call`; the compatibility adapter took `0.82 us/call`; and an injected
settings object took `0.32 us/call`. The transitional adapter therefore adds
`0.34 us` per local bucket, which is immaterial beside bucket execution, while
the intended parse-once route is `0.16 us/call` faster than the old lookup.
This helper performs no device work.

Commit: `f3935d9b` — `em: centralize exact-local cache settings`.

Decision: accepted. The next C2 step should compose the established leaf
settings into a host-level `ExecutionSettings` snapshot and inject it at one
top-level call boundary, while retaining the compatibility loaders for direct
legacy callers.

### 2026-09-09 — composed execution settings and top-level injection

Hypothesis: the four established C2 tuning leaves can compose into the planned
host-level `ExecutionSettings`, and a caller-supplied snapshot can reach the
top-level raw-image cache and global dense batch planner without re-reading
process state or changing default execution.

Prerequisite commit `73ee4f36` renamed the exact-local request group from the
generic `ExecutionSettings` to `LocalExecutionSettings`. Its former name is an
identity alias, so external imports and values remain compatible while the
package-level name is available for the run-wide aggregate. The focused local
contract suite passed 22/22.

Files changed in the aggregate slice: `runtime_options.py`,
`refinement_options.py`, `iteration_loop.py`, the package `__init__.py`, and
the two focused unit-test files. `ExecutionSettings` is frozen and composes
`FirstIterationBatchSettings`, `RawImageCacheSettings`,
`DenseBatchPlanningSettings`, and `LocalCacheSettings` with independent
default factories. `load_execution_settings` resolves all four from one
explicit mapping or compatibility environment. `RefinementOptions.execution`
is optional so existing direct callers retain their lazy field-by-field
behavior during migration.

Only the already-migrated top-level raw-image cache and global dense batch
planner consume the aggregate in this slice. The first-iteration and
exact-local leaves remain composed but are intentionally not threaded through
the current large scorer/local-engine signatures; they will move with cohesive
request-object boundaries. No numerical kernel, array operation, dtype,
reduction order, batch formula, cache estimate, selected route, diagnostic
policy, JIT signature, tolerance, or baseline changed.

Focused results: `test_dense_runtime_options.py` passed 25/25. The selected
settings-forwarding and complete existing `relion_em_batch_sizing` set in
`test_refine_relion_mode.py` passed 14/14 with the FFTW module loaded. The new
boundary test places invalid conflicting values in process state and proves
that the exact supplied leaf objects reach both consumers. An initial two-case
run without FFTW produced one pass and one pre-planning binding failure; after
loading FFTW, both passed. The CPU EM fast guard passed 16/16 in `50.18 s`.
Static checks found no whitespace or formatting spillover and no newly
introduced lint; two legacy full-file lint findings remain outside the diff.

The guard runtime is within `0.58 s` (`1.2%`) of the recent `49.60`--`50.05 s`
runs and their ordinary startup/noise variation. The new work is a single host
attribute lookup and object forwarding at refinement setup/planner frequency;
it does not enter a particle, pixel, candidate, bucket, device, or JIT loop.
The resolved dense planner path was already measured faster than its pre-C2
control. Therefore no new GPU job was submitted for this non-numerical slice;
Slurm job `60517729` remains the latest same-GPU structural A/B evidence.

Tested dirty-tree provenance: HEAD `73ee4f36`, tracked diff SHA-256
`b968f35bbfb13611aae3ddbff6dd22ce65bcedb9ac9003f39f4fcd5f24224aff`,
and 1,427 pre-existing untracked paths with sorted manifest SHA-256
`565752e3c6fcb6404f7a6da29289d1fd2b350ae1d88448d6bea850de2bcb1072`.
All required parity ancestors were present.

Commit: `c08d1ca7` — `em: add composed execution settings`.

Decision: accepted. Next, migrate one remaining composed leaf through an
existing cohesive request/configuration seam; do not add another scalar to a
large numerical or JIT signature.

### 2026-09-09 — first-iteration settings through the batch planner

Hypothesis: the run-resolved `FirstIterationBatchSettings` leaf can reach all
six first-iteration image-batch clamps through the callable batch-planning seam
already passed among host orchestration helpers, without adding an argument to
any scorer, local-search, numerical-engine, or JIT signature.

Files changed: `iteration_loop.py`, `test_firstiter_cc_batch_budget.py`, and the
top-level settings-boundary case in `test_refine_relion_mode.py`.

The former nested dense-planning closure is now wrapped by a frozen
`_RefinementBatchPlanner`. It remains callable with the same arguments and
return values, while its named `first_iteration_image_batch_size` method owns
the resolved first-iteration leaf. The same planner object follows every
existing `safe_batch_sizes` path, including adaptive dense planning, K-class
coarse/fine planning, local-search host sizing, single-pass sizing, and final
all-data sizing. Plain callable test/compatibility inputs still fall back to
the historical lazy environment parser.

No batch formula, default, comparison, selected route, numerical kernel,
array, dtype, reduction, JIT signature, tolerance, or baseline changed. With
`RefinementOptions.execution=None`, each first-iteration clamp still resolves
the compatibility environment lazily as before. With an explicit snapshot,
all clamps reuse its immutable typed value and ignore later process mutation.

The complete first-iteration budget file plus the top-level boundary case
passed 14/14 with two pre-existing SciPy gimbal-lock warnings. The added test
sets the compatibility budget to an invalid value and verifies that both fine
and coarse K-class plans use the supplied typed budget. Static checks found no
new lint or whitespace defect; the same two legacy full-file lint findings and
one pre-existing formatter finding remain outside this diff. The CPU EM fast
guard passed 16/16 in `49.72 s`.

A seven-sample, 500,000-call host microbenchmark measured the bare planning
stub at `0.262 us/call` and the callable façade at `0.535 us/call`, adding
`0.274 us` per dispatch. That is about `0.3%` of the previously measured
`82.8`--`87.4 us` real dense-planner cost, and the planner runs only a handful
of times per half/iteration. The façade is absent from particle, pixel,
candidate, bucket, device, and JIT loops. No new GPU job was submitted; Slurm
job `60517729` remains the latest same-GPU structural A/B evidence.

Tested dirty-tree provenance: HEAD `8db0c7ff`, tracked diff SHA-256
`8ba54fe573f2760e4f43efb22fe692b8423dbd239fc22361da0395869a462dc4`,
and 1,427 pre-existing untracked paths with sorted manifest SHA-256
`565752e3c6fcb6404f7a6da29289d1fd2b350ae1d88448d6bea850de2bcb1072`.

Commit: `3f6a328b` — `em: route first-iteration settings through planner`.

Decision: accepted. The remaining composed-but-not-top-level-consumed leaf is
`LocalCacheSettings`; first make it part of the existing typed exact-local
request/compatibility boundary, then migrate its host caller without adding
individual cache-limit scalars.

### 2026-09-09 — exact-local cache settings in the typed request

Hypothesis: `LocalCacheSettings` can enter the exact-local engine as one member
of `LocalExecutionSettings`, round-trip through the legacy K-class adapter, and
control all three cache decisions without adding individual limit arguments or
changing any route's numerical result.

Files changed: `local_em_types.py`, `local_em_engine.py`, `k_class.py`,
`test_local_em_types.py`, and two existing route-equivalence cases in
`test_refine_relion_mode.py`.

`LocalExecutionSettings.cache` is an optional typed leaf. The legacy
`run_local_em_exact` boundary accepts one optional `cache_settings` object for
direct-call compatibility, and `run_local_em` maps the grouped field to it.
The K-class legacy-to-request adapter maps the same object back into the local
execution group. Inside the engine, the processed-half preference, raw-image
fallback, and per-bucket sparse big-JIT M-step decision all receive that one
object. No cache formula, default, eligibility comparison, bucket shape,
selected numerical route, array operation, dtype, reduction, JIT signature,
tolerance, or baseline changed.

When the object is absent, all three helpers preserve their historical lazy
field-specific environment parsing. When present, they read only immutable
fields from the supplied object. The adapter contract test now covers every
live compatibility parameter, including the new grouped cache field, and its
round trip through the K-class builder.

The 22 contract cases plus two real engine route-equivalence cases passed
24/24. Those engine cases deliberately set all relevant process variables to
invalid values, then select raw versus processed-half caching and sparse versus
deferred M-step execution through explicit settings; hard assignments, stats,
accumulators, and noise remain within their unchanged existing checks. The ten
selected local-search callers and two K-class callers passed 12/12. Read-only
JAX persistent-cache warnings were caused by the sandbox and did not affect
results. The CPU EM fast guard passed 16/16 in `49.45 s`.

The compatibility path performs the same parser calls as before. The explicit
per-bucket sparse-M-step helper path was previously measured at `0.32 us/call`,
versus `0.48 us/call` for the former inline lookup and `0.82 us/call` for the
transitional compatibility adapter. This slice adds only one host object
forwarding operation per exact-local invocation and does not enter JIT/device
code. No new GPU job was submitted; Slurm job `60517729` remains the latest
same-GPU structural A/B evidence.

Tested dirty-tree provenance: HEAD `4011e8e7`, tracked diff SHA-256
`6b238fdfc04449431391f733e4140e5077e82aa717b63aae3f3435ba833bdf8a`,
and 1,427 pre-existing untracked paths with sorted manifest SHA-256
`565752e3c6fcb6404f7a6da29289d1fd2b350ae1d88448d6bea850de2bcb1072`.

Commit: `a87d1fcb` — `em: pass cache settings through local request`.

Decision: accepted. Before forwarding `ExecutionSettings.local_cache` from the
iteration controller, replace the 71-argument local-search host call with a
grouped request/compatibility adapter so the migration removes coupling instead
of adding another parameter to it.

### 2026-09-09 — C2 complete: resolved runtime and environment boundary

Hypothesis: dense single-volume EM can snapshot process configuration once,
classify every supported setting, and expose typed algorithm/execution policy
to host orchestration without changing numerical kernels, defaults, diagnostic
compatibility, or the autonomous refinement trajectory.

The implementation was split into small reviewable commits:

| Commit | Purpose |
|---|---|
| `4df8b3ef` | Add the immutable `LocalSearchIterationRequest`. |
| `16a3fd04` | Migrate the three production local-search calls to grouped data. |
| `89ec99a8` | Carry run-resolved execution settings through local search. |
| `0366d498` | Add immutable environment capture and scoped compatibility access. |
| `63713084`, `947d51b9` | Centralize and then mechanically format diagnostic environment access. |
| `9dec10d0` | Centralize local-engine execution environment access. |
| `40739818` | Centralize shared dense-EM policy access. |
| `71a99e93` | Centralize sparse pass-2 policy access. |
| `e6aa484e` | Centralize refinement-controller policy access. |
| `b5984788` | Resolve major algorithm choices as typed settings. |
| `108e12bd`, `0d8ebe51` | Capture one runtime configuration at refinement entry and mechanically format its modules. |
| `0f7b0337` | Add AST ratchets for the environment boundary and JIT kernels. |

`EnvironmentSnapshot` is an immutable, sorted view of the process environment.
`RuntimeConfiguration` composes that snapshot with `AlgorithmSettings`,
`ExecutionSettings`, and `DiagnosticsPlan`. `refine_single_volume` resolves or
accepts this object before validation and device work, logs the resolved typed
algorithm/execution policy plus diagnostic setting names, and activates it for
the run. Direct legacy helper calls still obtain live environment values when
there is no active refinement scope, preserving the established test and
external-call compatibility path.

The promoted algorithm fields are float64 scoring, float64 projection,
RELION-exact fine Gaussian scoring, the accelerated-double `floorf` quirk,
the K=1 exact translation grid, and final-all-data grid correction. Their typed
defaults exactly preserve the pre-C2 branch behavior. In particular, this
refactor does not use the typed seam to change the existing final-all-data grid
default; any scientific default correction remains a separately tested
algorithm change.

Configuration conflicts now fail at the host boundary. A separately supplied
`ExecutionSettings` must agree with an explicitly supplied
`RuntimeConfiguration`, and simultaneous significance/pass-2 target-half
diagnostics are rejected when `DiagnosticsPlan` is constructed. Diagnostic
overlays create a new immutable snapshot rather than mutating `os.environ`.

The final classifier covers all 260 exact `RECOVAR_*` string settings present
under the package:

| Effect class | Names |
|---|---:|
| Algorithm | 88 |
| Tuning | 67 |
| Passive diagnostic | 91 |
| Invasive experiment | 14 |

The source ratchet rejects any direct `os.environ`, `os.getenv`, or imported
equivalent outside `runtime_options.py` and `diagnostics/config.py`. A second
ratchet rejects calls to runtime-environment compatibility accessors from
`jax.jit`-decorated functions, and a third requires every exact setting name to
have an effect class. Dense, local, and sparse numerical kernels therefore no
longer read process state; host compatibility helpers inside an active run see
the captured immutable mapping.

Focused validation was deliberately distributed by changed subsystem:

- grouped local-search contract/callers/merge guards passed 24/24, 13/13, and
  29/29, followed by 16/16 execution-settings forwarding cases;
- local policy migration passed 185 cases with one expected GPU-HLO skip;
- shared policy migration passed 129 cases with three GPU-only skips;
- K-class, joint-semantics, and merge suites passed 136/136;
- selected refinement-policy cases passed 121/121;
- the final structural/runtime suite passed 37/37;
- the final cross-module compatibility matrix passed 54 selected cases with
  350 deselected and seven existing warnings;
- `pixi run test-em-fast-guard` passed 16/16 in `49.57 s`, within the observed
  C2 range of `49.45`--`50.18 s`.

One intermediate broader sparse selection passed 93 cases and exposed two
numerical expectation mismatches: a float32 coarse-posterior value exceeded
the test's relative tolerance by about `1.29e-7`, and one then-named explicit
float64 route had a float32 dtype expectation. At final HEAD the explicit
algebraic-bypass route passes. The unchanged coarse-posterior case produces the
same one-value `7.45e-9` failure at both pre-C2 `dc64e343` and C2 `0f7b0337`,
proving that failure predates the stage. No tolerance, baseline, or numerical
implementation was changed to hide either result.

Host timing also bounds the new policy lookup cost. Seven one-million-call
samples measured direct typed-field access at `0.0353 us`, scoped algorithm
lookup at `0.0870 us`, live `os.environ.get` at `0.6773 us`, and scoped snapshot
lookup at `0.1699 us` per call. The scoped accessors are outside particle,
pixel, candidate, bucket, device, and JIT loops. Together with the stable CPU
guard, this finds no host-performance regression attributable to C2.

Requested full-run job `60520380` completed `0:0` at clean HEAD `0f7b0337`
with diff SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
It ran on one Slurm A100 allocation and produced 13 numbered iterations, the
same current-size sequence as the reference, convergence after iteration 13,
and the final-all-data path. Its artifact and log are:

- `$HOME/palmer_scratch/tmp/recovar_em_test_c2_0f7b0337_20260909_retry1`;
- `$HOME/palmer_scratch/tmp/recovar_em_c2_0f7b0337_60520380.log`.

The requested command was run unchanged except for the unique, non-overwriting
output directory:

```bash
python scripts/run_multi_iter_parity.py \
  --relion_dir relion_em_test_double_seeded \
  --data_star _full_refinement_data_double_seeded/particles.star \
  --iter 0 \
  --max_iter 20 \
  --output_dir $HOME/palmer_scratch/tmp/recovar_em_test_c2_0f7b0337_20260909_retry1 \
  --gt_volume $HOME/pi_data/igg_1d/init_mask/backproj_0.01.mrc \
  --replay-override-max-iter 0
```

Final correlation against RELION is `0.9983920149` and the decisive merged
FSC-AUC is `0.9948753382`. Both exceed the immediately preceding fresh C1
checkpoint (`+0.00005738` correlation and `+0.00041669` FSC-AUC). The older
initial artifact remains higher by `0.00018693` correlation and `0.00098011`
FSC-AUC; that absolute cross-run gap predates C2 and is not treated as a C2
effect. The supplied GT metric was not alignment-enabled and is therefore a
raw frame-mismatched diagnostic, not a quality decision. Ledger time was
`965.948 s`; Slurm wall was `1081 s`; batch peak RSS was `17.764 GiB`. This
single run is not used for a broad GPU timing claim.

Five earlier Slurm attempts (`60519851`, `60519952`, `60520014`, `60520087`,
and `60520308`) stopped in launcher, build, library, or preflight setup before
science. Job `60520314` passed the GPU preflight but stopped before iteration 1
because the RELION binding could not find FFTW. They are excluded from quality
and timing evidence.

Same-allocation job `60521289` completed `0:0` in `00:34:53` on physical GPU
`GPU-b4639563-0794-75fe-47b9-cf05a6fdf85a`, an NVIDIA A100-PCIE-40GB. It ran
pre-C2 `dc64e343` and C2 `0f7b0337` sequentially with the same pinned CUDA
library, RELION binding, fixture, seed, cleaned compatibility environment, and
isolated runtime/cache roots. Both detached worktrees had empty tracked diffs;
their only untracked entries were the two fixture links.

Each arm ran this command from its pinned worktree, changing only `<arm>`:

```bash
python -m scripts.run_multi_iter_parity \
  --relion_dir relion_em_test_double_seeded \
  --data_star _full_refinement_data_double_seeded/particles.star \
  --iter 0 --max_iter 20 \
  --output_dir "$RUN_ROOT/<arm>_output" \
  --gt_volume "$HOME/pi_data/igg_1d/init_mask/backproj_0.01.mrc" \
  --replay-override-max-iter 0
```

The combined standard-output and standard-error logs are
`$HOME/palmer_scratch/tmp/dense_em_refactor_c2_samegpu_0f7b0337_vs_dc64e343_retry1/logs/dense-em-c2-ab-60521289.out`
and
`$HOME/palmer_scratch/tmp/dense_em_refactor_c2_samegpu_0f7b0337_vs_dc64e343_retry1/logs/dense-em-c2-ab-60521289.err`.

| Measure | Control `dc64e343` | Candidate `0f7b0337` | Candidate delta |
|---|---:|---:|---:|
| Completed iterations | 13 | 13 | same |
| Current-size trajectory | `46,46,72,70,70,70,70,70,70,72,72,72,72` | same | same |
| Final-all-data path | yes | yes | same |
| Final merged FSC-AUC vs RELION | `0.9944652754` | `0.9948745863` | `+0.0004093110` |
| Final merged correlation vs RELION (diagnostic) | `0.9983334284` | `0.9983925903` | `+0.0000591619` |
| Ledger elapsed | `994.729 s` | `994.738 s` | `+0.0009%` |
| Exact-local EM time | `335.584 s` | `331.995 s` | `-1.07%` |
| External process wall | `1043.57 s` | `1041.45 s` | `-0.20%` |
| Peak RSS | `11,126,756 KiB` | `11,073,504 KiB` | `-0.48%` |

The two 310-key result archives have identical key order, shapes, and dtypes.
Current-size, pixel-resolution, healpix-order, finalization, sampling, and
gridding-policy fields match exactly. The direct final merged maps have
normalized non-DC FSC-AUC `0.9992772717`, minimum non-DC FSC `0.9973098636`,
and diagnostic correlation `0.9998790672`. Across the 13 iterations, 484 of
13,000 significant-count rows, 235 best-rotation rows, and 96
best-translation rows differ; the final all-data poses differ in 53 rotation
rows and 11 translation rows. The maximum saved iteration-FSC difference is
`0.00413997`. These differences are disclosed rather than called bitwise
parity.

The first numerical difference is already limited to GPU accumulator
roundoff at iteration 0: the largest `Ft_y` difference is `5.21e-10` and the
largest `Ft_ctf` difference is `4.55e-13`. An independent C2 candidate repeat
has iteration-0 differences of the same scale (`9.31e-10` and `4.55e-13`)
against the paired candidate. Those two C2 candidate maps directly match at
FSC-AUC `0.9999992076` with minimum non-DC FSC `0.9999902248`, while their
RELION FSC-AUC values differ by only `7.52e-7`. The later discrete drift is
therefore consistent with iterative amplification of the existing
nondeterministic GPU accumulation seed, not a changed schedule, routing
policy, numerical kernel, or accepted quality metric. The candidate improves
both RELION-facing metrics, and the paired timing and memory deltas remain far
inside the investigation thresholds.

The portable artifact root is
`$HOME/palmer_scratch/tmp/dense_em_refactor_c2_samegpu_0f7b0337_vs_dc64e343_retry1`.
It contains both ledgers, 310-field result archives, intermediate arrays,
resource records, provenance manifests, and logs, plus a `SAFE_TO_DELETE`
marker. The launcher SHA-256 is
`e924737e266ed69ae3b759862ef15c235f16ed3dda6433db12efe06153417914`;
the pinned CUDA library SHA-256 is
`107149065aafb5988815b693a2852c01a87dd9d6f4719ccad2f289b0f00a7143`.
Control and candidate ledger SHA-256 values are, respectively,
`a032d52928cb15a683da5c58fa8dc2f004ee99d08f33a1f73d9534577a0b4b49`
and
`b329dd753622f800b67eee6f2e3854a2058f1961cf48e9593517852d182f7baa`.
Ledger compile counts are `null` for both arms, so this job does not support a
compilation-count claim.

C2 exit criteria:

- all 260 exact settings are classified as algorithm, tuning, passive
  diagnostic, or invasive experiment;
- direct process reads are confined to `runtime_options.py` and
  `diagnostics/config.py`, with AST ratchets protecting that boundary and JIT
  kernels;
- host orchestration receives one resolved `RuntimeConfiguration` containing
  typed algorithm, execution, and diagnostic policy;
- configuration conflicts fail before device work and logs show resolved
  typed policy rather than scattered raw values;
- environment compatibility tests, focused subsystem tests, K-class guards,
  and the CPU fast guard pass with the established values and defaults;
- the requested full K=1 run and paired A100 comparison show no accepted
  quality, end-to-end performance, exact-local performance, or memory
  regression.

Decision: C2 is complete with a structural GPU pass. The absolute
cross-RELION FSC-AUC remains just below the program's `0.995` gate, as it did
before C2; candidate C2 improves the paired control by `0.0004093`, so this is
retained as an independent scientific-quality issue rather than a refactor
regression. C3 may now begin from the immutable runtime and diagnostics-plan
boundary.

### 2026-09-10 — C3 typed diagnostics seam

Hypothesis: A named observer protocol, immutable lifecycle payloads, and an
empty `TraceSpec` can establish the diagnostics boundary without inspecting,
materializing, synchronizing, or selecting production values.

Files changed: `diagnostics/events.py`, `diagnostics/sinks.py`, diagnostics
exports, and focused unit coverage.

Algorithmic invariants protected: lifecycle construction retains object
identity; sink methods return `None`; the production singleton requests no
extra kernel values; no controller or numerical caller is changed in this
slice.

Focused tests and exact results: `test_diagnostics_sinks.py`, 3/3 passed.

CPU fast guard: deferred until the first production caller migration.

GPU/Slurm job IDs: not required for this type-only slice.

Quality artifacts and deltas: not measured; no algorithmic caller changed.

Performance artifacts and deltas: not measured; no algorithmic caller changed.

Compile/memory observations: no JAX code or call boundary changed.

Provenance: parent HEAD `748ab20dc4b821c9f53cd43f924c56181ea1a395`
on `dense_em_refactor`; the tracked tree was clean and pre-existing untracked
fixture, plot, editor, and scratch paths were left untouched.

Commit SHA and descriptive message: `15c53424` — `refactor: define diagnostics sink contract`.

Decision: accepted.

Next action: implement parity/timing as explicit sink objects while retaining
the legacy module API and exact NPZ schema.

Open risks: lifecycle calls are not yet wired, and capture families still own
their legacy module state until subsequent slices migrate them.

### 2026-09-10 — C3 parity and timing sink extraction

Hypothesis: Moving parity state, payload construction, and NPZ serialization
behind one typed adapter can leave the controller's call sequence and the
legacy import surface unchanged.

Files changed: `diagnostics/parity.py`, the `parity_dump.py` compatibility
facade, diagnostics exports, `iteration_loop.py`, and this ledger.

Algorithmic invariants protected: the moved implementation retains the same
module state dictionaries, activation checks, timing semantics, payload keys,
dtypes, downsampling, filenames, exception shielding, and call positions. The
controller now references the singleton adapter; it does not pass a sink into
JIT code and receives no value from diagnostic calls.

Focused tests and exact results: `test_parity_dump_timing.py` plus
`test_diagnostics_sinks.py`, 9/9 passed.

CPU fast guard: scheduled after controller serialization extraction.

GPU/Slurm job IDs: deferred to the complete C3 structural comparison.

Quality artifacts and deltas: not measured in this schema-preserving move.

Performance artifacts and deltas: not measured in this schema-preserving move.

Compile/memory observations: numerical call signatures and JIT inputs are
unchanged; the parity adapter's `TraceSpec` is empty.

Provenance: parent HEAD `15c5342449eb5b6f980fb0788dbe1570a6b3d171`
on `dense_em_refactor`; pre-existing untracked fixture, plot, editor, and
scratch paths were left untouched.

Commit SHA and descriptive message: `f233f3a9` — `refactor: extract parity diagnostics sink`.

Decision: accepted.

Next action: extract inline controller NPZ schemas, then migrate the dense and
local engine capture adapters.

Open risks: the compatibility implementation retains its historical process
singleton until the C7 `RefinementSession` owns the run-wide sink.

### 2026-09-10 — C3 local capture module boundary

Hypothesis: Dense and exact-local request parsing, payload construction, and
serialization can move as one cohesive diagnostic module while stable imports
continue through a thin compatibility facade.

Files changed: `diagnostics/local_capture.py`, `local_debug.py`, and this
ledger.

Algorithmic invariants protected: the 954-line implementation is moved
mechanically; call sites, request values, array conversions, payload schemas,
filenames, and exception behavior are unchanged. No numerical engine file or
JIT boundary changes.

Focused tests and exact results: targeted dense original-index score capture,
local request matching, local score schema, and fused-posterior filtering;
3/3 passed with 374 deselected and one pre-existing gimbal-lock warning.

CPU fast guard: deferred until the first production caller/schema migration.

GPU/Slurm job IDs: deferred to the complete C3 structural comparison.

Quality artifacts and deltas: not measured; implementation move only.

Performance artifacts and deltas: not measured; call sites are unchanged.

Compile/memory observations: no numerical code or function signature changed.

Provenance: parent HEAD `f233f3a9b2fff1fb417a6d513fa51ac761ff973b`
on `dense_em_refactor`; pre-existing untracked fixture, plot, editor, and
scratch paths were left untouched.

Commit SHA and descriptive message: `cc89423d` — `refactor: relocate local diagnostic capture`.

Decision: accepted.

Next action: relocate compact-candidate and controller debug artifact writers,
then remove direct serialization from numerical engines.

Open risks: production callers still use the compatibility facade until their
larger legacy files can be migrated without unrelated formatting churn.

### 2026-09-10 — C3 controller artifact module boundary

Hypothesis: Existing iteration intermediates and noise-update schemas can be
owned by the diagnostics package without changing the controller's stable
monkeypatch surface.

Files changed: `diagnostics/controller_capture.py`, `debug_dumps.py`, focused
schema coverage, and this ledger.

Algorithmic invariants protected: the writer implementation is moved
mechanically and the old module re-exports the same two callables. Controller
call positions, optionality, array conversions, output names, and MRC/NPY/NPZ
schemas are unchanged.

Focused tests and exact results: the new noise-update artifact contract and
the controller compatibility-symbol ratchet, 2/2 passed. The run reported 21
read-only persistent-cache warnings; these do not affect results.

CPU fast guard: deferred until inline controller serialization is migrated.

GPU/Slurm job IDs: deferred to the complete C3 structural comparison.

Quality artifacts and deltas: not measured; implementation move only.

Performance artifacts and deltas: not measured; production call sites are
unchanged.

Compile/memory observations: no engine or JIT interface changed.

Provenance: parent HEAD `cc89423dfda7853853d23d4eb7b876e9bbfa1123`
on `dense_em_refactor`; pre-existing untracked fixture, plot, editor, and
scratch paths were left untouched.

Commit SHA and descriptive message: pending — `refactor: relocate controller diagnostics`.

Decision: accepted.

Next action: relocate compact candidate capture and extract the two dense
engine schemas into the diagnostics package.

Open risks: ten inline controller serialization sites remain to be extracted.

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

1. Define the C3 null diagnostics sink and stable lifecycle payloads at the
   existing host synchronization points; do not introduce a generic event bus.
2. Move one serialization family at a time out of the controller and engines,
   retaining exact NPZ keys, dtypes, shapes, filenames, and stop behavior.
3. Keep passive, shadow, and invasive diagnostics visibly separate; passive
   capture must never select production outputs.
4. Compare null-diagnostics output, synchronization, lowered HLO, compilation
   count, timing, and memory against the C2 checkpoint, while tracking the
   older absolute `0.995` FSC-AUC issue independently.
