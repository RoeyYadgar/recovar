# Dense Single-Volume EM Refactor Progress

Last updated: 2026-09-11

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

C4 audit: [`dense_single_volume_refactor_audit_2026-09-11.md`](dense_single_volume_refactor_audit_2026-09-11.md)

C4.5 inventory: [`dense_single_volume_refactor_c45_inventory.md`](dense_single_volume_refactor_c45_inventory.md)

Historical C0--C4 log:
[`dense_single_volume_refactor_progress_archive_c0_c4.md`](dense_single_volume_refactor_progress_archive_c0_c4.md)

Use `$REPO_ROOT` for the checkout and `$HOME` for user-owned artifacts in this
file. Do not record user-specific absolute paths.

## Status board

| Phase | Status | Outcome / next gate |
|---|---|---|
| C0 Baseline | Complete | Inventory, quality reference, focused tests, and performance guardrails recorded. |
| C1 Data contracts | Complete | Typed dense/local request and result boundaries retained. |
| C2 Settings boundary | Complete | `RuntimeConfiguration`, `AlgorithmSettings`, and `ExecutionSettings` remain the resolved host boundary. |
| C3 Diagnostics | Complete | Diagnostic persistence and effects have dedicated owners; obsolete debug re-export shims are gone. |
| C4 Exact-local engine | Complete | Grouped JAX boundary and planning seams retained. |
| C4.5 Foundation consolidation | Complete | Accepted at `d6bf42da`: all structural, focused-test, CPU, and paired GPU quality/performance gates pass. |
| C5 Sparse pass 2 | Ready | Split and simplify the sparse pass from the accepted C4.5 checkpoint. |
| C6--C10 | Not started | Follow the authoritative plan in order. |

## Current structural scorecard

Scope: `recovar/em/dense_single_volume/**/*.py`.

| Measure | C4 checkpoint `6041093d` | Current C4.5 | Gate | Result |
|---|---:|---:|---:|---|
| Production Python files | 75 | 72 | <=74 | Pass |
| Production lines | 71,509 | 69,212 | <=70,509 | Pass |
| Nonblank production lines | 65,975 | 63,751 | decrease | Pass |
| Functions/methods | 1,254 | 1,239 | decrease | Pass |
| Classes | 165 | 152 | <=155 | Pass |
| Functions with >=20 args | 37 | 35 | <=35 | Pass |
| Calls with >=20 args | 75 | 65 | <=65 | Pass |
| Largest function span | 5,620 | 5,500 | decrease | Pass |

The current values include the repository formatter normalization commits.
Those commits contain no algorithm changes and were isolated so the semantic
refactors remain readable.

## C4.5 completed implementation slices

| Commit | Slice | Structural outcome |
|---|---|---|
| `a1035d25` | Canonical typed local EM | `run_local_em(request)` owns the implementation; the historical exact-local function is one-way compatibility only. |
| `8baa9ee7` | Canonical typed dense EM | `run_dense_em(request)` owns the implementation; `run_em` is an external compatibility facade. |
| `c0500a79` | Canonical typed local search | Removed the 71-argument internal implementation and variable result tuple. |
| `b4b9ed14` | One-way result compatibility | Removed reverse tuple parsers and output-shape records; only outward serialization remains. |
| `9236e221` | Shared local request groups | Removed duplicate local-search correction, reconstruction, and diagnostic records. |
| `2acb58f7` | Diagnostic owners | Deleted the obsolete `local_debug.py` and `debug_dumps.py` re-export modules. |
| `fa307aa0`, `ec36abc5` | Local planning | Folded one-lifecycle Fourier, route, and bucket records into their owning plans. |
| `a4540108` | State-swap diagnostics | Replaced a 20-argument diagnostic helper with one grouped state value. |
| `81789206` | K-class engine views | Removed signature reflection and raw local/dense K-class request round trips. |
| `fcad2e89`, `e8090e01` | Routing/result cleanup | Collapsed single-use wrappers and constructed typed results directly. |
| `032a1b9d` | Parity facade cleanup | Deleted the unused parity re-export module. |
| `95442a5a`, `79df6209` | Mechanical formatting | Isolated repository-required formatting from semantic changes. |
| `884c7a6f` | Shared local-search settings | Local search now consumes exact-local search, scoring, projection, and output groups directly. |
| `981c810c` | Payload mappings | Expressed six existing option/payload bags as mapping literals, completing the long-call gate. |
| `bf30e335`, `d6bf42da` | Ledger/test cleanup | Split the active ledger from its archive and migrated local-search tests to the shared settings contracts. |

## C4.5 invariants

- Normal in-package execution is typed-to-canonical with no
  typed-to-legacy-to-tuple round trip.
- Compatibility facades are one-way, outside hot loops, and preserve their
  historical return tuples.
- JAX inputs, array order, numerical branches, dispatch policy, and diagnostic
  artifact schemas are unchanged.
- Dense/local K-class engines receive typed requests; sparse routing remains
  explicitly deferred to C5/C8.
- `RuntimeConfiguration` and `ExecutionSettings` remain host-owned settings
  snapshots. Lower JAX functions do not receive the runtime context.

## C4.5 validation ledger

| Scope | Result |
|---|---|
| Dense/local request and compatibility contracts | 31 passed after direct result construction. |
| Shared local-search request groups and merge guards | 49 passed. |
| Dense, local, and K-class typed factory slice | 77 passed. |
| Local Fourier planning/caches | 18 passed. |
| Local microbatch planning | 6 passed. |
| State-swap grouping | 26 passed. |
| Sparse/significance/K-class mapping slice | 108 passed; four additional tests could not load `libfftw3.so.3` in the host environment before their assertions. |
| Settings/diagnostics/import boundaries | 13 assertions passed; the host-side `pixi` process was interrupted after pytest completion when its cleanup did not exit. |
| CPU fast guard | 16 passed in 52.02 s. |
| Prescribed full K=1 Slurm GPU replay | Job `60564274` completed `0:0` on an A100-PCIE-40GB: 13 numbered iterations, expected size trajectory, final-all-data, correlation `0.9983948853`, FSC-AUC `0.9948839316`, and ledger time `988.889 s`. |
| Same-allocation C4/C4.5 gate | Clean reverse-order job `60569641` completed `0:0`; trajectory and 310-field result schema matched, quality improved, and all runtime/memory deltas stayed inside the guardrails. |

No test tolerance, expected numerical value, or quality threshold has been
changed during C4.5.

## Reference quality and performance

The initial same-code artifact is
`$HOME/palmer_scratch/tmp/recovar_em_test_regenerated`. Its ledger records
final merged correlation `0.9985789461317439`, RELION FSC-AUC
`0.995855447698812`, and elapsed time `1184.1040608882904 s` across 13
completed numbered iterations plus final-all-data.

The prescribed standalone C4.5 artifact is
`$HOME/palmer_scratch/tmp/c45_validation_981c810c_20260911`. Job `60564274`
completed in `00:18:29` on an A100-PCIE-40GB with peak batch RSS
`12,645,360 KiB`. Its cold-cache timing is not compared directly with the
older warm-cache reference.

The decisive quality/performance artifact is
`$HOME/palmer_scratch/tmp/c45_samegpu_clean_d6bf42da_vs_60b2b154_20260911`.
Job `60569641` ran clean detached C4.5 candidate `d6bf42da` followed by C4
control `60b2b154` on the same A100-PCIE-40GB with identical pinned CUDA and
RELION bindings and separate fresh caches:

| Measure | C4 control | C4.5 candidate | Candidate delta |
|---|---:|---:|---:|
| Completed iterations / final-all-data | 13 / yes | 13 / yes | same |
| Final correlation vs RELION | `0.9983087334` | `0.9983323956` | `+0.0000236623` |
| Final FSC-AUC vs RELION | `0.9943136480` | `0.9944544068` | `+0.0001407588` |
| Ledger elapsed | `971.486 s` | `984.628 s` | `+1.35%` |
| Exact-local EM | `321.321 s` | `325.453 s` | `+1.29%` |
| Process wall | `1017.68 s` | `1032.48 s` | `+1.45%` |
| Transfer to host | `7.882 s` | `8.006 s` | `+1.58%` |
| Peak RSS | `11,057,104 KiB` | `11,073,212 KiB` | `+0.15%` |

The direct candidate/control merged-map comparison has correlation
`0.9999801377`, non-DC FSC-AUC `0.9998748181`, and minimum non-DC FSC
`0.9994840284`. Both result archives contain the same 310 keys in the same
order, with identical shapes and dtypes. C4.5 therefore introduces no material
quality, runtime, transfer, or memory regression.

Job `60564455` was excluded from the decisive gate because a concurrent
uncommitted sparse-pass edit appeared in the primary checkout between its
control and candidate arms. That edit remains preserved and is not part of
C4.5. Job `60569641` used detached worktrees with empty tracked diffs for both
arms.

## Immediate next actions

1. Begin C5 from the accepted `d6bf42da` implementation checkpoint.
2. Inventory sparse-pass-2 routes and delete only dead/shadow paths with
   focused evidence.
3. Establish the smallest typed sparse orchestration boundary before any
   module split, preserving candidate order, dtypes, reductions, and JIT
   topology.
