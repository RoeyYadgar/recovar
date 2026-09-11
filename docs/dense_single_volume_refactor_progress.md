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
| C4.5 Foundation consolidation | Validation in progress | All static exit gates pass. Run final CPU guard and Slurm GPU parity/performance validation, then unblock C5. |
| C5 Sparse pass 2 | Blocked by C4.5 validation | Split and simplify the sparse pass only after C4.5 acceptance. |
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
| CPU fast guard | Pending final rerun; earlier C4.5 run passed 16/16 in 51.01 s. |
| Full K=1 Slurm GPU replay | Pending. |

No test tolerance, expected numerical value, or quality threshold has been
changed during C4.5.

## Reference quality and performance

The initial same-code artifact is
`$HOME/palmer_scratch/tmp/recovar_em_test_regenerated`. Its ledger records
final merged correlation `0.9985789461317439`, RELION FSC-AUC
`0.995855447698812`, and elapsed time `1184.1040608882904 s` across 13
completed numbered iterations plus final-all-data.

The final C4.5 Slurm run must record the job ID, accelerator, completed
iterations, final correlation, FSC-AUC if available, wall/ledger time, peak
memory, and any comparison caveat. A runtime difference is not attributed to
the refactor unless hardware and execution context are comparable.

## Immediate next actions

1. Run the final focused/static suite and `pixi run test-em-fast-guard`.
2. Submit the prescribed full K=1 replay to the Slurm GPU partition and check
   final correlation and timing against the recorded reference.
3. Mark C4.5 accepted in this ledger and the inventory only if those checks
   pass; otherwise revise or revert the responsible slice.
4. Begin C5 only after C4.5 is accepted.
