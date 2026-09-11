# C4.5 Compatibility and Consolidation Inventory

Status: active

Started: 2026-09-11

Baseline: C4 implementation checkpoint `6041093d`

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

Audit: [`dense_single_volume_refactor_audit_2026-09-11.md`](dense_single_volume_refactor_audit_2026-09-11.md)

## Classification rules

- **Canonical**: normal in-package production execution enters here.
- **External compatibility**: one-way adapter retained for direct callers of a
  historical module API; it must never be called by normal in-package flow.
- **Independent boundary**: the object owns a separately testable decision,
  crosses a module/JAX boundary, or prevents invalid state.
- **Merge candidate**: the object has the same producer, consumer, and lifetime
  as a neighboring object and may only rename intermediate locals.
- **Dead/test-hook debt**: no production purpose remains, or production imports
  it solely to preserve an old monkeypatch location.

Occurrence counts below are navigation aids, not API-usage counts. They count
textual references in production and unit-test Python at the start of C4.5.

## Engine and controller adapters

| Surface | Initial state | Classification / action | Expiry |
|---|---|---|---|
| `run_local_em(request)` | Typed wrapper expanded into `run_local_em_exact`, then parsed its tuple | Canonical after `a1035d25`; existing algorithm body now lives here and returns `LocalEMResult` directly | Complete |
| `run_local_em_exact(...)` | Historical 58-argument implementation and tuple result | External compatibility; now a one-way request builder over the canonical core. No in-package execution caller may use it | Re-evaluate at C9 after external API decision |
| `LocalEMResult.from_legacy_tuple` | Used by typed production wrapper and tests | No production consumer after `a1035d25`; candidate for deletion when legacy conversion tests are reduced to the one-way facade contract | C4.5 |
| `LocalEMOutputSpec` / `to_legacy_tuple` | Selected historical tuple shapes | Compatibility-only after `a1035d25`; keep only the minimum serializer required by the external facade | C4.5/C9 |
| `run_dense_em(request)` | Typed wrapper expands into `run_em`, then parses its tuple | Additive migration; invert so the typed function owns the existing body and returns `DenseEMResult` | C4.5 dense slice |
| `run_em(...)` | Historical 39-argument canonical implementation | Convert to one-way external compatibility facade after dense inversion | C4.5 dense slice |
| `dense_em_request_from_legacy_kwargs` | Reflects against `run_em` for K-class | Internal compatibility debt; replace with typed K-class request construction | C4.5 K-class slice |
| `run_local_search_iteration(request)` | Typed wrapper expands into 71-argument `_run_local_search_iteration` and returns a variable tuple | Additive migration; move the existing body behind the typed request and return one stable result | C4.5 local-search slice |
| `_run_local_search_iteration(...)` | Historical canonical implementation | Retain only if a documented external/private compatibility requirement remains; otherwise delete after caller/test migration | C4.5 local-search slice |
| `_unpack/_pack_local_search_engine_outputs` | Translate tuple shapes around the local-search body | Internal tuple debt; replace with direct typed results | C4.5 local-search slice |
| `legacy_runner` parameters | 16 production occurrences at inventory time | Test-hook debt. Patch the canonical owner in tests instead of injecting the old implementation | C4.5 |

## K-class bridges

| Surface | Initial state | Classification / action | Expiry |
|---|---|---|---|
| `_local_em_request_from_legacy_kwargs` | Binds a raw dictionary against the old exact-local signature | Internal compatibility debt. Construct typed requests from a typed class view | C4.5 |
| `_run_local_em_typed` | Request wrapper around the bridge above | Fold into the typed K-class execution seam after callers no longer pass raw engine dictionaries | C4.5 |
| `_dense_engine_kwargs_for_class` | Copies and mutates a raw settings dictionary per class | Replace engine-owned fields with a typed dense class view; preserve unrelated sparse/K-class routing separately | C4.5/C8 |
| `_local_engine_kwargs_for_class` | Copies and mutates a raw settings dictionary per class | Replace engine-owned fields with a typed local class view | C4.5 |
| broad `engine_kwargs` plumbing | 78 production textual occurrences | Classify by owner. Remove local/dense engine fields in C4.5; leave sparse and route redesign for C5/C8 with explicit ownership | C4.5/C8 |

## Runtime/settings surfaces

| Surface | Initial state | Classification / action | Expiry |
|---|---|---|---|
| `RuntimeConfiguration` | Captured immutable host snapshot | Canonical; retain |
| runtime `ExecutionSettings` | Owns first-iteration, raw-image, dense planning, and local cache settings | Independent host boundary; retain, then verify lower components receive owned settings explicitly |
| `LocalExecutionSettings` | Exact-local batching/cache request group | Independent per-call engine boundary; retain unless dense/local request consolidation proves a clearer shared owner |
| `ExecutionSettings = LocalExecutionSettings` | Compatibility alias in `local_em_types.py` | External/test compatibility only; remove tests/imports using the ambiguous alias, then delete unless an external API requirement is documented | C4.5 |
| `current_environment()` reads in lower host modules | Captured mapping rather than direct process access | Hidden dependency. Move values into the nearest existing settings/diagnostic request when consolidating that component | C4.5--C8 |

## Diagnostics compatibility surfaces

| Surface | Production dependency at inventory | Classification / action | Expiry |
|---|---|---|---|
| `local_debug.py` | `local_em_engine.py` imports a compatibility symbol; tests import old owner | Test-hook/re-export debt; import the diagnostics owner directly and retarget tests | C4.5 |
| `debug_dumps.py` | `iteration_loop.py` retains an import/re-export for tests | Test-hook/re-export debt; retarget tests and delete if no supported external consumer | C4.5 |
| `parity_dump.py` | Compatibility facade used mainly by parity tests | Potential external compatibility; inventory script imports before deletion | C4.5/C9 |
| `helpers/compact_candidate_capture.py` | Compatibility facade used by sparse code/tests | Keep until C5 migrates sparse consumers; it must not grow | C5 |
| typed diagnostic sinks/events | Separation and null-sink behavior are validated | Audit producer/consumer/lifetime. Merge only identical lifecycle records; preserve stable artifact schemas | C4.5 |

## Exact-local plan and route records

These are candidates, not assumed deletions. Reference counts include class
definitions, imports, annotations, constructions, and tests.

| Type | Production/test occurrences | Initial classification |
|---|---:|---|
| `LocalEMModePlan` | 12 / 2 | Independent decision: validates and names numerical routes; retain unless merged without obscuring validation |
| `LocalEMInputPlan` | 5 / 2 | Merge candidate: single producer/consumer validation handoff |
| `LocalEMGeometryPlan` | 12 / 2 | Shared by reconstruction, Fourier, and batch planning; likely independent boundary |
| `LocalEMReconstructionPlan` | 6 / 0 | Merge candidate with geometry/Fourier planning |
| `LocalProjectionPlan` | 4 / 0 | Merge candidate inside Fourier plan |
| `LocalEMFourierPlan` | 8 / 2 | Crosses setup, batching, and compiled-input preparation; likely independent boundary |
| `LocalMicrobatchRoute` | 4 / 0 | Merge candidate with `LocalMicrobatchPlan` |
| `LocalMicrobatchPlan` | 4 / 3 | Independent measured memory decision, but route record may fold into it |
| `LocalBucketSummary` | 5 / 2 | Observability-only view; consider computing only for logging/profile consumers |
| `LocalBucketPlan` | 3 / 2 | Single producer/consumer; merge candidate unless topology validation benefits from the boundary |
| `LocalCacheRouteConstraints` | 4 / 2 | Input-only record; merge candidate into the route planner call |
| `LocalCacheRoute` | 3 / 0 | Named cache decision consumed by multiple setup branches; likely retain |
| `LocalRelionProjectionCachePlan` | 3 / 0 | Single lifecycle; merge candidate with cache state |
| `LocalRelionProjectionCacheStats` | 3 / 2 | Mutable measurement owner; retain or merge with diagnostic/profile accumulator |
| `LocalDiagnosticRequest` | 6 / 2 | Explicit diagnostic selection; retain if it prevents repeated parsing |
| `LocalDiagnosticsSession` | 4 / 7 | Owns enabled diagnostic lifetime; retain, while removing old-module imports |

## Initial scorecard and ratchet

| Measure | C4 checkpoint | After local inversion `a1035d25` | C4.5 gate |
|---|---:|---:|---:|
| Production files | 75 | 75 | <=74 |
| Production lines | 71,509 | 71,471 | <=70,509 |
| Nonblank production lines | 65,975 | 65,930 | decrease |
| Functions/methods | 1,254 | 1,254 | decrease |
| Classes | 165 | 165 | <=155 |
| Functions with >=20 args | 37 | 37 | <=35 |
| Calls with >=20 args | 75 | 74 | <=65 |

The local facade still accounts for one long signature because compatibility
tests call it directly. It is outside normal production flow. The remaining
line/class/call reductions must come from dense/local-search inversion and
deleting redundant compatibility/planning/diagnostic structures, not from
compressing the algorithm body.
