# C5 Sparse Pass-2 Inventory

Status: active

Started: 2026-09-15

Baseline: accepted C4.5 checkpoint `76803b0e`

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

Progress: [`dense_single_volume_refactor_progress.md`](dense_single_volume_refactor_progress.md)

Use `$REPO_ROOT` for the checkout and `$HOME` for user-owned artifacts. Do not
record user-specific absolute paths.

## Scope and invariants

C5 replaces sparse pass-2 host interfaces and ownership without changing score
formulas, posterior support, candidate identity/order, bucket topology,
reduction order, dtype/layout, random-number use, JAX dispatch, or diagnostic
artifact schemas. Exact RELION, algebraic, rectangular, compact-pair, K=1, and
joint K-class routes remain distinct until focused evidence proves otherwise.

The primary implementation surfaces are:

- `helpers/sparse_pass2_bucketed.py`: bucket planning, scoring, posterior,
  M-step, noise/norm accumulation, and interleaved diagnostic capture;
- `helpers/oversampling.py`: historical sparse facade, reference route, and
  bucketed-route selection;
- `helpers/significance.py`: coarse support selection plus fine-scoring CUDA
  imports back from the sparse implementation;
- `k_class.py`: K=1/K-class sparse orchestration and tuple parsing;
- `diagnostics/sparse_capture.py` and `helpers/compact_candidate_capture.py`:
  existing diagnostic owners and one compatibility facade.

## Initial structural scorecard

The C5 core below comprises sparse bucketed, significance, oversampling,
compact-candidate capture, and sparse diagnostic ownership modules.

| Measure | Package | C5 core |
|---|---:|---:|
| Production Python files | 72 | 5 |
| Production lines | 69,212 | 24,749 |
| Nonblank production lines | 63,751 | 23,177 |
| Functions/methods | 1,239 | 365 |
| Classes | 152 | 9 |
| Functions with >=20 args | 35 | 16 |
| Calls with >=20 args | 65 | 25 |
| Largest function span | 5,500 | 3,938 |

Primary sparse hotspots:

| Surface | Arguments | Span | Initial role |
|---|---:|---:|---|
| `compute_pass2_stats_sparse_bucketed` | 61 | 3,938 | Canonical K=1 bucketed implementation behind a legacy facade. |
| `compute_k_class_pass2_stats_sparse_fused` | 46 | 3,530 | Canonical joint-normalized K-class implementation. |
| `_compute_k_class_significance_batched` | 35 | 1,879 | Coarse support selection; outside pass 2 but participates in the import cycle. |
| `_compute_significance_batched` | 28 | 674 | K=1 coarse support selection. |
| `_prepare_bucket_io` | 27 | 535 | Single-bucket input preparation and preprocessing. |
| Sparse diagnostic writers | 19--53 | 100--605 | Persistence mixed into the implementation module. |

## Route and compatibility classification

| Surface | Current consumer | Classification / C5 action |
|---|---|---|
| `compute_pass2_stats_sparse(...)` | Five in-package K-class calls, scripts, and tests | External compatibility facade and route selector. Migrate production callers to a typed canonical entry; retain a one-way facade for scripts/tests. |
| `_compute_pass2_stats_sparse_perimage_reference(...)` | The historical facade and parity tests | Required reference backend, not production canonical. Keep explicitly named and outside normal execution. |
| `compute_pass2_stats_sparse_bucketed(...)` | The historical facade and direct tests | Canonical K=1 algorithm body. Move behind a cohesive request and stable result without a typed-to-legacy-to-tuple round trip. |
| `compute_k_class_pass2_stats_sparse_fused(...)` | K-class orchestration and direct tests | Canonical joint K-class body. Give it a cohesive typed request and retain joint normalization semantics. |
| `SparseKClassPass2FusedResult` | K-class orchestration | Stable result concept, currently coupled to the implementation module; move to the neutral sparse contract owner. |
| K=1 variable result tuples | K-class orchestration and compatibility callers | Internal tuple debt. Canonical result must have fixed named fields; serialize outward only at the historical facade. |
| `common` / `fused_common` dictionaries in K-class routing | Sparse K=1 and fused K-class calls | Raw settings debt. Replace sparse-owned fields with typed request groups; defer general K-class route redesign to C8. |
| `helpers/compact_candidate_capture.py` | Sparse code and tests | Compatibility facade. Migrate sparse production imports to the diagnostics owner, then delete if no external consumer remains. |

## Import-cycle result

The cycle is structural rather than merely syntactic:

1. `sparse_pass2_bucketed` imports `ComplementSignificantSampleIndices` from
   `significance` and support/grid helpers from `oversampling`.
2. `significance` imports seven RELION fine-scoring/CUDA helpers from
   `sparse_pass2_bucketed` inside coarse diagnostic/experimental routes.
3. `oversampling` imports the bucketed K=1 implementation inside its route
   selector.

C5 will move shared support representations and fine-scoring primitives to
neutral lower-level owners. The historical facade may import the canonical
sparse runner, but the canonical runner must not import its facade.

## Planned implementation slices

1. Move shared support and RELION fine-scoring primitives to neutral modules,
   update imports, and add a dependency-layer guard.
2. Define grouped K=1 sparse inputs/settings/outputs and one stable result.
   Mechanically make the existing bucketed body consume that request.
3. Migrate K-class K=1 orchestration from raw dictionaries and tuple parsing to
   the typed sparse boundary; leave the historical facade one-way.
4. Define and migrate the fused K-class request/result boundary without
   changing joint class/pose normalization.
5. Extract diagnostic persistence and stop policy to existing diagnostics
   owners. Keep observation points explicit and preserve schemas.
6. Extract planning or preparation only where the component has an independent
   lifecycle or focused test surface; do not create one-file-per-concept
   scaffolding.
7. Delete superseded adapters, compatibility aliases, and dead/shadow routes;
   then rerun structural, CPU, and paired GPU gates.

## C5 ratchets and exit gates

- The touched sparse/significance subsystem is net smaller in production
  lines, functions/classes, long signatures, and long calls.
- Package totals do not exceed the accepted C4.5 checkpoint: 69,212 lines, 152
  classes, 35 long signatures, and 65 long calls.
- Normal in-package execution uses typed sparse requests/results and never
  expands them into a legacy signature or reparses a variable tuple.
- No import edge from significance or a neutral primitive module points back to
  the canonical sparse implementation.
- K=1 and K-class bucket topology, candidate identity/order, result fields,
  compile count, peak memory, and quality remain within their accepted gates.
- Diagnostic null routes remain observational; invasive stop behavior remains
  owned by diagnostics.

## Initial validation ledger

| Scope | Result |
|---|---|
| K=1 sparse bucketed parity | 24 passed in `87.12 s`. |
| Sparse planning/performance suite | 174 passed, one stale diagnostic-owner assertion failed, and two GPU-only tests skipped in `127.00 s`. |
| Diagnostic-owner correction | The stale assertion was retargeted to `diagnostics.sparse_capture`; focused rerun passed 1/1. |
| C4.5 GPU quality/performance reference | Clean same-allocation job `60569641`; use the artifact recorded in the active progress ledger. |

The host `pixi` wrapper did not exit after pytest reported completion in these
baseline runs and was interrupted after results were printed. No test process
was interrupted before pytest completion.
