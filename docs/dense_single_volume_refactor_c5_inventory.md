# C5 Sparse Pass-2 Inventory

Status: complete and accepted at implementation checkpoint `80737456`

Started: 2026-09-15

Completed: 2026-09-16

Baseline: accepted C4.5 implementation checkpoint `d6bf42da` (documentation
checkpoint `76803b0e`)

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
  lines, functions, long signatures, and long calls. The class count may grow
  by exactly two for the typed sparse data/settings/result contracts that
  replace raw dictionaries, a variable K=1 tuple, and duplicate K-class
  records; deleting those semantic
  boundaries merely to satisfy a raw class-count ratchet is not an accepted
  simplification.
- Package totals do not exceed the accepted C4.5 checkpoint for production
  lines, functions, long signatures, or long calls. The same explicit
  two-class contract exception applies package-wide.
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

## Completed implementation slices

| Commits | Outcome |
|---|---|
| `34a39516`, `fe22289a`, `54a97989` | Moved significant-support encoding, RELION fine-scoring primitives, and significance thresholds to neutral lower-level owners, eliminating the significance-to-sparse implementation edge. |
| `670eb1e2` | Centralized dataset-index translation shared by sparse and local routes. |
| `7d5d6c2f`, `d42672b5` | Added the stable K=1 sparse data/settings/result boundary and migrated all K-class K=1 production callers away from long calls and tuple parsing. |
| `5ce5f3e5`, `a8fee2ab`, `c1bd3d60` | Added the distinct fused K-class result while sharing the demonstrated data/settings contract and centralized request construction. Joint class/pose normalization remains a separate implementation. |
| `2ef9f0e3`, `36921c7b`, `e01b7144` | Removed the compact-capture alias, retargeted tests to the diagnostic owner, and reduced invasive routing to an explicit diagnostics policy function. |
| `fc7fbf8b`, `211e79de`, `2fcffeca` | Deleted a dead residual helper, collapsed sparse option forwarding, and removed superseded internal adapters. The historical external facade remains one-way. |
| `8b069c2a`, `cd194e0e`, `80737456` | Cached successful optional RELION binding resolution in the per-image preparation loop while preserving retries after transient loader failures. |

The two canonical algorithm bodies remain large because scoring, posterior,
and M-step arithmetic share one compiled lifecycle. C5 did not split them into
files with no independent owner or test boundary. Their external host
interfaces are now two arguments each, and normal production flow no longer
round-trips through the historical 61- or 46-argument forms.

## Final structural scorecard

The final C5 core is the original sparse/significance/oversampling/diagnostic
surface after deleting the compatibility capture module and adding the five
neutral owners created by C5.

| Measure | C4.5 package | C5 package | Delta | Initial C5 core | Final C5 core | Core delta |
|---|---:|---:|---:|---:|---:|---:|
| Production Python files | 72 | 76 | +4 | 5 | 9 | +4 |
| Production lines | 69,212 | 69,199 | -13 | 24,749 | 24,728 | -21 |
| Nonblank production lines | 63,751 | 63,742 | -9 | 23,177 | 23,158 | -19 |
| Functions/methods | 1,239 | 1,238 | -1 | 365 | 364 | -1 |
| Classes | 152 | 154 | +2 | 9 | 11 | +2 |
| Functions with >=20 args | 35 | 33 | -2 | 16 | 14 | -2 |
| Calls with >=20 args | 65 | 64 | -1 | 25 | 24 | -1 |
| Largest function span | 5,500 | 5,500 | 0 | 3,938 | 3,931 | -7 |

The file increase is the cost of replacing cyclic/mixed ownership with neutral
modules; it did not increase total code. C5 added `SparsePass2Data`,
`SparsePass2Settings`, and the stable K=1 `SparsePass2Result`, while deleting
the invasive-diagnostics policy class; the fused K-class result replaces an
existing implementation-owned record. That leaves a net increase of two
classes for three real sparse contracts.

## C5 validation and performance audit

| Scope | Result |
|---|---|
| Sparse parity and sampling after the binding fix | 43 passed in `57.81 s`. |
| Adaptive oversampling, sparse parity, and RELION worker scale with FFTW loaded | 98 passed with six pre-existing complex-cast warnings in `121.93 s`. |
| Sparse and K-class semantics | 70 passed in `62.27 s`; the K-class merge/joint slice also passed 109 tests during migration. |
| Full sparse performance suite | 175 passed with two expected GPU-only skips in `125.52 s`. |
| Diagnostics structure | 4 passed. |
| Runtime settings and BPref stop policy | 36 passed. |
| Dependency, capture, fine-score, and weighted-average ownership | 68 passed; read-only JAX cache warnings only. |
| CPU fast guard at final implementation | 16 passed in `49.56 s`. |
| Native binding preparation profile | Slurm job `60844523` completed `0:0` on a V100. With the same external binding pinned in both arms, candidate preparation median was about `0.0712 s` versus control `0.0720 s`; both loaded the native extension. |
| Warm sparse ABBA gate | Slurm job `60844524` completed `0:0` on one V100. Pooled candidate median was `0.17647 s` versus control `0.17978 s` (`-1.84%`); compile times remained within the normal paired spread. |
| Final prescribed paired replay | Slurm job `60844838` completed `0:0` control-first on one A100-PCIE-40GB. Both arms ran 13 numbered iterations plus final-all-data with identical size trajectories and 310-field result schemas. |

Final clean paired replay artifact:
`$HOME/palmer_scratch/tmp/c5_samegpu_final_80737456_vs_d6bf42da_20260916`.

| Measure | C4.5 control | C5 candidate | Candidate delta |
|---|---:|---:|---:|
| Final correlation vs RELION | `0.9983925071` | `0.9983953637` | `+0.0000028566` |
| Final FSC-AUC vs RELION | `0.9948770218` | `0.9948863890` | `+0.0000093672` |
| Ledger elapsed | `985.772 s` | `993.264 s` | `+0.76%` |
| Exact-local EM | `332.928 s` | `328.630 s` | `-1.29%` |
| Process wall | `1047.09 s` | `1038.43 s` | `-0.83%` |
| Transfer to host | `7.414 s` | `7.343 s` | `-0.97%` |
| Peak RSS | `11,114,976 KiB` | `11,053,948 KiB` | `-0.55%` |

The direct candidate/control final merged maps have correlation
`0.9999998900` and relative L2 `0.0004684`. Both result archives contain 310
fields in the same order with identical shapes and dtypes. The ledger wall
increase is inside the 5% gate and is contextualized by lower process wall,
EM time, transfer time, and RSS; no algorithm or performance regression is
present.

The first warm investigation was intentionally not accepted. Jobs `60841550`
and `60841926` exposed a repeatable slowdown, and jobs `60842862` and
`60843417` narrowed it to optional native binding resolution. The apparent
remaining difference was then traced to an ignored native `.so` present only
in the control worktree. Those runs compared different backends and are not
performance evidence. The decisive jobs above pin the same checksum-identical
external binding for both arms. This audit also produced a small robustness
fix: successful lookups are cached, but transient import failures are retried.
