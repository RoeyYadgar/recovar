# R1 Retention and Deletion Inventory

Date: 2026-09-23

Status: complete; first three R2 deletion slices landed

Baseline checkout: `20df1bcf`

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

Audit:
[`dense_single_volume_refactor_audit_2026-09-22.md`](dense_single_volume_refactor_audit_2026-09-22.md)

Use `$REPO_ROOT` for the checkout and `$HOME` for user-owned artifacts. Do not
record user-specific absolute paths.

## Purpose

R1 converts the audit's broad deletion targets into an ownership inventory.
It introduces no production abstraction. Each retained surface needs a real
consumer, a numerical or lifecycle distinction, and a deletion condition.

The classifications are:

- **canonical**: normal production implementation or stable contract;
- **required variant**: numerically or operationally distinct behavior with an
  active policy and test;
- **compatibility**: old callable/result surface outside normal production;
- **active diagnostic**: current parity evidence or artifact owner;
- **merge candidate**: valid behavior whose representation does not justify a
  separate class/module/lifecycle;
- **dead**: invoked or represented, but no implementation consumes its value or
  performs an effect.

## Structural baseline

| Measure | R1 baseline |
|---|---:|
| Production files | 76 |
| Production lines | 69,205 |
| Nonblank production lines | 63,734 |
| Functions/methods | 1,238 |
| Classes | 157 |
| Functions with at least 20 arguments | 31 |
| Calls with at least 20 arguments | 62 |
| Largest function | 5,500 lines |

The two largest files contain 41.0% of production source and the five largest
contain 58.0%. R1 therefore prioritizes deletion inside existing hotspots over
moving code into additional modules.

## Compatibility and reference surfaces

| Surface | Actual consumers | Classification | R2 disposition |
|---|---|---|---|
| `run_dense_em(DenseEMRequest)` | Iteration loop, K-class orchestration, adaptive paths | Canonical | Keep |
| `run_em(...)` | Four repository scripts and direct compatibility/reference tests; no normal package caller | Compatibility with repository script consumers | Retain until scripts migrate; then decide external API support |
| `DenseEMResult.to_legacy_tuple(...)` | `run_em` only | Compatibility support | Delete with `run_em`, not independently |
| `run_local_em(LocalEMRequest)` | Local search and K-class production | Canonical | Keep |
| `run_local_em_exact(...)` | Direct tests only; no package or repository-script caller | Compatibility, external consumer unknown | Strong removal candidate after explicit external API decision |
| `LocalEMResult.to_legacy_tuple(...)` | `run_local_em_exact` only | Compatibility support | Delete with `run_local_em_exact` |
| `compute_pass2_stats_sparse_bucketed(data, settings)` | K=1/K-class production and focused tests | Canonical | Keep |
| `compute_k_class_pass2_stats_sparse_fused(data, settings)` | Joint K-class production and focused tests | Required variant | Keep while joint normalization remains distinct |
| `compute_pass2_stats_sparse(...)` | Two parity/capture scripts and direct tests; no normal package caller | Compatibility plus reference-route selector | Migrate scripts; retain only if external API support is required |
| `SparsePass2Result.to_legacy_tuple(...)` | `compute_pass2_stats_sparse` only | Compatibility support | Delete with sparse facade |
| `_compute_pass2_stats_sparse_perimage_reference(...)` | Compatibility facade and parity tests | Reference backend | Move to a test/reference owner or delete after frozen equivalence fixtures replace live comparison |
| `compute_e_step_weights(...)` | Public/reference tests; no normal production caller found | Materialized-posterior reference API | Retain until consumer/API decision; keep visibly outside production path |

External compatibility is the main unresolved policy question. Repository
evidence supports removing `run_local_em_exact` first and migrating the sparse
and dense script callers. It cannot prove that downstream, unversioned Python
users do not import these functions directly.

## Diagnostic ownership

The package contains 3,493 lines in nine `diagnostics` modules, plus diagnostic
routing and observation code in the numerical/controller modules. Static
inspection finds 257 complete `RECOVAR_*` environment names. The current
heuristic classifier labels 88 algorithm, 67 tuning, 88 passive diagnostic,
and 14 invasive. These counts expose another problem: token-based
classification mislabels supporting capture fields and several experimental
algorithm switches, so classification alone is not an ownership registry.

| Family | Primary owner | Current role | Classification | Deletion condition |
|---|---|---|---|---|
| Iteration lifecycle sink/events | `diagnostics/events.py`, `diagnostics/sinks.py` | `IterationStarted` triggers parity timing; five other event methods are no-ops in every sink | One active event plus dead scaffolding | Delete the five no-op events/methods/call sites immediately |
| Parity/timing dumps | `diagnostics/parity.py` | Accepted per-iteration and stage evidence | Active diagnostic | Retain until strict trajectory work no longer depends on its schema |
| Controller intermediate/noise dumps | `diagnostics/controller_capture.py` | Iteration/state and noise debugging | Active diagnostic | Re-audit after controller first-divergence closure |
| Dense/local score, CC, per-pose, and noise captures | `diagnostics/local_capture.py` | Focused arithmetic and operand replay | Active diagnostic | Retain active schemas; remove routes with no current parity program consumer |
| Local diagnostic session | `local_diagnostics.py` | Parses and owns local capture eligibility/lifetime | Merge candidate | Merge with capture plan if producer/consumer remains one lifecycle |
| Sparse pass-2 and BPref capture | `diagnostics/sparse_capture.py` | Current K=1/K=4 causal evidence, atomic shard validation/finalization | Active diagnostic | Retain while parity program cites these artifacts; expire individual schemas separately |
| Significance capture/stop policy | `diagnostics/significance_capture.py` | Coarse-support evidence and explicit invasive stop | Active diagnostic | Retain while support parity remains open |
| Projector, tau2, premask, K-class, BPref boundary dumps | `iteration_loop.py`, `mean_helpers.py` | Direct environment checks and payload assembly outside diagnostic owners | Merge/deletion candidates | Move retained persistence behind diagnostics or delete expired routes |
| Pass-1/pass-2 top-two debug logs | `k_class.py`, `helpers/sparse_pass2_bucketed.py` | Targeted first-divergence logging | Active but narrow | Delete after the associated parity discrepancy is closed and captured by fixtures |
| Device-signature and shadow reductions | Sparse/local engines plus capture modules | Checked alternative arithmetic for parity investigation | Active diagnostic shadow | Delete each shadow after its production route is accepted and the parity board no longer cites it |

### First proven dead slice

`HalfScored`, `MstepAccumulated`, `MapsUpdated`, `ConvergenceUpdated`, and
`IterationFinished` are constructed conditionally in the main loop, but both
`NullDiagnostics` and `ParityDiagnostics` discard them. The protocol methods
also return `None` and no other sink exists. They add five event classes, ten
no-op method implementations, protocol surface, imports, tests, and five
controller branches without observing or persisting anything.

`IterationStarted` is different: `ParityDiagnostics.iteration_started` starts
the parity timer. It stays until timing ownership is redesigned. Removing only
the five no-op event families was the first R2 slice (`6d77f4ce`).

Two further dead surfaces became visible immediately after that deletion:

- `TraceKind`/`TraceSpec` classified environment names, but no numerical
  kernel, engine, or diagnostic capture consulted the result. The only
  consumer printed the names in a startup log. Commit `24cdb6d4` deletes that
  unused future-facing abstraction while retaining the real effect routes.
- `_compute_significance_batched` was a 677-line single-class predecessor to
  `_compute_k_class_significance_batched`. It had no production or script
  caller; only tests invoked it. Commit `3b41d735` migrates those tests to the
  class-aware engine used by production K=1 and deletes the duplicate.

## Numerical and execution variants

| Variant family | Distinction | Classification | Consolidation rule |
|---|---|---|---|
| Dense Gaussian versus normalized CC/hard winner | Different score and posterior semantics | Required variants | Keep explicit |
| Dense big-JIT versus component/fallback execution | Compiled production route versus diagnostic observation points | Canonical plus active diagnostic/reference | Do not merge until the diagnostic route can observe canonical outputs |
| Local big-JIT versus split/component execution | Memory/diagnostic topology differs | Required execution variant for now | Delete only after equivalent memory and capture behavior is demonstrated |
| Exact RELION versus algebraic scoring | Floating-point operation order differs | Required variants | Keep named and separately tested |
| Sparse bucketed versus per-image reference | Production implementation versus slow reference | Canonical plus reference | Prefer frozen fixtures over retaining a full duplicate live implementation |
| Sparse K=1 versus fused K-class | Per-image normalization versus joint class/pose normalization | Required variants | Share only preparation proven identical |
| Rectangular active rows versus compact pairs | Different sparse layout/performance topology | Required while both routes are selected | Remove legacy route only after same-binding quality/HLO/performance evidence |
| Native half-volume versus RELION x-half BPref | Different accumulator layout and parity purpose | Required variants | Keep policy explicit |
| Fused atomic, per-particle launch, and sequential translation reduction | Device accumulation order and parity behavior differ | Required/experimental mix | Retain accepted production route; expire diagnostic shadows individually |
| Dense, sparse, fused, and local K-class routing | Dataset/support/search-dependent execution | Required routing | Decide once before engine invocation; delete repeated route reconstruction |
| RELION replay, restart, final-all-data, and autonomous state | Different ownership transitions | Required controller variants | Isolate as explicit stage routes, not interleaved flags |

The inventory does not authorize merging variants based on algebra alone.
Score order, candidate order, dtype, reductions, HLO, quality, and performance
must establish equivalence first.

## Settings, plans, and state records

| Group | Examples | Classification | Action |
|---|---|---|---|
| Resolved runtime boundary | `RuntimeConfiguration`, `AlgorithmSettings`, `ExecutionSettings` | Canonical | Keep; remove lower-level environment reads |
| Engine request/results | `DenseEMRequest/Result`, `LocalEMRequest/Result`, `SparsePass2Data/Settings/Result` | Canonical | Keep while they remain the direct production boundary |
| JAX grouped inputs/policy/results | Dense bucket records and local big-JIT records | Canonical compiled boundary | Keep unless a smaller tree preserves HLO and compile topology |
| Controller state/results | `RefinementState`, `HalfScoreResult`, per-update named results | Canonical or R4 inputs | Keep provisionally; merge only with the R4 state-transition design |
| Dense/local sub-settings | Search, scoring, projection, correction, posterior, reconstruction, requested-output groups | Valid semantic grouping, but often one constructor and one consumer | Audit pairwise; merge groups that never vary or cross a boundary independently |
| Local planning/cache records | `LocalEMModePlan`, `LocalEMInputPlan`, `LocalEMGeometryPlan`, `LocalMicrobatchPlan`, cache route/plan/stats | Merge candidates | Retain independently validated decisions; inline single-lifecycle wrappers |
| Diagnostic request/capture records | Dense/local capture requests and sessions | Active diagnostic or merge candidate | Keep payloads that prevent invalid schema; merge routing-only wrappers |
| Lifecycle event records | Six event classes | One active, five dead | Delete dead five in first R2 slice |
| Replay input/result records | Replay/projector/half-input/override state | Cross-stage contracts | Revisit during R5, not before replay state ownership is explicit |

Reference count alone does not decide a type's fate. Low-reference JAX results
can be essential, while a frequently referenced forwarding record can still be
redundant. Each removal must inspect lifetime and validation responsibility.

## Raw K-class `engine_kwargs` inventory

`k_class.py` reads, copies, mutates, or reconstructs 50 distinct raw fields.
They are grouped below by intended owner. R5 must assign every field to an
existing typed boundary or delete it; it must not replace one 50-key dictionary
with one 50-field context object.

| Intended owner | Fields |
|---|---|
| Search/candidate priors | `rotation_log_prior`, `translation_log_prior`, `class_rotation_log_prior`, `coarse_translation_log_prior`, `class_local_rotation_log_prior`, `rotation_translation_mask`, `class_rotation_translation_mask`, `translation_prior_centers`, `coarse_rotation_ids`, `coarse_healpix_order`, `translation_phase_source` |
| Geometry/batching | `current_size`, `reconstruction_current_size`, `image_batch_size`, `rotation_block_size`, `projection_padding_factor`, `reconstruction_padding_factor`, `half_spectrum_scoring`, `square_window`, `do_gridding_correction`, `coarse_relion_projector_texture_interp` |
| Corrections/data ownership | `image_corrections`, `scale_corrections`, `group_ids`, `scale_correction_group_count`, `scale_correction_data_vs_prior`, `image_pre_shifts` |
| Scoring/precision | `score_with_masked_images`, `use_float64_scoring`, `use_float64_projections`, `relion_exact_fine_gaussian`, `relion_firstiter_score_mode`, `relion_firstiter_winner_take_all`, `source_faithful_spectrum_norm`, `adaptive_fraction` |
| Projection/M-step/reconstruction | `relion_projector_half`, `relion_projector_r_max`, `relion_half_volume_mstep`, `mstep_relion_x_half`, `relion_fine_mstep_prune`, `reconstruct_significant_only`, `reconstruction_probability_threshold`, `preserve_bpref_particle_order` |
| Routing/output/diagnostics | `sparse_pass2`, `return_profile`, `return_half_volume_accumulators`, `return_class_best`, `return_class_second`, `debug_iteration`, `bpref_device_signature_active` |

This inventory contains all 50 keys found by AST inspection at the R1
baseline. Some keys are popped into route-local dictionaries and later added
back under the same name; those transformations are part of the debt, not new
fields.

## Ordered deletion queue

1. **Complete:** remove the five no-op lifecycle event families and their
   controller calls.
2. **Complete:** remove unused trace-specification scaffolding and the
   production-unreachable single-class significance engine.
3. Audit direct controller dump families against the current parity program;
   delete expired routes and move only retained persistence to diagnostics.
4. Merge/in-line the smallest one-lifecycle local cache/planning records where
   focused tests already cover the derived decision.
5. Migrate repository scripts from `run_em` and
   `compute_pass2_stats_sparse`; decide whether undocumented external imports
   remain supported.
6. If external source compatibility is not required, delete
   `run_local_em_exact` first because it has no package or repository-script
   consumer, then delete its tuple serializer and migrate direct tests.
7. Replace live per-image sparse reference comparisons with frozen focused
   fixtures, then remove or relocate the reference implementation.
8. Start the large sparse/significance and controller reductions only after
   the residue above is removed.

## R1 exit result

R1 covers the requested surface families, lists all 50 K-class raw option
keys, records the compatibility uncertainty, and identifies a deletion whose
deadness is demonstrated by every implementation of the sink protocol.

R2 currently stands at 68,305 production lines, down 900 from this inventory's
baseline, with no numerical path, JAX tree, output, artifact schema, or active
diagnostic effect changed. The next action is continued proven-dead residue
removal and the controller dump audit.
