# Dense Single-Volume EM Refactor Progress

Last updated: 2026-09-23

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

C4 audit: [`dense_single_volume_refactor_audit_2026-09-11.md`](dense_single_volume_refactor_audit_2026-09-11.md)

C4.5 inventory: [`dense_single_volume_refactor_c45_inventory.md`](dense_single_volume_refactor_c45_inventory.md)

C5 inventory: [`dense_single_volume_refactor_c5_inventory.md`](dense_single_volume_refactor_c5_inventory.md)

C6 inventory: [`dense_single_volume_refactor_c6_inventory.md`](dense_single_volume_refactor_c6_inventory.md)

C1--C6 plan audit:
[`dense_single_volume_refactor_audit_2026-09-22.md`](dense_single_volume_refactor_audit_2026-09-22.md)

R1 retention/deletion inventory:
[`dense_single_volume_refactor_r1_inventory.md`](dense_single_volume_refactor_r1_inventory.md)

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
| C5 Sparse pass 2 | Complete | Accepted at `80737456`: typed sparse boundaries, ownership cleanup, focused/CPU gates, and paired warm/full GPU gates pass. Post-acceptance double-BPref correction: `5c7e7c74`. |
| C6 Dense/global scoring | Complete | Grouped big-JIT boundary, diagnostics plan, score-constraint ownership, local score capture, correction-branch cleanup, and dense pass-2 host-stage extraction are validated. Final five-iteration GPU replay `61011471` reached merged FSC-AUC `0.9970972997` and correlation `0.9999947703` with the expected size trajectory. |
| R1 Retention/deletion inventory | Complete | Compatibility, diagnostic, numerical-variant, type/plan, and all 50 raw K-class `engine_kwargs` fields are classified; five no-op lifecycle event families are the first proven deletion. |
| R2 Delete stale diagnostics and compatibility residue | In progress | Three proven-dead slices are complete: no-op lifecycle events, unused trace-specification scaffolding, and the superseded single-class significance engine. Continue to the 66,000-line R2 gate. |
| R3--R6 Structural simplification and acceptance | Pending | Start only after R2 reaches its deletion gate without changing numerical behavior. |

## Current structural scorecard

Scope: `recovar/em/dense_single_volume/**/*.py`.

| Measure | R1 baseline | Current R2 | R2 delta | Result |
|---|---:|---:|---:|---|
| Production Python files | 76 | 76 | 0 | Pass; no new owner module. |
| Production lines | 69,205 | 68,305 | -900 | In progress toward the R2 gate of at most 66,000. |
| Nonblank production lines | 63,734 | 62,931 | -803 | Pass |
| Functions/methods | 1,238 | 1,213 | -25 | Pass |
| Classes | 157 | 150 | -7 | Pass |
| Functions with >=20 args | 31 | 30 | -1 | Pass |
| Calls with >=20 args | 62 | 61 | -1 | Pass |
| Largest function span | 5,500 | 5,459 | -41 | Improved, but the controller remains the package maximum. |

R2 has reduced production source by 1.30% from the R1 baseline without adding
a production file or class. The five largest files still contain 39,405 lines
(57.7% of the package), so the concentration problem remains and the next
slices must continue deleting inside existing hotspots.

## R2 completed deletion slices

| Commit | Slice | Structural outcome |
|---|---|---|
| `6d77f4ce` | No-op lifecycle events | Removed five event classes, fifteen sink/protocol methods, and four inert controller emission sites. |
| `24cdb6d4` | Unused trace specification | Removed two classes and the environment-name-to-trace mapping that no kernel or engine consumed; retained passive/shadow/invasive routing. |
| `3b41d735` | Superseded single-class significance engine | Deleted the 677-line production-unreachable implementation and migrated its remaining tests to the canonical class-aware engine used by K=1, K-class, scripts, and the initial-model adapter. |

The three slices together remove 900 production lines, 25 functions/methods,
seven classes, one long signature, and one long call. The first slice also
includes repository-required formatting in the controller; the 677-line
significance deletion is direct duplicate-code removal.

## R2 validation ledger

| Scope | Result |
|---|---|
| Diagnostic sink, structure, timing, and runtime routing | 51 passed after lifecycle deletion; 50 passed after trace-spec deletion. |
| Significance/pass-1 focused selection | 67 passed with one expected GPU-only skip. |
| Adaptive oversampling and K-class merge guards | 101 passed with four host-only exclusions; the excluded tests require the unavailable `libfftw3.so.3` RELION binding. A direct full-file attempt reached 38 passes before those same four dependency failures. |
| CPU EM fast guard | 16 passed after the diagnostic slices and again after the significance deletion (`50.80 s` final run). |
| GPU/performance gate | Not run: all three slices delete production-unreachable or no-op host scaffolding, so no numerical/JAX/HLO path changes. |

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

## C5 completed implementation

- Neutral owners now contain significant-support encoding, significance
  thresholds, RELION fine-scoring primitives, and shared dataset indexing.
  `significance` no longer imports the canonical sparse implementation.
- K=1 and fused K-class sparse engines consume the shared `SparsePass2Data`
  and `SparsePass2Settings` contracts and return distinct stable named
  results. Their public canonical boundaries are two arguments each.
- All production K-class callers construct typed requests and consume named
  results. Variable tuple parsing and typed-to-legacy round trips remain only
  at the one-way historical external facade.
- Sparse capture imports point directly to `diagnostics.sparse_capture`; the
  compatibility capture module, dead residual helper, superseded request
  adapters, and redundant option forwarding are deleted.
- K=1 and K-class share only input/settings assembly. Their numerically
  distinct score, posterior, normalization, and M-step implementations remain
  separate.
- Successful optional RELION binding lookups are cached outside the per-image
  preparation loop. Import exceptions are deliberately not cached so a
  transient loader failure cannot pin execution to a different backend.

The C5 implementation is the sequence `34a39516` through `80737456`, with
small commits for each ownership, contract, caller-migration, deletion, and
performance correction. The exact commit ledger is in the C5 inventory.

## C5 validation ledger

| Scope | Result |
|---|---|
| Sparse sampling/parity after the final binding correction | 43 passed in `57.81 s`. |
| Adaptive oversampling, sparse parity, and worker scale with FFTW loaded | 98 passed with six pre-existing complex-cast warnings in `121.93 s`. |
| Sparse and K-class semantic slices | 70 passed; the broader merge/joint slice passed 109 tests during migration. |
| Full sparse performance suite | 175 passed with two expected GPU-only skips. |
| Diagnostics, runtime settings, stop policy, dependency, capture, and fine-score ownership | Focused suites passed (4, 36, and 68 tests respectively). |
| CPU fast guard | 16 passed in `49.56 s`. |
| Same-binding preparation profile | Slurm GPU job `60844523`: candidate median about `0.0712 s`, control about `0.0720 s`; both used the same native extension. |
| Same-binding warm sparse ABBA | Slurm GPU job `60844524`: pooled candidate `0.17647 s`, control `0.17978 s`, candidate delta `-1.84%`; completed `0:0` on one V100. |
| Final prescribed paired replay | Slurm GPU job `60844838` completed `0:0` control-first on one A100-PCIE-40GB; trajectory/schema matched, quality improved, and runtime/memory gates passed. |
| Post-acceptance exact-double BPref correction | Commit `5c7e7c74`: focused corr-image tests 5 passed; explicit sparse float64 test passed; full sparse performance suite 176 passed with two expected GPU-only skips; CPU fast guard 16 passed. Slurm GPU job `61006782` completed `0:0` through five iterations on one V100. |

The initial warm measurements were not accepted because the candidate appeared
slower. Investigation found two separate causes: repeated optional-binding
resolution after C5 changed import timing, and an ignored native RELION `.so`
present only in the control worktree. The production lookup is now cached
safely, and decisive jobs pin the same checksum-identical external binding in
both arms. Earlier jobs `60841550`, `60841926`, `60842862`, and `60843417` are
diagnostic evidence only, not acceptance comparisons.

Commit `5c7e7c74` fixes a latent double-only interface mismatch exposed by the
exact RELION BPref operand configuration: the sparse caller requested a
float64 correction image, but the extracted native-noise helper neither
accepted nor preserved an output dtype. The helper now has an explicit
float32/float64 output contract; its default float32 operation order is
unchanged. The exact reported configuration was replayed for five iterations
in Slurm job `61006782`. It completed with size trajectory
`[46, 46, 72, 70, 70]`, final recovar-vs-RELION merged-map correlation
`0.9999962378`, ledger elapsed time `425.455 s`, and no final-all-data pass as
requested by `--max_iter 5`. The portable artifact is
`$HOME/palmer_scratch/tmp/double_bpref_fix_5c7e7c74_20260921`.

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

The accepted C5 full-replay artifact is
`$HOME/palmer_scratch/tmp/c5_samegpu_final_80737456_vs_d6bf42da_20260916`.
Job `60844838` ran C4.5 control `d6bf42da` followed by C5 candidate `80737456`
on the same A100-PCIE-40GB, with clean detached worktrees, separate caches, and
the same pinned CUDA and RELION bindings:

| Measure | C4.5 control | C5 candidate | Candidate delta |
|---|---:|---:|---:|
| Completed iterations / final-all-data | 13 / yes | 13 / yes | same |
| Final correlation vs RELION | `0.9983925071` | `0.9983953637` | `+0.0000028566` |
| Final FSC-AUC vs RELION | `0.9948770218` | `0.9948863890` | `+0.0000093672` |
| Ledger elapsed | `985.772 s` | `993.264 s` | `+0.76%` |
| Exact-local EM | `332.928 s` | `328.630 s` | `-1.29%` |
| Process wall | `1047.09 s` | `1038.43 s` | `-0.83%` |
| Transfer to host | `7.414 s` | `7.343 s` | `-0.97%` |
| Peak RSS | `11,114,976 KiB` | `11,053,948 KiB` | `-0.55%` |

Both arms followed size trajectory
`[46, 46, 72, 70, 70, 70, 70, 70, 70, 72, 72, 72, 72]`. Their 310-field
result archives have identical key order, shapes, and dtypes. Direct final-map
correlation is `0.9999998900` with relative L2 `0.0004684`. C5 therefore
introduces no material quality, runtime, transfer, or memory regression.

## Immediate next actions

1. Continue R2 with private definitions that have no production consumer;
   verify decorators, aliases, scripts, and tests before each deletion.
2. Audit direct controller dump families against the active parity program;
   delete only expired routes and preserve accepted artifact schemas.
3. Reach at most 66,000 production lines before beginning the R3 sparse and
   significance simplification. Run GPU quality/performance gates only when a
   slice can reach a numerical or compiled path.
