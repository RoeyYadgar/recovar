# C6 Dense/Global Scoring Inventory

Status: in progress; inventory and baseline gate

Started: 2026-09-21

Baseline: accepted C5 implementation checkpoint `80737456`, including the
post-acceptance exact-double correction `5c7e7c74` and documentation checkpoint
`2d64b501`

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

Progress: [`dense_single_volume_refactor_progress.md`](dense_single_volume_refactor_progress.md)

Use `$REPO_ROOT` for the checkout and `$HOME` for user-owned artifacts. Do not
record user-specific absolute paths.

## Scope and invariants

C6 replaces dense/global host plumbing and compiled-kernel interfaces without
changing score formulas, score/prior ordering, candidate identity/order,
normalization or reduction order, M-step/noise arithmetic, dtype/layout,
Fourier windows, random-number use, JAX dispatch, or diagnostic artifact
schemas. Gaussian and first-iteration normalized-CC/hard-winner routes remain
explicit. Exact RELION and algebraic primitives remain distinct unless an
existing exact test establishes equivalence.

The primary implementation surfaces are:

- `em_engine.py`: the typed dense host entry, preprocessing and block planning,
  two-pass orchestration, M-step/noise accumulation, finalization, diagnostics,
  and compatibility facade;
- `dense_big_jit.py`: the compiled per-rotation-bucket score, normalization,
  M-step, adjoint, and noise boundary;
- `dense_em_types.py`: the public typed dense request/result contracts;
- `helpers/preprocessing.py`, `helpers/projection.py`, and
  `helpers/scoring.py`: existing numerical owners retained by C6;
- `diagnostics/local_capture.py`: the existing dense/local diagnostic parsing
  and serialization owner.

## Initial structural scorecard

The C6 core below comprises those seven modules.

| Measure | Package | C6 core |
|---|---:|---:|
| Production Python files | 76 | 7 |
| Production lines | 69,205 | 6,060 |
| Nonblank production lines | 63,748 | 5,504 |
| Functions/methods | 1,238 | 107 |
| Classes | 154 | 24 |
| Functions with >=20 args | 33 | 4 |
| Calls with >=20 args | 64 | 3 |
| Largest function span | 5,500 | 1,538 |

Primary dense hotspots:

| Surface | Arguments/fields | Span | Initial role |
|---|---:|---:|---|
| `run_dense_em` | 1 typed request | 1,538 | Canonical host implementation; all dense stages still share one body. |
| `run_dense_bucket_big_jit` | 44 arguments | 217 | Canonical compiled bucket implementation. |
| `_DenseBigJitBatchRunner` | 27 fields; 15-argument `run` | 100 | Host adapter that expands its fields into the 44-argument JIT call. |
| `run_em` | 39 arguments | 55 | One-way external compatibility facade. |
| `compute_e_step_weights` | 10 arguments | 199 | Public materialized-posterior compatibility/reference route. |

Two of the four C6-core long signatures and one long call belong to shared
local diagnostic serialization rather than dense production execution. They
are counted for a conservative subsystem ratchet but are not C6 algorithm
targets unless dense diagnostic migration touches them.

## Route and compatibility classification

| Surface | Current consumer | Classification / C6 action |
|---|---|---|
| `run_dense_em(DenseEMRequest)` | Iteration loop, K-class orchestration, and adaptive oversampling | Canonical typed host entry. Retain and simplify in place. |
| `run_em(...)` | Scripts, initial-model callers, and direct tests | External compatibility facade. Keep one-way request construction and legacy tuple serialization; no production caller may use it. |
| `make_dense_em_request(...)` | Compatibility facade and typed-request tests | Host compatibility constructor. Keep outside numerical loops. |
| `run_dense_bucket_big_jit(...)` | `_DenseBigJitBatchRunner` and focused tests | Canonical compiled bucket body. Replace its flat 44-argument interface with grouped array/state and static-policy contracts without changing its result tree or static specialization axes. |
| `_DenseBigJitBatchRunner` | `run_dense_em` only | Thin host adapter target. It may assemble block-specific inputs but must not own runtime environment or diagnostic context. |
| non-big-JIT branch inside `run_dense_em` | Dense diagnostic and component-capture routes | Required explicit fallback/reference route. Preserve its arithmetic and observation points; do not silently merge it with big-JIT. |
| `compute_e_step_weights(...)` | Public export and focused tests only | Materialized-posterior compatibility/reference route, not normal production execution. Keep explicitly separate unless a stable typed posterior-result contract can replace it without hiding its memory semantics. |
| `DenseEMResult.to_legacy_tuple(...)` | `run_em` only | Allowed outward serialization at the compatibility boundary. Typed production callers consume named fields directly. |

Repository routing inspection found no in-package production call to `run_em`
and no in-package consumer of a flag-dependent dense tuple. All normal callers
use `run_dense_em` and `DenseEMResult`.

## Numerical and diagnostic boundaries

- First-iteration normalized CC deliberately ignores pose priors while keeping
  candidate and padding masks. Its hard-winner M-step is a separate explicit
  route from Gaussian posterior normalization.
- Gaussian pass 1 uses streaming block logsumexp; pass 2 recomputes scores and
  performs soft or hard M-step accumulation. The existing merge/reduction
  order is frozen.
- Dense noise accumulation and RELION half-volume BPref accumulation are
  optional products of pass 2. Their exact/algebraic layout choices remain
  visible policy, not inferred diagnostics.
- Dense debug environment parsing and per-call eligibility now resolve through
  `diagnostics/local_capture.py::DenseDiagnosticsPlan`; `em_engine.py` keeps
  only the explicit observation calls. The plan includes dense noise, CC,
  per-pose, and noise-split routes, while the capture writers retain their
  existing artifact schemas and observation order.
- Runtime configuration remains host-owned. It must not become a JAX PyTree or
  implicit global read inside numerical kernels.

## Planned implementation slices

1. Freeze dense result PyTrees, Gaussian and normalized-CC/hard-winner outputs,
   compiled specialization topology, peak memory, and warm timing.
2. Define grouped compiled bucket arrays/state and a hashable static policy;
   migrate `_DenseBigJitBatchRunner` and focused tests in the same commit, with
   no legacy internal expansion path.
3. Separate immutable dense run planning and per-batch preprocessing only where
   the extracted value owns independently testable validation or derived
   metadata. Delete superseded local representation in each slice.
4. Isolate pass-1 normalization, pass-2 M-step/noise accumulation, and result
   finalization along the existing numerical boundaries. Keep big-JIT and
   diagnostic fallback execution explicit.
5. Move the remaining dense environment parsing and serialization policy
   behind diagnostics while preserving capture schemas and observation order.
6. Delete superseded adapters and dead/shadow dense routes, then run structural,
   focused CPU, and paired GPU quality/performance gates.

## C6 ratchets and exit gates

- The C6 core is net smaller in production lines, functions, long signatures,
  and long calls. Package totals do not increase from this checkpoint.
- Any new class must replace a flat representation or prevent an invalid state
  in the same commit; extraction alone is not sufficient.
- Normal production execution remains typed request to typed result and never
  expands through `run_em` or reparses a variable tuple.
- `run_dense_bucket_big_jit` has stable grouped inputs and a stable named
  result. Static policy remains explicit so JAX specialization topology does
  not change accidentally.
- Gaussian and first-iteration normalized-CC/hard-winner outputs match their
  frozen focused baselines exactly on CPU and within the established GPU
  arithmetic contract.
- Dense HLO/module topology, compile count, peak memory, and warm timing remain
  within the accepted gate. Quality gates use FSC/FSC-AUC; map correlation is
  diagnostic only.
- Diagnostic-null execution remains observationally identical and does not
  disable the canonical compiled path.

## Completed implementation slices

| Commit | Outcome |
|---|---|
| `c379a512` | Replaced the 44-argument compiled bucket signature and 27-field host adapter with batch-lifetime data, block/pass state, and hashable static-policy groups. Removed the superseded window-constant record and both long production calls. |
| `0b6e2bb9` | Moved dense debug-route resolution from `_DenseDebugOptions` and direct environment reads in `em_engine.py` into `DenseDiagnosticsPlan`; added a focused route-resolution regression test. The numerical path remains unchanged. |
| `77eec9c5` | Moved class-prior application into `DenseScoreConstraints`, added a bound per-batch block view, and removed the dense engine's nested score-constraint closure plus redundant runner fields. The existing fallback and big-JIT routes use the same bound block inputs. |
| `702df30b` | Grouped the exact-local score diagnostic tensors into `LocalScoreDumpCapture`; the writer now accepts eight routing arguments instead of 26 while preserving the existing payload and artifact schema. |
| `3c758d5c` | Shared the dense pass-1/pass-2 score-constraint application helper; both passes retain the same constraint ordering and JAX call while removing duplicated host unpacking. |
| `0829d908` | Removed the score-mode parameter and duplicate branch from the RELION image-correction helper; Gaussian and normalized-CC routes already used identical correction factors. |

The current structural measurement after this slice is 76 production files,
69,196 lines, 63,727 nonblank lines, 1,237 functions, 157 classes, 31 long
signatures, 62 long calls, and a 5,500-line package maximum. The seven-file
C6 core is 6,029 lines, 5,464 nonblank lines, 104 functions, 27 classes, two
long signatures, one long call, and a 1,481-line dense-engine maximum. The
core and package line totals are below the C6 baseline; the added helper is a
small host-only consolidation and does not add a runtime context layer.

## Initial validation ledger

| Scope | Result |
|---|---|
| Required parity ancestry at documentation checkpoint `2d64b501` | All five required parity-fix ancestors present. |
| Dense engine/JIT prescribed baseline | 94 passed with two expected custom-CUDA-only skips in `47.67 s`. |
| Dense result-tree contract | Commit `aea8e282`: Gaussian and normalized-CC pass-1 variants have the same named 12-leaf shapes/dtypes; focused file passes 24 tests. |
| Compiled-kernel baseline | Slurm GPU job `61006843` completed `0:0` on one A100. Four static variants compiled once each; result trees, StableHLO fingerprints, output hashes, warm timing, and peak device memory are frozen below. |
| Grouped compiled-kernel validation | Slurm GPU job `61006876` completed `0:0` on the same A100 UUID. Every result leaf hash, StableHLO operation sequence/count, compile count, and peak-memory byte count matches the baseline. Sub-millisecond warm samples remain diagnostic pending the paired end-to-end gate. |
| Dense diagnostics-plan slice | `tests/unit/test_dense_runtime_options.py tests/unit/test_dense_big_jit.py -q`: 60 passed. `git diff --check` passed. |
| Dense score-constraint slice | Focused constraint/refinement cases: 16 passed; dense big-JIT/runtime options: 60 passed; half-spectrum, sampling/M-step, and normalized-CC suite: 72 passed with two expected custom-CUDA skips; CPU fast guard: 16 passed in `51.26 s`. |
| Local score diagnostic capture slice | `tests/unit/test_local_diagnostics.py tests/unit/test_refine_relion_mode.py -k 'local_score_debug_dump_operands_stay_on_big_jit or local_score_debug_dump_records_attempted_pose_metadata' -q`: 5 passed including operand dtype/schema, routing, and artifact-schema cases; structural score is 31 package long signatures and 2 C6-core long signatures. |
| Five-iteration GPU replay at `77eec9c5` | Slurm job `61007337` completed `0:0` in `00:06:39` on one V100 after five numbered iterations. Final merged RECOVAR-vs-RELION FSC-AUC was `0.9970972713`, correlation was `0.999995`, ledger elapsed time was `280.133 s`, and peak batch RSS was `12,691,752 KiB`. Artifact: `$HOME/palmer_scratch/tmp/c6_dense_replay_77eec9c5_20260921`. Jobs `61007076` and `61007106` failed before EM because the first runtime root was unavailable and the second lacked the FFTW module; neither is algorithm evidence. |
| C5 full replay reference | Accepted same-allocation job `60844838`; use the artifact and quality/performance table in the active progress ledger. |
| Exact-double post-C5 replay | Job `61006782` completed `0:0` through five iterations with final merged RELION FSC-AUC `0.9946195501`; artifact `$HOME/palmer_scratch/tmp/double_bpref_fix_5c7e7c74_20260921`. |

The host `pixi` wrapper remained alive after pytest reported completion and was
interrupted only after the final result was printed. No pytest process was
interrupted before completion.

## Compiled-kernel GPU baseline

Artifact:
`$HOME/palmer_scratch/tmp/c6_dense_baseline_aea8e282_20260921`.
The deterministic fixture uses 64-pixel images/volumes, 8 images, 64
rotations, 8 translations, and seed `20260921`. Each row was measured after
clearing JAX caches; cache size changed from zero to one in every row.

| Variant | Compile | Warm median | Peak device memory | StableHLO ops | Operation-sequence SHA-256 |
|---|---:|---:|---:|---:|---|
| Gaussian pass 1 | `0.8684 s` | `0.651 ms` | `18.33 MiB` | 120 | `5de5a89b6b78f67a` |
| Gaussian M-step | `0.4259 s` | `0.906 ms` | `26.23 MiB` | 179 | `5f8e03e9c8f3e640` |
| Normalized-CC pass 1 | `0.2451 s` | `0.635 ms` | `26.23 MiB` | 118 | `f3160ebc376ebc6e` |
| Normalized-CC hard M-step | `0.3297 s` | `1.130 ms` | `26.23 MiB` | 243 | `be1722a4b104d0c` |

The JSON artifact retains complete operation and text hashes plus every output
shape, dtype, and byte hash. C6 acceptance compares the same fixture on the
same GPU model; compile time is contextual, while compile count, operation
topology, output hashes, peak memory, and warmed distributions are gates.

The grouped-boundary candidate artifact is
`$HOME/palmer_scratch/tmp/c6_dense_candidate_c379a512_20260921`. The baseline
and candidate used the same A100 UUID. The Gaussian M-step warm median differed
by `0.285 ms` in seven sub-millisecond samples even though the compiled HLO and
peak memory are identical; this is not used as a speed claim. The required
paired full replay remains the material runtime gate.
