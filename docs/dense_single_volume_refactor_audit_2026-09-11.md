# Dense Single-Volume Refactor Audit Through C4

Date: 2026-09-11

Scope: completed phases C1--C4

Baseline: `1e2f229b3e0e8edaec029d2b604937f692148578`

C4 implementation checkpoint: `6041093d`

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

Progress: [`dense_single_volume_refactor_progress.md`](dense_single_volume_refactor_progress.md)

## Executive finding

The completed work has preserved numerical behavior and produced two valuable
boundaries: runtime/environment resolution is centralized, and diagnostic
persistence is largely outside the numerical modules. The exact-local JIT
boundary also improved from 98 flat arguments to eight grouped arguments.

It has not yet delivered the expected package-wide simplification. Production
Python grew by 3,510 lines (`+5.16%`), 19 files, 130 functions, and 90 classes.
The number of functions with at least 15 or 20 arguments fell by only one in
each category. Calls with at least 10 arguments increased by seven. Every
completed phase increased production line count.

The main cause is additive migration: typed requests, results, plans, sinks,
and compatibility adapters were added while the old functions, tuple
protocols, raw keyword dictionaries, and orchestration paths remained
authoritative. In several important paths the new interface expands back into
the old interface, so a reader must understand both representations.

The refactor therefore pauses before C5. A new C4.5 consolidation phase must
make typed entry points authoritative, delete internal legacy round trips, and
reduce structural debt before another subsystem is split. This is a correction
to the execution strategy, not a rejection of the preserved numerical and
configuration work.

## Audit method

The static inventory covers `recovar/em/dense_single_volume/**/*.py` only.
Tests and documentation are reported separately; they do not offset production
growth. Counts were collected at the following logical implementation
checkpoints:

| Checkpoint | Commit | Meaning |
|---|---|---|
| Baseline | `1e2f229b` | Original plan inventory |
| C1 | `204384c7` | Local and dense typed-result caller migrations complete |
| C2 | `0f7b0337` | Runtime/environment boundary complete |
| C3 | `13b97bfa` | Diagnostics extraction plus restart lifecycle correction |
| C4 | `6041093d` | Exact-local implementation complete; later commits are documentation |

“Lines” are physical Python source lines and “nonblank” excludes empty lines.
Function and call counts come from the Python AST. An argument-bearing call
counts positional and keyword arguments. These measurements are indicators,
not standalone design goals: hiding unrelated state in one opaque context
would improve the table while making the code worse.

## Structural scorecard

| Measure | Baseline | C1 | C2 | C3 | C4 | C4 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Production Python files | 56 | 58 | 62 | 69 | 75 | `+19` |
| Production lines | 67,999 | 68,721 | 69,699 | 70,398 | 71,509 | `+3,510` |
| Nonblank production lines | 63,119 | 63,712 | 64,513 | 65,049 | 65,975 | `+2,856` |
| Functions/methods | 1,124 | 1,137 | 1,174 | 1,226 | 1,254 | `+130` |
| Classes | 75 | 100 | 122 | 140 | 165 | `+90` |
| Functions with >=10 args | 125 | 125 | 125 | 125 | 124 | `-1` |
| Functions with >=15 args | 59 | 59 | 59 | 59 | 58 | `-1` |
| Functions with >=20 args | 38 | 38 | 38 | 38 | 37 | `-1` |
| Calls with >=10 args | 341 | 337 | 341 | 342 | 348 | `+7` |
| Calls with >=15 args | 133 | 131 | 131 | 131 | 125 | `-8` |
| Calls with >=20 args | 81 | 80 | 78 | 78 | 75 | `-6` |

Per-phase production growth was:

| Phase | Lines added net | Functions added net | Classes added net |
|---|---:|---:|---:|
| C1 | `+722` | `+13` | `+25` |
| C2 | `+978` | `+37` | `+22` |
| C3 | `+699` | `+52` | `+18` |
| C4 | `+1,111` | `+28` | `+25` |

Across baseline to the current documented C4 checkout, the scoped diff is:

| Area | Added | Removed | Net |
|---|---:|---:|---:|
| Production package | 9,222 | 5,712 | `+3,510` |
| Unit tests | 3,474 | 502 | `+2,972` |
| Progress document | 2,691 | 0 | `+2,691` |

The test growth is useful safety infrastructure. It does not demonstrate that
the production design became simpler. The progress log itself has also become
too large for a status document and should be split into a concise active
ledger plus a historical archive.

## Hotspot result

| Boundary | Baseline lines | C4 lines | Change |
|---|---:|---:|---:|
| `_run_relion_iteration_loop` | 5,564 | 5,620 | `+56` |
| `compute_pass2_stats_sparse_bucketed` | 4,155 | 4,155 | `0` |
| `compute_k_class_pass2_stats_sparse_fused` | 3,704 | 3,704 | `0` |
| `run_local_em_exact` | 3,348 | 3,045 | `-303` |
| `_compute_k_class_significance_batched` | 2,005 | 2,005 | `0` |
| `run_em` | 1,587 | 1,530 | `-57` |
| `run_local_bucket_big_jit` | 843 | 856 | `+13` |
| `_score_half_local` | 697 | 755 | `+58` |
| `_score_half_dense` | 617 | 624 | `+7` |

The six main hotspot files together fell from 42,894 to 42,232 lines, a
reduction of only 662 lines (`1.54%`). C4 made the clearest local improvement:
`local_em_engine.py` fell by 856 lines and `run_local_em_exact` by 303 lines.
That local reduction was outweighed by newly extracted modules and adapters.

## Phase-by-phase evaluation

### C1 — typed contracts

What worked:

- Stable `LocalEMResult` and `DenseEMResult` names made result consumption less
  dependent on optional tuple positions.
- In-package call sites began expressing inputs as cohesive groups.
- Focused and GPU comparisons established a strong equivalence baseline for
  later deletion.

What did not close:

- The typed functions are wrappers around the legacy engines rather than the
  engines themselves.
- `LocalEMOutputSpec` and `DenseEMOutputSpec` exist mainly to reconstruct and
  parse legacy tuple shapes.
- Long historical signatures remain the internal source of truth.
- The phase added 722 production lines and 25 classes without changing any
  long-function count.

Verdict: useful migration scaffolding, but incomplete until the adapter
direction is inverted and in-package tuple compatibility is removed.

### C2 — runtime and environment boundaries

What worked:

- Process environment access was confined to `runtime_options.py` and
  `diagnostics/config.py`; algorithm modules consume a captured mapping rather
  than calling `os.environ` directly.
- All named settings were classified, and major algorithm behavior now has a
  typed resolved configuration.
- The runtime snapshot improves reproducibility and makes resolved settings
  printable and testable.

What did not close:

- The boundary added multiple nested setting types and compatibility accessors
  while engine-local settings types remained separate.
- Some lower host modules still pull values from `current_environment()`.
  This is no longer ambient process access, but it still hides a dependency
  that should become an explicit setting at the owning host boundary.
- Argument metrics did not improve during the phase.

Verdict: real architectural value and worth retaining. C4.5 should reduce
accessor/type duplication and pass the already-resolved setting to components
that currently query the snapshot themselves.

### C3 — diagnostics extraction

What worked:

- Artifact schemas, serialization, lifecycle handling, and effect
  classification have clear ownership in a diagnostics package.
- Passive, shadow, and invasive behaviors are explicitly distinguished.
- The old `local_debug.py`, `parity_dump.py`, `debug_dumps.py`, and
  `compact_candidate_capture.py` implementations shrank from 2,550 lines to
  60 lines of compatibility re-exports.
- Null-diagnostics HLO, compile behavior, output quality, timing, and memory
  passed the C3 gate.

What did not close:

- The new diagnostics package is 3,452 lines, 962 more than those four former
  debug modules before accounting for removals elsewhere.
- The four compatibility re-export modules remain, and production modules
  still import some of them to preserve old ownership or test-hook locations.
- Numerous specialized events and sink methods add concepts a reader must
  learn; their consumer counts and distinct lifecycles have not been audited.
- The phase added 699 production lines and 18 classes.

Verdict: separation from algorithm code is valuable, but “moved” must not be
confused with “simplified.” Remove internal shims and merge event/sink types
whose lifecycle and payload ownership are identical.

### C4 — exact-local engine

What worked:

- The compiled local bucket boundary fell from 98 flat arguments to eight
  grouped arguments while preserving JAX array leaves, HLO, and compile count.
- Planning, array setup, cache policy, projection cache, and diagnostic
  lifetime now have named owners.
- `local_em_engine.py` and `run_local_em_exact` became smaller.
- Paired GPU validation found no material quality, runtime, or memory
  regression.

What did not close:

- `run_local_em(request)` expands the request into a roughly 60-keyword call to
  `run_local_em_exact(...)`, then converts the returned tuple back into
  `LocalEMResult`.
- `local_search_iteration.py` explicitly injects the legacy runner. K-class
  first binds a raw `engine_kwargs` dictionary against the old signature,
  creates a request, and then injects the old runner again.
- The local-search typed wrapper likewise expands into the 71-argument legacy
  function.
- C4 introduced seven component files totaling 2,246 lines and 27 classes.
  Several plans and route/result records have only one production lifecycle and
  may be merge or inline candidates.
- Global long-argument counts improved by only one function; calls with at
  least 10 arguments increased by six during C4.

Verdict: the JIT boundary change succeeded, but the host API migration stopped
halfway. C4 cannot be considered a readability success until the typed core is
authoritative and redundant host-side representations are deleted.

## Consolidation inventory

These are audit targets, not pre-approved deletions. Each item must first have
its production and external consumers checked, and each retained compatibility
surface needs an owner and removal rationale.

### Priority 0 — invert additive migrations

1. Make `run_local_em(LocalEMRequest) -> LocalEMResult` the canonical local
   implementation. Keep a thin legacy facade only at a documented package/API
   edge if external compatibility is required.
2. Make the typed dense entry point canonical by the same rule.
3. Make the typed local-search entry point canonical; remove its
   `legacy_runner` expansion and variable tuple packing from production flow.
4. Remove in-package `legacy_runner` injection, tuple round trips, and
   signature-reflection builders.
5. Replace K-class `engine_kwargs` bags with typed class views before beginning
   a broader K-class refactor.

### Priority 1 — collapse redundant structure

Audit and either justify, merge, or inline:

- `LocalEMOutputSpec` / `DenseEMOutputSpec` and their legacy tuple conversion
  methods;
- the `ExecutionSettings = LocalExecutionSettings` alias;
- mode, geometry, reconstruction, Fourier, projection, microbatch, bucket,
  cache-route, and projection-cache plans that are constructed and consumed in
  one uninterrupted host stage;
- private imports and re-exports retained only so tests can monkeypatch an old
  owner module;
- the four legacy debug-module re-export shims;
- diagnostic event or sink classes with one producer and one identical
  consumer lifecycle.

The objective is not to eliminate small types indiscriminately. A type stays
when it names an independently testable decision, crosses a meaningful module
boundary, or prevents invalid state. It should be folded back when it merely
renames local variables, duplicates a parent request, or exists only to bridge
old and new APIs.

### Priority 2 — improve the documentation surface

- Keep the active progress board and latest accepted evidence concise.
- Move the long per-slice chronology to an archive without changing its
  historical content.
- Record one structural scorecard per accepted slice instead of narrative that
  repeats the implementation diff.

## Revised success criteria

Line count is not the goal, but sustained line growth is a warning when the
stated goal is simpler code. Future phases use both semantic and quantitative
gates:

- production, test, and documentation size are reported separately;
- a phase cannot pass solely because code was split into more files or wrapped
  in more types;
- new compatibility scaffolding must be removed in the same phase or carry an
  explicit owner, consumer, removal condition, and short expiry;
- every phase must delete or replace its superseded representation before the
  next subsystem begins;
- C4.5 must be net-negative in production lines, classes, long functions, and
  long calls, and must remove the typed-to-legacy-to-typed production route;
- by the end of C8, overall production lines must be at or below the 67,999-line
  baseline unless a documented user-approved exception explains concrete new
  capability; tests and comments do not offset that budget;
- final acceptance still depends on readability review, numerical equivalence,
  JAX/HLO stability, and performance. No metric target authorizes compressed,
  opaque, or numerically changed code.

The exact C4.5 numeric exit thresholds and the revised later-phase ratchets are
defined in the main plan so there is one authoritative execution contract.
