# Dense Single-Volume Refactor Audit Through C6

Date: 2026-09-22

Scope: completed phases C1--C6 and the proposed C7--C10 sequence

Original structural baseline: `1e2f229b`

Audited checkout: `24246441`

Plan: [`dense_single_volume_refactor_plan.md`](dense_single_volume_refactor_plan.md)

Progress: [`dense_single_volume_refactor_progress.md`](dense_single_volume_refactor_progress.md)

Earlier C1--C4 audit:
[`dense_single_volume_refactor_audit_2026-09-11.md`](dense_single_volume_refactor_audit_2026-09-11.md)

## Executive finding

The concern that the refactor has not compressed the codebase is correct.
From the original baseline to the accepted C6 checkout, production source grew
by 1,206 lines, 20 files, 114 functions, and 82 classes. Long signatures and
long calls improved, but the dominant functions and files remain almost as
large as they were initially.

The work has nevertheless produced valuable infrastructure:

- canonical typed local, dense, and sparse host boundaries;
- grouped JAX inputs with stable compilation and result trees;
- centralized runtime resolution and substantially separated diagnostic
  persistence;
- extensive focused tests and paired GPU quality/performance evidence;
- one-way compatibility facades rather than typed-to-legacy-to-typed normal
  execution.

Those gains make deeper deletion safer, but they should not be mistaken for
completion of the readability goal. Most completed phases optimized interface
safety and numerical confidence. Only C4.5 produced material package-wide
compression. C5 and C6 satisfied ratchets that were too permissive: reductions
of 13 and effectively zero package lines were enough to declare phases
complete even though their largest algorithm bodies barely changed.

The current C7--C10 sequence should therefore pause. The next phase should be
a deletion-first consolidation pass with explicit code-removal and hotspot
targets. Proceeding directly to a `RefinementSession` and more extracted stage
objects risks repeating C1--C4: more names and files around the same amount of
algorithmic code.

## Audit question and method

The measurable hypothesis was:

> C1--C6 improved interface safety and validation, but only C4.5 materially
> reduced structural complexity; C5 and C6 met local ratchets without
> simplifying their dominant algorithm bodies.

The audit compares:

- the original baseline `1e2f229b`;
- the C4 checkpoint and C4.5 consolidation checkpoint recorded in the prior
  audit and C4.5 inventory;
- the C5 and C6 accepted scorecards;
- a fresh AST and physical-line inventory of the current package;
- baseline-to-current Git additions, deletions, file status, and test growth;
- current long functions, long calls, hidden environment reads, compatibility
  surfaces, and code concentration.

Line count is treated as a warning and budget, not as a standalone design
goal. A grouped request can improve an interface without removing code, while
an opaque context can improve an argument count and make comprehension worse.
The conclusions below use both structural measurements and ownership/control-
flow inspection.

## Baseline-to-current structural result

Scope: `recovar/em/dense_single_volume/**/*.py`.

| Measure | Original baseline | Current C6 | Delta | Evaluation |
|---|---:|---:|---:|---|
| Production Python files | 56 | 76 | +20 (`+35.7%`) | Regression |
| Production lines | 67,999 | 69,205 | +1,206 (`+1.77%`) | Regression |
| Nonblank production lines | 63,119 | 63,734 | +615 (`+0.97%`) | Regression |
| Functions/methods | 1,124 | 1,238 | +114 (`+10.1%`) | Regression |
| Classes | 75 | 157 | +82 (`+109.3%`) | Major regression |
| Functions with at least 20 arguments | 38 | 31 | -7 (`-18.4%`) | Improvement |
| Calls with at least 20 arguments | 81 | 62 | -19 (`-23.5%`) | Improvement, still high |
| Largest function span | 5,564 | 5,500 | -64 (`-1.2%`) | Essentially unchanged |

The production diff contains 9,106 inserted and 7,900 deleted lines: 17,006
lines of gross churn for a net increase of 1,206. It adds 20 production files,
modifies 25, and records four renames; no baseline production file disappears
as a deletion in the aggregate diff.

Unit tests grew by 4,819 inserted and 1,410 deleted lines, a useful net safety
investment of 3,409 lines. Test growth is a refactoring asset, but it does not
offset production complexity.

## Where the code remains concentrated

Adding modules did not substantially distribute the main complexity:

- the two largest files contain 28,342 lines, or `41.0%` of the package;
- the five largest files contain 40,141 lines, or `58.0%`;
- the ten largest files contain 48,079 lines, or `69.5%`.

| File | Current lines | Primary unresolved issue |
|---|---:|---|
| `helpers/sparse_pass2_bucketed.py` | 18,153 | Two multi-thousand-line algorithm bodies plus preparation and diagnostic plumbing |
| `iteration_loop.py` | 10,189 | Controller, route assembly, replay, updates, convergence, and finalization remain interleaved |
| `local_em_engine.py` | 4,279 | Canonical typed entry, but its implementation remains a 2,955-line function |
| `k_class.py` | 3,832 | Extensive `engine_kwargs` copying, mutation, and route-specific assembly |
| `helpers/significance.py` | 3,688 | Large K=1/K-class implementations and diagnostic branches |
| `em_engine.py` | 2,356 | Dense entry is typed, but the main body remains 1,444 lines |

The six largest function bodies alone occupy 19,224 lines (`27.8%` of all
production source):

| Function | Span | Arguments | Finding |
|---|---:|---:|---|
| `_run_relion_iteration_loop` | 5,500 | 9 | Main readability objective remains unsolved |
| `compute_pass2_stats_sparse_bucketed` | 3,931 | 2 | Typed boundary improved; algorithm body did not materially shrink |
| `compute_k_class_pass2_stats_sparse_fused` | 3,532 | 2 | Same result for joint K-class sparse execution |
| `run_local_em` | 2,955 | 1 | Typed request hides argument width but not control-flow size |
| `_compute_k_class_significance_batched` | 1,862 | 35 | Both long and very large |
| `run_dense_em` | 1,444 | 1 | Smaller, but stage ownership is still mostly one lifecycle |

Argument grouping did not eliminate the worst raw boundaries. Current
examples include 61-, 58-, 57-, and 53-argument functions and calls with up to
89 arguments. The aggregate count improved, but the remaining cases are still
large enough to dominate local readability.

## Phase-by-phase verdict

| Phase | What was genuinely achieved | What remains | Audit verdict |
|---|---|---|---|
| C1 data contracts | Stable named requests/results and less tuple-position coupling | Initially additive and class-heavy | Valuable foundation, paid down later by C4.5 |
| C2 settings boundary | Runtime resolution is printable, testable, and host-owned | Settings hierarchy remains large; `helpers/relion_fine_scoring.py` still reads `current_environment()` below the intended owner boundary | Retain architecture, finish boundary and reduce types/accessors |
| C3 diagnostics | Persistence and configuration have clearer ownership | Diagnostics still occupy 3,493 lines across nine modules; capture branches remain interleaved at observation sites | Separation succeeded; simplification did not |
| C4 exact-local | Local JIT boundary fell from 98 flat arguments to grouped inputs with stable HLO | `run_local_em` remains 2,955 lines and several one-lifecycle planning/cache records remain | JAX boundary success, host readability partial |
| C4.5 consolidation | Removed typed/legacy round trips, reverse tuple machinery, shims, and redundant records | Did not address the largest algorithm bodies | Clear success and the model future work should follow |
| C5 sparse pass 2 | Stable two-argument typed contracts, import-cycle repair, named K=1/K-class results, strong performance evidence | Package fell only 13 lines; C5 core fell 21 lines; largest sparse function fell seven lines and the file is still 18,153 lines | Contract migration complete; “split and simplify” incomplete |
| C6 dense/global | 44-argument JIT call grouped, diagnostics/settings ownership improved, dense result/HLO gates frozen | Package ended at its C6 starting line count; core fell only 22 lines; `run_dense_em` remains 1,444 lines and the planned stage separation is incomplete | Kernel interface complete; host simplification incomplete |

It is more accurate to describe C5 and C6 as validated interface migrations
than completed simplification phases.

## What should be kept

The audit does not recommend reverting the refactor. These boundaries have
demonstrated value and should remain unless a concrete consolidation replaces
them:

- `RuntimeConfiguration`, `AlgorithmSettings`, and host-side
  `ExecutionSettings` as the resolved configuration boundary;
- canonical typed local, dense, and sparse request/result entry points;
- grouped static/array JAX kernel inputs whose HLO and compile topology have
  been validated;
- explicit separation of Gaussian, normalized-CC/hard-winner, exact RELION,
  algebraic, K=1, and joint K-class numerical variants;
- diagnostics configuration and persistence outside the numerical kernels;
- focused equivalence tests, frozen boundary fixtures, and paired GPU gates;
- one-way legacy facades outside hot paths while compatibility is still an
  explicit requirement.

These are safety rails for deletion, not reasons to keep every planning or
capture record introduced along the way.

## What should be reconsidered

### Completion criteria

“Net smaller” is too weak when a phase starts around 25,000 lines. A 21-line
reduction can pass while leaving every dominant function intact. Future gates
need an absolute reduction target and hotspot targets, not only a negative
delta.

### Abstraction accounting

Every new class currently counts as acceptable if it gives a concept a name.
That led to a net increase of 82 classes. A class should instead earn its cost
by crossing a real module/JAX boundary, owning independently testable
validation, or preventing an invalid state. One-producer/one-consumer records
that simply rename local variables should be merged or inlined.

### Compatibility timing

The plan schedules final compatibility and duplicate cleanup in C9. That is
too late. A replacement and the representation it supersedes must be handled
in the same phase. Any retained facade needs an identified external consumer,
owner, deletion condition, and expiry.

### Controller decomposition

The proposed C7 `RefinementSession` can help by owning stable host-side data,
but it can also become a context bag or a 5,500-line god class. It should own
only lifecycle state and orchestration. Numerical stages should remain
functional JAX-style operations with explicit small inputs. Merely turning
free functions into methods or moving blocks out of the main function does
not count as simplification.

### Global line target

The current plan's final 67,999-line target only removes the net code added by
this refactor. Reaching it would restore the original size, not demonstrate
that the original codebase became meaningfully cleaner. A stronger target is
needed if compression is an explicit objective.

## Recommended plan reset before C7

Do not start the current C7 implementation yet. Replace C7--C9 with the
following deletion-first sequence; keep C10 as the final acceptance phase.

### R1. Retention and deletion inventory

- Enumerate every compatibility facade, reference backend, diagnostic route,
  numerical variant, settings/plan class, and raw `engine_kwargs` field.
- Record production and external consumers, numerical distinction, owner,
  test, and deletion condition.
- Classify each item as canonical, required variant, public compatibility,
  active diagnostic, merge candidate, or dead.
- Obtain an explicit decision about whether undocumented external Python APIs
  must remain source-compatible. Without that decision, compatibility removal
  will remain artificially constrained.

This phase is documentation and tests only; it adds no production abstraction.

### R2. Delete stale diagnostics and compatibility residue

- Remove expired parity experiments, shadow implementations, unused capture
  schemas, one-way facades with no supported consumer, and obsolete test-hook
  ownership.
- Move any retained diagnostic operand assembly completely out of hot loops
  when it is observational and disabled.
- Finish the C2 boundary by removing lower-level environment lookup from
  `relion_fine_scoring.py`.
- Delete or merge one-lifecycle plan/cache/result records that do not cross a
  meaningful boundary.

No replacement abstraction is introduced unless it deletes more production
surface in the same commit.

### R3. Simplify sparse/significance implementations

- Start with duplicated preparation, diagnostic, and route branches rather
  than splitting the two canonical numerical bodies into more files.
- Share code only after exact focused tests establish identical ordering,
  dtype, and reduction semantics.
- Retain distinct K=1 and joint K-class kernels where their normalization is
  genuinely different.
- Eliminate large diagnostic call signatures and raw option forwarding before
  introducing further sparse data classes.

### R4. Simplify the controller around explicit state transitions

- Define one host-side refinement state and one immutable data/settings owner.
- Make the main iteration loop a stage sequence: initialize, derive plan,
  apply replay, score halves, update reconstruction/noise/prior/corrections,
  evaluate convergence, record history, finalize.
- Each extraction must remove duplicated assembly, branches, or state
  mutation. A move-only extraction does not satisfy the phase gate.
- Keep final-all-data, restart, and replay as explicit routes rather than flags
  checked throughout ordinary iteration logic.

### R5. Remove `engine_kwargs` and clarify K-class/replay routing

- Decide route policy once before invocation.
- Pass typed class views into dense/local/sparse engines.
- Delete copying, mutation, and default reconstruction of `engine_kwargs`.
- Separate STAR reading, validation, override construction, and state
  application without duplicating iteration control.

### R6. Final residue and acceptance

- Audit every remaining variant and compatibility surface.
- Run the existing focused, CPU, HLO, quality, and paired performance gates.
- Keep C10's K=1 and K-class acceptance requirements.

## Proposed quantitative gates

These are audit recommendations, not yet accepted changes to the plan.

### Per-slice rules

- Production deletions must exceed additions. A temporary exception requires
  an owner and deletion commit in the same phase.
- Production file and class counts may not increase.
- An extraction must reduce duplicated assembly, branch count, or state
  mutation; reducing only the caller's argument count is insufficient.
- Tests and documentation are measured separately and never offset production
  growth.

### First consolidation milestone

Before controller decomposition begins:

- production source at or below 66,000 lines (`-3,205` from current);
- classes at or below 145;
- functions with at least 20 arguments at or below 25;
- calls with at least 20 arguments at or below 45;
- no lower numerical/helper module reads the runtime environment snapshot;
- every retained compatibility facade and diagnostic route has a documented
  consumer and expiry/rationale.

### Final structural objective

A credible final objective is 63,000--65,000 production lines, subject to the
retention inventory and explicit compatibility decision. That is a reduction
of roughly 6--9% from the current checkout and 4--7% from the original
baseline. It is large enough to demonstrate actual simplification without
encouraging opaque code compression.

Hotspot objectives should be tracked independently:

- `_run_relion_iteration_loop` below 1,000 lines, as already planned;
- no other production function above 2,000 lines;
- `run_dense_em` below 900 lines;
- `run_local_em` below 1,500 lines;
- the top five files below 35,000 combined lines;
- no compatibility conversion or diagnostic persistence in a numerical hot
  loop.

If the R1 inventory cannot identify a safe path to the first 3,205-line
reduction, the plan should stop and explicitly choose between preserving every
historical/diagnostic surface and achieving substantial simplification. It
should not continue adding abstractions while assuming later phases will pay
the debt.

## Decision

The codebase is still cleanable, but the current plan is not sufficient to
deliver that outcome. Its strongest result is a safer set of boundaries and a
much better equivalence/performance test harness. Its weakest result is that
those boundaries were accepted without requiring material deletion from the
dominant implementations.

C1--C6 should remain as validated groundwork. Their “complete” labels should
be interpreted as interface/validation milestones, not evidence that the
overall readability goal has been met. C7 should remain paused until the user
accepts, rejects, or modifies the deletion-first reset proposed above.

## Validation scope

This is a docs-only structural audit. No production code, numerical path, JAX
tree, or runtime behavior changed. The current checkout contains all required
parity ancestors. No CPU/GPU algorithm test is required for the audit itself;
the accepted test and GPU evidence cited above remains recorded in the phase
inventories.
