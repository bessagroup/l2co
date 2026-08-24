# State transfer is an interlingua, not a pairwise table

What one optimizer can usefully hand another genuinely depends on both:
CMA-ES's covariance means something to Sep-CMA-ES and nothing to a GA
with a scalar mutation width. The obvious encoding is a
`(outgoing, incoming)` table. This rejects that.

**Decision.** Each optimizer gets one *reader* that projects its own
state onto a small canonical bundle and one *writer* that folds a bundle
into its own state — `2N` adapters, not `N²`. Three reasons, and it
matters which:

- **Maintainability.** The registry holds 59 optimizers. `2N` is 118
  hand-written adapters; `N²` is 3481.
- **Compile time and code size.** A pairwise dispatch traces and
  compiles `N²` branches even though only one executes per tick.
  `map_unroll`'s own docstring already advises reducing branch fan-out
  on GPU.
- **Redundancy.** Most of those `N²` cells would be computed from the
  same handful of statistics anyway, so the table would encode the same
  few projections repeatedly.

**This is explicitly not a runtime argument, and an earlier draft got
that wrong.** The claim that `vmap` forces every branch of a
data-dependent `lax.switch` to execute is true in general, but both
relevant call paths already avoid it: rl2co's `map_unroll` /
`map_unroll_sharded` have run only the selected branch since rl2co
PR #10/#13, and `batch_run_sequential_` does the same for evaluation
(`UpdateClass.sequential_realizations`). Per-tick cost does not scale
with branch count on those paths. The `2N` decision stands on the three
reasons above; recording the false one would have made a future reader
trust a measurement that does not exist.

**What the bundle carries: scale only.** A single positive scalar
`sigma` — an RMS per-coordinate standard deviation for a sampler, an RMS
step length for a stepper — plus the location it belongs to. Over the
registry 27 optimizers hold an explicit `std` and 20 a diagonal second
moment, so scale is where nearly all the available signal is; and on a
menu mixing families the *initial* scales differ by three orders of
magnitude (`rprop` at `learning_rate=1e-3` against an ES at
`std_init=1.0`), which is signal that is currently discarded at every
switch.

Shape is deliberately absent, and would be **diagonal plus rank-1** if
added. Two independent arguments converge on that. Statistically, a
weighted covariance estimated from `popsize ≈ 10` samples against `d` up
to 40 has rank at most `K`, so a full `(d, d)` metric is not estimable
from anything in the menu. Empirically, every full-covariance provider
in the task pool — `cmaes`, `iamalgamfull`, `maes`, `asebo`, `xnes` —
was dropped from schedule candidacy for NaN coverage at high ambient
dimension (`l2co_experiments/docs/plans/headroom-dataset.md`), and the
shape-bearing schedules that survive are exactly the diagonal and
diag+rank-1 ones. `crfmnes` is the registry's only rank-1 provider: at
l2co's `(1, d)` parameter shape `adafactor` does not factorize, so
`v_row`/`v_col` are dummy `(1,)`.

**Writes rescale; they do not overwrite.** The governing principle is
that **scale goes stale and shape does not.** An optimizer that has not
run for many generations holds a `std` that is wrong about the current
search, but its relative per-coordinate structure is still informative.
So a per-coordinate quantity is scaled so its RMS matches the incoming
`sigma`, leaving the structure intact: `step_sizes * (sigma / rms)` for
Rprop, `nu * (rms_own / sigma)²` for the adam family. Writing `sigma`
flat into `step_sizes` would destroy exactly the per-coordinate ratchet
that makes Rprop work. This needs no blending coefficient.

**`conf` has three levels but only one behaviour.** A reader reports
`CONF_EXACT` (read off a field that *is* the scale), `CONF_ESTIMATED`
(inferred from samples, or from a shape field whose transform is not
recognised), or `CONF_ABSENT`. Only `CONF_ABSENT` changes what a writer
does — it skips. The exact/estimated distinction is carried because a
later blend rule would need it and re-deriving it in every reader would
be worse, but it is not acted on today. A degenerate scale (zero, NaN)
is downgraded to `CONF_ABSENT`: PSO's velocity is zero until a
generation runs, and writing zero would freeze the receiving optimizer.

**Roles are asymmetric, deliberately.** L-BFGS's last accepted step is
recoverable from `diff_params_memory`, but after `init` its state is all
zeros with `learning_rate = 1.0` and `value = inf` and the first step
length comes from a capped inverse gradient norm the line search
overwrites — so it is a provider and not a receiver. DE holds neither
`std` nor `velocity`, so its scale is the population spread: readable,
but the population is the caller's handshake to set, so it is also
provider-only. RMSProp and Adagrad are the converse — a preconditioner
with no stored step cannot report a scale, but can be rescaled against a
unit-gradient reference. Each of these no-ops is a place a plausible
"fix" would be wrong, so each has a test.

**Reader and writer are bound at construction, not looked up per call.**
Both halves need values the state does not retain: the learning rate
(`scale_by_learning_rate` carries an `EmptyState`), Rprop's
`min_step_size` / `max_step_size` clip bounds, and the ravel function an
evosax algorithm uses to flatten a solution pytree — evosax stores
`mean` raveled. A name-keyed lookup at call time could not supply any of
them. What stays name-keyed is the small `TRANSFER_OVERRIDES` table of
departures from the family default, so that `UpdateClass` and `SubOpt`
bind the *same* implementations from the *same* source — the pattern
`docs/adr/0009` established for the handshake policy.

**Rejected alternatives.**

- *A pairwise table.* See above.
- *An optimizer-agnostic empirical sigma* from `‖x* - x*_prev‖` or the
  last generation's spread, which would need no per-optimizer readers at
  all. It fails on the production menu: progress-distance goes to zero
  for a stalled-but-wide-searching optimizer, and writing zero freezes
  the receiver; population spread is undefined at `popsize == 1`, which
  is `lbfgs` and `rprop` — half the menu.
- *Tracking an archive of evaluated points in `EnvState`.* Unnecessary:
  for population-based optimizers `(X, f)` already *is* the state, and
  distribution-based ones hold `(mean, std, M)` exactly, which beats a
  rank-`K` estimate of it. It is also the `O(N²)` carry cost rl2co's
  `HistorySummary` was introduced to remove.
- *Reserving unfilled bundle slots* for shape and direction so that
  adding them later is not a layout change. It buys nothing: every
  reader has to be updated to fill a new slot regardless, so the
  struct change is not the cost. NaN-filled fields nothing reads are
  dead weight in a pytree that crosses a `lax.switch`.
