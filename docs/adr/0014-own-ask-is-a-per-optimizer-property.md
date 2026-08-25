# Own-ask is a per-optimizer property, not a family one

`docs/adr/0012` made a `distribution`-family optimizer regenerate its own
post-switch population, and said the population family was "deliberately
unchanged" because for `pso`, `mr15ga` and `de` the stored population
*is* the state, so substituting one is not lossy. The first half stands.
The second half is wrong for three of them, and this supersedes it.

**The criterion was never the cached draws.** CR-FM-NES caches `z`/`y`
in state and consumes them on the next `tell`, so a substituted
population credits each candidate's fitness to a direction that produced
a different point. That is one instance of a general hazard: *the
optimizer's next update reads state the handshake just wrote*. The
mutation GAs cache no draws and hit the same wall through their
baseline:

- **MR15-GA** — the 1/5 rule is `mean(fitness < state.fitness)`
  (`mr15_ga.py:96`). The repair writes that baseline; a foreign
  population is then scored against a baseline it was not drawn from.
- **GESMR-GA** — ranks its std groups by `fitness - state.fitness`
  (`gesmr_ga.py` `_tell`, Eq. 5). Same baseline, one level deeper: with
  a stale one the group ranking is noise.
- **SAMR-GA** — inherits per-member stds through `argsort(state.fitness)`
  in `_ask`, so which mutation widths survive depends on it.

**Decision.** `TransferSpec` grows `own_ask: bool`, defaulted by family
(`distribution` true, the rest false) and overridden per registry name.
`mr15ga`, `samrga` and `gesmrga` opt in, together with the repairing
writer. Callers branch on `own_ask` — `UpdateClass.ask_fn`'s presence in
the environment, `SubOpt.own_ask` in the deployment wrappers — never on
`family`, which goes back to meaning only what kind of search state an
optimizer holds. `ask` is declared on evosax's shared base
(`algorithms/base.py:106`), so one closure serves both families.

**Who stays out, and why it has to be per-optimizer rather than per
family.**

- **DE and SHADE.** `_ask` builds mutants from difference vectors of the
  stored population (`differential_evolution.py` `_ask`). The repair
  makes every row identical, and identical rows have no differences, so
  own-ask would emit the same point `popsize` times.
- **PSO.** `_ask` reads `population_best` / `fitness_best`, which no
  writer repairs, so its own ask would draw against a baseline the
  handshake left stale — the very thing this fixes elsewhere.
- **SimpleGA.** A mutation GA of the same shape as the three that opt
  in, and it still stays out: its `_tell` overwrites `std` from a fixed
  schedule, so a transferred scale expires one generation later, and it
  replaces the population wholesale rather than selecting `(µ+λ)`, so
  there is no stored baseline to repair. Opting it in would create a
  write whose effect silently expires. It remains a scale *provider*.

## A repaired baseline claims a measurement, and claims it anyway

Writing `fitness = bundle.f` into every slot asserts that re-evaluating
`bundle.x` would return `bundle.f`. On a deterministic loss that is
true. On a stochastic one it is not: `best_loss` is a running *minimum*
over every evaluation ever made, so it is an extreme-value statistic —
the luckiest draw, not what the point returns now. Claimed into every
slot it is unbeatable: nothing satisfies `fitness < state.fitness`, the
1/5 rule reads a 0% success rate, and `std` halves geometrically.

**An earlier version of this ADR gated the claim on `task.pass_rng`.
That was wrong, and not because of the arithmetic.**

*Whether a loss is deterministic is not a property a handshake may
read.* A real problem does not arrive labelled; the caller usually
cannot say; and an update rule whose shape depends on how the objective
was declared is not one anyone can reason about, on paper or in a
report. The same run, wrapped in a noise term the user forgot to
declare, would take a different code path. So the behaviour is uniform.

**Decision.** The repair always rebases `population` / `fitness` on the
handed-over best. `build_transfer_fns` takes the optimizer's name,
family and hyperparameters — nothing that could tell it what kind of
problem it is attached to.

The cost of being uniform here is small, and measured. The optimistic
bias the claim introduces is the *same* bias the algorithm already has:
MR15-GA's `(µ+λ)` selection keeps the better of each pair, so its own
stored `fitness` is a running minimum too. Under noise its mutation
scale collapses geometrically with **no switch at all** — `1 → 0.0039`
over six generations from a reset on a `noise=0.1` BBOB task; after a
switch, `0.0048 → 4.7e-06`, the same rate from the transferred scale;
on the deterministic task the same switch gives `0.0048 → 0.0192`, no
collapse. Refusing the claim does not fix that. Whether MR15-GA belongs
on noisy tasks at all is a menu question, not a handshake one.

And refusing it would cost something real. With own-ask, the rebase is
the only way the handed-over *location* reaches a mutation GA: skip it
and its next generation is mutated around whatever population it was
already holding, making the handshake scale-only for exactly the three
optimizers this ADR just gave a location channel to.

**What is still refused** is a bundle whose numbers are unusable — a
non-finite `x` or `f`. That is a property of the values in hand, not of
the problem, so it is a judgement the writer is entitled to make.
`+inf` is the case that matters and it is the opposite failure: every
candidate beats it, so the rule reads 100% and *doubles* the width
immediately after the scale write set it. `best_loss` starts at `+inf`
and stays there while nothing has improved, which at high ambient
dimension can be a whole first generation of NaN.

**Rejected alternatives.**

- *Gate on `task.pass_rng`.* What this section replaces. Buys nothing
  the algorithm's own bias does not already impose, and pays for it by
  making the update rule depend on a declaration the caller often
  cannot make.
- *Write `+inf` instead of `best_f`.* Measured above: the opposite
  failure, and it is why the existing non-finite guard skips rather
  than softens.
- *Re-measure `bundle.x` inside the handshake.* Correct in principle
  and impossible in practice: `SubOpt` carries no loss function — the
  deployment wrappers are ask/tell strategies whose *caller* evaluates
  (`sub_optimizer.py`). Adding a loss channel to make the environment
  and the wrapper agree would break the strategy contract, and the two
  diverging is precisely what `docs/adr/0009` exists to prevent.
- *Claim only in one slot.* Keeps one elite row honest and leaves the
  rest measured, which is appealing. It also makes the 1/5 rule's
  index-wise comparison a mixture of two baselines, and no code fact
  says whether that reads better or worse than either. Left for the
  measurement pass.

## A handshake does two things, and only one of them is about the jump

The writers mix two justifications. Refreshing a stale scale is about
*time* — the incoming optimizer has not run for a while, so its `std` is
wrong about the current search. Re-centring a `mean`, zeroing a
cumulation path, rebasing an elite, clearing Rprop's `prev_updates` are
about *place*: they repair the damage of moving the iterate somewhere
the optimizer's accumulated state does not describe.

Nothing distinguished them, so a switch back into an optimizer that
already held the best point paid the full repair bill for a jump of zero
distance. Its `mean` was reset from wherever its search had drifted to
back onto its own best; its cumulation paths — the step-size control it
had spent generations building — were zeroed; MR15-GA's population
collapsed to `popsize` identical copies. All to protect it from moving
somewhere it was not moving.

**Decision.** `TransferBundle` grows `moves_iterate`, defaulting to
`True` (a caller that does not know moves the iterate, which is what
every write was written for). The place-justified writes gate on it via
`_jumping`, which also folds in the finiteness check they already
needed. The scale write does not: an outgoing optimizer that found
nothing better may still have learned the local scale, and that is worth
having.

Deciding *when* it is false stays with the caller, because the predicate
is theirs — rl2co's env compares the bundle's loss against the incoming
optimizer's own best (its `docs/adr/0009`). This module only says what
changes when it is.

**Consequence.** Environment dynamics change again, so this wave's
policies are not comparable to `0012`'s. Accepted on the same terms:
the artifacts are being retrained. The wave's own gate is a
scripted-policy A/B run at each stacked commit, which needs no policy at
all.
