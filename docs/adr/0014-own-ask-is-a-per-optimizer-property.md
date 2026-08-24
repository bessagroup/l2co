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

## A repaired baseline is a claim, and on a noisy task it is false

Writing `fitness = bundle.f` into every slot asserts that re-evaluating
`bundle.x` would return `bundle.f`. On a deterministic loss that is
true. On a stochastic one it is not: `best_loss` is a running *minimum*
over every evaluation ever made, so it is an extreme-value statistic —
the luckiest draw, not what the point returns now.

Claimed into every slot it becomes unbeatable. Nothing satisfies
`fitness < state.fitness`, the 1/5 rule reads a 0% success rate on every
generation, and `std` halves geometrically until it underflows and the
optimizer stops moving — at which point `read_sigma_population` reports
`CONF_ABSENT`, so it stops providing a scale too. Measured over six
generations: `0.25 → 0.0039`, exactly `2⁻⁶`, against `0.0625` when the
claim is skipped.

This is not a corner case for this project. 25 of the 100 tasks in
`bbob_headroom_train` are `bbob_noisy`, which sets `pass_rng=True`.

**What this does not fix, and an earlier draft implied it did.** MR15-GA
collapses its mutation scale under noise *whatever the handshake does*,
because the 1/5 rule scores each generation against its own `(µ+λ)`
selected fitness — a running minimum over the pair, so biased optimistic
in exactly the way `best_loss` is. Measured on a `noise=0.1` BBOB task:
six generations from a reset with **no switch at all** run `std` from
`1 → 0.0039`; after a switch they run `0.0048 → 4.7e-06`, the same rate
from the transferred scale. The same switch on the deterministic task
gives `0.0048 → 0.0192`, no collapse. So the collapse is the algorithm's
under noise, pre-existing and unrelated to state transfer. Gating the
claim stops the handshake *adding* an unbeatable baseline on top of it;
it does not, and cannot, stop that. Whether MR15-GA belongs on noisy
tasks at all is a menu question, not a handshake one.

**Decision.** `build_transfer_fns` takes `deterministic` (the negation
of `task.pass_rng`) and the repairing writer skips the rebase entirely
where it is false, leaving the optimizer its own measured baseline. The
scale write is unaffected — only the *claim* is gated.

**Rejected alternatives.**

- *Write `+inf` instead of `best_f`.* The opposite failure, and already
  measured: every candidate beats it, the rule reads 100% and doubles
  the width immediately after the scale write set it. It is why the
  existing non-finite guard skips rather than softens.
- *Re-measure `bundle.x` inside the handshake.* Correct in principle and
  impossible in practice: `SubOpt` carries no loss function — the
  deployment wrappers are ask/tell strategies whose *caller* evaluates
  (`sub_optimizer.py`). Adding a loss channel to make the environment
  and the wrapper agree would break the strategy contract, and the two
  diverging is precisely what `docs/adr/0009` exists to prevent.
- *Claim only in one slot.* Keeps one elite row honest and leaves the
  rest measured, which is appealing. It also makes the 1/5 rule's
  index-wise comparison a mixture of two baselines, and no code fact
  says whether that reads better or worse than either. Left for the
  measurement pass.

**Consequence.** Environment dynamics change again, so this wave's
policies are not comparable to `0012`'s. Accepted on the same terms:
the artifacts are being retrained. The wave's own gate is a
scripted-policy A/B run at each stacked commit, which needs no policy at
all.
