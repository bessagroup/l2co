# Ask-coupled optimizer state cannot survive a population substitution

The switching handshake hands the incoming optimizer a population it did
not draw. For most algorithms that is merely lossy. For a class of
evosax algorithms it is **wrong**, because they cache the artefacts of
their own `ask` in state and consume them on the next `tell`.

CR-FM-NES is the clearest case. `_ask` stores its draws
(`cr_fm_nes.py:184`, `state.replace(z=z, y=y)`) and `_tell` then reads
those cached draws rather than the population it is told about:

- `p_std` accumulates `jnp.dot(fitness, state.z)` (`:201`);
- the distance weights are `exp(alpha_dist * ‖state.z‖)` (`:207`);
- the entire `v` / `D` covariance update is built from
  `exY = append(state.y, p_c / D)` (`:235`).

Only `wxm = dot(weights, population - state.mean)` reads the real
population. So after a `best_one` handshake, candidate *i*'s fitness is
credited to a search direction that produced a completely different
point — possibly generated many generations earlier, under a different
`mean` and `std`. The evolution path, the rank-one and rank-mu updates
and the step-size branch (`movement_cond`) are all driven by
noise-correlated garbage. Nothing raises; the algorithm simply adapts
against a fiction.

The same pattern holds for `maes` and `lmmaes` (`z`), `xnes` (`z`),
`guidedes` (`grad_subspace`, `z`), `noisereusees` (`pert`),
`persistentes` (`pert_accum`), `asebo` (`grad_subspace`, `UUT`,
`UUT_ort`) and `rmes` (`fitness_elites_sorted`) — nine algorithms, all
`distribution_based`, and **none of them tabulated in
`HANDSHAKE_POLICY`**, so every one resolved to `DEFAULT_HANDSHAKE` =
`best_one` + `continue`: exactly the broken combination. `crfmnes` is in
the production optimizer menu.

**Decision.** For the `distribution` family, do not substitute a
population at all. Write the handed-over location and scale into the
state and then call the incoming optimizer's **own** `ask` to produce
the population. The cached draws are then consistent with the population
that gets told back, by construction — there is no invalidation logic to
write and no per-algorithm list of fields to remember. `UpdateClass`
gains an `ask_fn` for this; `SubOpt` already exposed one
(`pop_sub_optimizer` stores `step_fn=ask_fn`), which is why the two
deployment wrappers need no new plumbing.

This subsumes the population half of the handshake for that family: the
population stops being an independently chosen layout and becomes a
consequence of the state write. `family` on `UpdateClass` is what a
caller keys on. It is set by the constructing factory, which knows what
it is building — not inferred from evosax registry membership, which
`FEEDBACK_evosax.md` already records as friction.

**Where the best point goes, then.** Into the `mean`. It stops being a
member of the population and becomes the *location* of the search
distribution, which the writer sets before `ask` runs:
`mean ← ravel(bundle.x)`, `std ← sigma / sqrt(factor)`. Everything the
handshake conveys to this family is those two writes — `bundle.f` is not
read at all (only `mr15ga`'s baseline repair uses it). The population is
what falls out of them. At `d = 10`, writing a best point and a scale of
`0.05` into CR-FM-NES and then asking:

    mean == best        : True
    best IS a member?   : False        <- the centre, not a sample
    min |row - best|    : 0.1317
    population centroid : |centroid - best| = 0.0000
    per-coord RMS spread: 0.0523       (sigma asked for 0.0500)

The centroid landing exactly on the best point is CR-FM-NES's mirrored
sampling — it draws `concat([z, -z])` (`cr_fm_nes.py:176-177`), so the
population is symmetric about the mean by construction.

Losing the best point as a *member* costs nothing, which is worth stating
because it looks like a regression. A distribution-based algorithm keeps
no members: its memory of a good point **is** the mean, so putting that
point in the population was never how it remembered it — under `best_one`
the point sat in a slot the algorithm would discard after one `tell`,
while the mean it actually searches from came from somewhere else
entirely. The caller's own best-so-far record is untouched by the
handshake (the `RunState`'s `best_params` / `best_loss` carry through
unchanged, and a switching caller holds a cross-optimizer record besides),
so nothing is forgotten by not re-listing it.

**The population family is deliberately unchanged**, and the split is
exactly whether an optimizer holds points or holds a distribution over
points. For `pso`, `mr15ga` and `de` the stored population *is* the
state, so `best_one` still injects the best point as a member there —
there is no mean to write it into, and substituting a population is not
lossy for an algorithm whose state is a population.

> **Amended by `docs/adr/0014`.** That paragraph is wrong for the
> mutation GAs. The criterion is not "does it cache its draws" but "does
> its next update read state the handshake just wrote", and `mr15ga`,
> `samrga` and `gesmrga` do — through the `state.fitness` baseline the
> repairing writer sets, not through a cached draw. They now draw their
> own post-switch population too, decided by `TransferSpec.own_ask`
> rather than by family. `pso`, `de` and `shade` do stay on `best_one`,
> for the reasons `0014` records.

Three further problems fall out at the same time. CMA-ES's `_tell`
reconstructs steps as `y_k = (population - mean) / std`
(`cma_es.py:222`), which is only meaningful if the population was drawn
from that distribution — now it is. The wasted generation goes away: the
`popsize` evaluations the first post-switch generation costs now buy a
real generation from a warm distribution instead of a draw from the
global sampler, and none is spent re-measuring the handed-over point's
already-known loss the way the occupied slot was. And the cost model is
unchanged, so handshake variants stay comparable on feval-billed
metrics.

**Rejected alternatives.**

- *Tabulate the nine as `("best_one", "reset")`.* The interim fix, and
  it is worse than it looks. evosax's distribution `init_fn` takes its
  mean from slot 0 (`update_class.py:373`) while `best_one` writes the
  best point to slot −1, so the distribution would be re-centred on a
  random draw. Fixable by re-initialising at `best_params` instead — but
  `reset` also discards `D` and `v`, and restores `std` to `std_init` on
  *every* switch-in. On a menu whose members' initial scales span three
  orders of magnitude (`rprop` at `learning_rate=1e-3` against an ES at
  `std_init=1.0`), a converged late-episode search restarting at
  `std = 1.0` is a larger error than the mis-attribution it fixes.
- *A surgical `invalidate` variant* — zero the ask-coupled fields and
  the cumulation paths, keep `mean`/`std`/`D`/`v`. Correct in principle
  and it preserves shape, but it needs per-field surgery across nine
  algorithms and a division-safety argument for each: with `z = y = 0`
  the `v`/`D` update runs on zero information, and nothing guarantees
  that path is NaN-free. Own-ask reaches the same end state without
  hand-written field lists.
- *Leave it and document.* The affected algorithm is one of four members
  of the production menu, and the failure is silent adaptation against a
  fiction rather than a visible error.

**Consequence.** Environment dynamics change, so policies trained under
the old handshake are not comparable to ones trained after it. That was
accepted deliberately: the existing artifacts are being retrained.
