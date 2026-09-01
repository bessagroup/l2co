# evosax integration friction

Constraints of the `evosax` algorithm API encountered while implementing
algorithms against it (first case: SHADE, `l2co._src.optimizers.shade`).
Recorded so future integrations don't re-discover them.

## No bounds channel into algorithms

`EvolutionaryAlgorithm.__init__` and the `ask`/`tell` API
(`evosax/algorithms/base.py:56-150`) have no notion of box bounds; l2co
applies bounds by clipping *after* `ask` in the adapter
(`l2co._src.optimizers.update_class.evosax_population_based_fn`, the
`jnp.clip(p, *bounded)` line). Algorithms whose internals need bounds —
SHADE's midpoint boundary repair (which repairs toward the *parent*, not
the bound) and its turning-based mutation (ring radii derived from the
search-range span) — must smuggle them in as `Params` fields. SHADE uses
`x_min = -inf` / `x_max = +inf` defaults so every bounds-dependent
operation degrades to a no-op via `jnp.where` when unbounded, and the
`shade_update` factory injects the run's `bounded` tuple into the params
(explicit hyperparameters win).

## Constructor kwargs vs Params split

l2co forwards `OptimizationStep.hyperparameters` as
`optimizer.default_params.replace(**hyperparameters)`
(`update_class.py`, `evosax_population_update`). Flax-struct `Params`
can only hold value-like leaves — anything that determines an array
*shape* (SHADE: `memory_size`, `archive_size`) must be a constructor
argument instead, which the generic evosax factory classmethods never
pass. Custom algorithms therefore need their own factory
(`shade_update`) that pops shape-affecting kwargs before the
`.replace(...)`. The same split leaks into rl2co's
`_optstep_to_subopt`, which replicates the `default_params.replace`
call.

## Registry membership drives family dispatch

Both l2co (`evosax_implementations._factory_for`) and rl2co
(`sub_optimizer._optstep_to_subopt`) decide distribution- vs
population-based by *membership in evosax's registry dicts*
(`distribution_based_algorithms.values()` /
`population_based_algorithms.values()`), not by `issubclass`. A
third-party subclass like SHADE is invisible to that dispatch; it needs
an explicit registry entry (`shade_mapping` merged in
`l2co._src.optimizers.mapping`) and explicit handling on the rl2co
side.

## Budget-blindness

Algorithms never learn the run's total evaluation budget: neither
`__init__` nor `Params` receives it from the framework. FES-dependent
schedules (SHADE's turning-based outer radius, and the reason L-SHADE's
linear population-size reduction was not implemented) require the user
to pass `max_fevals` as a hyperparameter, with `FES` reconstructed as
`population_size * generation_counter` inside `_ask`.

## Ask/tell-coupled state

Several algorithms cache the artefacts of their own `ask` in `State` and
consume them on the next `tell`, pairing them with the reported fitness
**index for index**. CR-FM-NES is the clearest: `_ask` stores its draws
(`cr_fm_nes.py:184`, `state.replace(z=z, y=y)`) and `_tell` then reads
`state.z` / `state.y` rather than the population it is told about —
`p_std` via `jnp.dot(fitness, state.z)` (`:201`), the distance weights
via `exp(alpha_dist * ‖state.z‖)` (`:207`), and the whole `v` / `D`
update via `exY = append(state.y, p_c / D)` (`:235`). Only
`wxm = dot(weights, population - state.mean)` reads the population.

The API does not express this. `tell(key, population, fitness, state,
params)` looks like a pure function of the population it is handed, so
nothing signals that calling it with a population the algorithm did not
draw is meaningless rather than merely lossy. There is no way to
invalidate the cache through the public interface, and no marker
distinguishing an algorithm that has one.

Affected, all `distribution_based`: `cr_fm_nes` (`z`, `y`), `ma_es` and
`lm_ma_es` (`z`), `xnes` (`z`), `guided_es` (`grad_subspace`, `z`),
`noise_reuse_es` (`pert`), `persistent_es` (`pert_accum`), `asebo`
(`grad_subspace`, `UUT`, `UUT_ort`), `rm_es` (`fitness_elites_sorted`).

Any consumer that substitutes a population — an optimizer-switching
handshake, a warm start, an externally seeded generation — must
regenerate it through the algorithm's own `ask` instead of writing one
in. l2co does that via `UpdateClass.ask_fn`; see `docs/adr/0012`.

## Distribution `init` takes its mean from slot 0

`DistributionBasedAlgorithm.init(key, mean, params)` sets
`state.mean = self._ravel_solution(mean)` (`distribution_based/base.py:
60-69`), so when the caller passes a whole population the mean comes
from **row 0** specifically. A caller whose population layout puts the
distinguished point elsewhere — l2co's `population_best_one` puts the
best point last — silently re-centres the distribution on an arbitrary
row. Also note `state.mean` is stored *raveled* while the argument is a
solution pytree, so anything writing `mean` directly needs the
algorithm's `_ravel_solution`.

## Population-based `State.fitness` is documented as shaped, but usually is not

`population_based.base.State.fitness` is annotated `# Shaped fitness`,
and `tell` does apply `fitness_shaping_fn` before `_tell`. In practice
`differential_evolution`, `pso`, `mr15_ga`, `simple_ga`, `samr_ga` and
`gesmr_ga` all default to `identity_fitness_shaping_fn`, so the stored
values are raw losses — while `cma_es` and `cr_fm_nes` default to
`weights_fitness_shaping_fn` and would store recombination weights if
they stored fitness at all. A consumer reading `State.fitness` as a loss
is relying on the algorithm's *default shaping function*, not on the
documented contract, and a caller passing a non-default
`fitness_shaping_fn` breaks that silently.
