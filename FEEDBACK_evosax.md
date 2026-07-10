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
