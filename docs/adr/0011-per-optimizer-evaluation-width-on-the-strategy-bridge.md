# A generation evaluates the active optimizer's width, not the menu's

A strategy wrapper must emit every population at one fixed width —
`max(popsize)` over the menu — because a single `ask` returns a single
shape whichever optimizer is active. `l2co_population_based_fn` then
took that width as the *evaluation* width too: it called
`vmapped_loss_and_grad` over all of it, unconditionally, every
generation. Two costs followed, and neither was visible in the feval
accounting (`fevals=popsizes[active_idx]` was always honest, so ERT
curves were never wrong — only the wall clock was). A gradient
generation, which consumes exactly one row, paid a `max(popsize)`-wide
forward *and* backward pass; a population generation paid a backward
pass it discards entirely, since population optimizers read no
gradients. On a Helmholtz PINN (`d=2971`, menu popsize 27) one
generation cost 1067 ms where a population generation needs 219 ms and
a gradient one 42 ms — up to **25× the compute the generation's own
feval count claimed**.

The evaluation therefore moves behind `menu_loss_and_grad`
(`l2co._src.optimizers.menu_eval`, exposed on the `l2co.optimizers`
authoring facade per ADR `0007`), which builds one `jax.lax.switch`
branch per distinct `(kind, width)` pair in the menu and dispatches on
`active_opt_idx`. Because `SubOpt.is_pop` and `SubOpt.popsize` are both
static, the branch bodies specialise at trace time and the
gradient/population distinction costs no runtime predicate at all — the
`lax.cond` on `active_needs_grads` that rl2co's bridge uses is not
needed here. rl2co's `rl2co_population_based_fn` still carries the
padding half of this waste; the helper takes `(sub_optimizers,
population_size)` rather than a strategy object so adopting it there is
a signature match rather than a rewrite.

**Branches are deduplicated for tidiness, not for speed — we measured,
and it does not matter.** The production driver is `batch_run_`, which
vmaps `step` over `n_realizations` (25 in
`conf/base_create_optimization_dataset.yaml`). Under `vmap` the
dispatch index is per-realization, so `lax.switch` lowers to a `select`
over *every* branch and all of them execute — which suggested that
collapsing the canonical menu `[sepcmaes, adam, lbfgs, pso]` from four
entries to its two distinct shapes (`("pop", 27)` for sepcmaes and pso,
`("grad", 1)` for adam and lbfgs) would halve the work. It does not:
measured per-realization, deduplicated is 264.9 ms against 261.1 ms
not, i.e. no difference beyond noise. XLA's CSE already collapses
identical branch bodies, so the duplicate branches were never separate
work.

Deduplication is kept anyway, for two reasons that survive: it cuts
trace-and-compile time (2.17 s against 2.92 s for the same menu, 1.35×),
and it makes the `branches` accessor an honest statement of how many
distinct evaluation shapes a menu really has — the thing a reader
debugging an unexpectedly expensive menu wants to see. Anyone tempted
to remove it should know it buys no runtime, so the bar is only whether
0.75 s of compile and the introspection are worth ~20 lines.

**The vmapped path gets a uniform speedup, and that is the floor.**
With realizations genuinely running different optimizers in the same
generation, no single narrow width serves them all; the union of the
distinct widths is the best achievable, and the dispatch reaches it.
Measured on the menu and task above: the sequential drivers (`evaluate`
/ `run_`) execute only the taken branch and go from 1067 ms to 219 ms
on a population generation (**4.9×**) and 42 ms on a gradient one
(**25.3×**); vmapped over 8 realizations the per-realization cost goes
from 1208 ms to 248 ms (**4.9×**).

Rejected: splitting the rollout into a per-phase scan at each phase's
native width. Phase A *is* uniform across realizations (`stage0_idx` is
static per task), so it could run narrow — but at the configured
`t: 0.1` phase A is a tenth of the run, so from the measured
per-generation costs it would move the vmapped figure from 4.9× to
about 5.3×. Not worth restructuring `run_` / `batch_run_` carry types
and `HistoryState` widths for.

**Gradient generations evaluate one row, and that costs bit-exactness.**
`GRAD_EVALUATION_WIDTH = 1`: the iterate `population[0]` is the only
row any gradient sub-optimizer consumes. At width exactly one XLA takes
a degenerate-vmap code path whose gradient differs from the wide pass
by 4.33e-06 relative; every width ≥ 2 is bit-identical to it. Neither
answer is closer to the float64 truth — they sit ~30 float32 ulp apart
— but Adam trajectories are chaotic, so over a few hundred steps a
stored convergence curve will visibly diverge rather than merely
wobble. Measured through the helper on the menu above, a gradient
generation costs 41.7 ms at width 1 (26.7× against the 1114 ms
pre-change pass) and 131.7 ms at width 2 (8.5×, bit-exact). Width 2 was
rejected in favour of the speed, so **runs predating this change whose
menu contained a gradient optimizer do not bit-reproduce**. This also
converges the l2co bridge on rl2co's existing convention, which already
evaluates the iterate alone. `tests/unit/test_menu_loss_and_grad.py::
test_grad_iterate_matches_full_width_within_tolerance` pins the
tolerance, so tightening it to exact equality fails loudly and sends
the reader back here.

**Population generations now record all-NaN gradients.** They no longer
compute any, so `OptHistory.grads` is NaN for them. This is
unobservable on every configured path — `_reduce_entry` discards
gradients before they reach a run's output, and the only consumer,
`gradient_based_stopping_criteria`, is opt-in and already inapplicable
to population optimizers, because l2co's native evosax factories write
`full_like(p, nan)` there too. The change brings the strategy bridge
into line with them rather than away.
