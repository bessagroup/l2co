# The single "lbfgs" entry keys its noise semantics on the task, with unchanged feval accounting

Stochastic tasks (`task.pass_rng`) hand their loss function an
explicit PRNG key, and the evosax adapters implement a clear noise
contract: one independent realization per function evaluation
(`jr.split(key, popsize)` per generation). The previous `"lbfgs"`
wiring (`optax.lbfgs` via `optax_extra_kwargs_fn`, key closed over
into the loss) could not honor that contract: `optax.lbfgs` freezes
the extra keyword arguments of `value_fn` — including the key — for
the whole inner zoom linesearch, so every trial evaluation of one
outer step shared a single realization, and
`optax.value_and_grad_from_state` re-serves the value/gradient cached
from the *previous* step's realization (each outer step straddles
exactly two). Upstream optax has no hook to vary `fn_kwargs` per
linesearch iteration (see `FEEDBACK_optax.md`).

We decided to keep a **single registry entry** `"lbfgs"`
(`l2co._src.optimizers.lbfgs`) whose noise semantics are keyed on
the task:

* On stochastic tasks the factory wires `lbfgs_per_eval_key`:
  optax's own `zoom_linesearch` `(init_fn, step_fn, cond_fn)` triple
  driven by a custom `while_loop` that carries a key and splits one
  subkey per inner evaluation. Only the key threading is ours;
  bracketing, Wolfe criteria and failure handling stay upstream's.
* On deterministic tasks fresh keys are a mathematical no-op, so the
  factory uses the stock `optax.lbfgs` wiring of
  `UpdateClass.optax_update_extra_kwargs` — no key is ever passed to
  the loss, and trajectories are bit-identical to the previous
  registry wiring (tested).

A separate registry name (`"lbfgs_per_eval_key"`) was considered and
rejected: `task.pass_rng` already fully determines which semantics
are meaningful, so a second name would be a configuration knob with
exactly one valid choice per task, and the registry stays minimal.
The accepted cost is that `"lbfgs"` trajectories recorded on
*stochastic* tasks before this change (per-step frozen realizations)
are not reproducible under the same label — noisy-task L-BFGS data
in existing databanks predating this ADR must be regenerated before
being compared against new runs. Deterministic-task data is
unaffected.

**Feval accounting is deliberately unchanged**: 1 feval per
iteration, even though the linesearch may spend up to
`max_linesearch_steps` evaluations (each now with its own
realization). The under-reporting rule is the same one the previous
wiring used, so relative comparisons stay fair; fixing the
under-reporting itself is a separate, registry-wide decision in the
spirit of `docs/adr/0004` (it would shift every ERT curve, so it
must not ride along in this change).

Rejected alternatives: patching the behavior into optax upstream
(no appetite for per-iteration `fn_kwargs` upstream yet; tracked in
`FEEDBACK_optax.md`), and deriving keys from the iterate's bits
inside `loss_fn` (works with stock optax but silently redefines the
noise model as a frozen random field — repeated evaluations at the
same point return the same draw, which breaks resampling-based noise
handling and the BBOB-noisy contract).
