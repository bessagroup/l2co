# Per-evaluation noise keys for L-BFGS live in a coexisting registry entry, with unchanged feval accounting

Stochastic tasks (`task.pass_rng`) hand their loss function an
explicit PRNG key, and the evosax adapters implement a clear noise
contract: one independent realization per function evaluation
(`jr.split(key, popsize)` per generation). The stock `"lbfgs"` entry
cannot honor that contract: `optax.lbfgs` freezes the extra keyword
arguments of `value_fn` — including the key — for the whole inner
zoom linesearch, so every trial evaluation of one outer step shares a
single realization, and `optax.value_and_grad_from_state` re-serves
the value/gradient cached from the *previous* step's realization
(each outer step straddles exactly two). Upstream optax has no hook
to vary `fn_kwargs` per linesearch iteration (see
`FEEDBACK_optax.md`).

We decided to provide the honest contract as a **separate,
coexisting registry entry** `"lbfgs_per_eval_key"`
(`l2co._src.optimizers.lbfgs_per_eval_key`), not as a flag on the
stock `"lbfgs"` wiring:

* The stock entry keeps its exact semantics — databanks already
  contain `lbfgs` trajectories, and the two noise regimes are
  different optimizers for benchmarking purposes (per-step frozen
  realizations are markedly *gentler* on L-BFGS than per-evaluation
  noise, which corrupts the zoom bracketing invariants themselves).
  A flag would blur which regime produced a stored trajectory; a
  distinct name keeps the `update_step` hash and databank labels
  unambiguous.
* The variant reuses optax's own zoom machinery: the
  `zoom_linesearch` `(init_fn, step_fn, cond_fn)` triple is driven
  by a custom `while_loop` that carries a key and splits one subkey
  per inner evaluation. Only the key threading is ours; bracketing,
  Wolfe criteria and failure handling stay upstream's.
* On deterministic tasks fresh keys are a mathematical no-op, so the
  factory falls back to the stock `optax.lbfgs` wiring
  (trajectories bit-identical to `"lbfgs"`). This keeps the
  registry-wide `test_all_optimizers` sweep meaningful and makes the
  entry safe to include in mixed task sets.

**Feval accounting is deliberately unchanged**: 1 feval per
iteration, matching the stock `"lbfgs"` entry, even though the
linesearch may spend up to `max_linesearch_steps` evaluations
(each now with its own realization). Both L-BFGS entries
under-report by the same rule, so relative comparisons between them
are fair; fixing the under-reporting itself is a separate,
registry-wide decision in the spirit of `docs/adr/0004` (it would
shift every ERT curve and invalidate comparisons against existing
databank trajectories, so it must not ride along in this change).

Rejected alternatives: patching the behavior into optax upstream
(no appetite for per-iteration `fn_kwargs` upstream yet; tracked in
`FEEDBACK_optax.md`), and deriving keys from the iterate's bits
inside `loss_fn` (works with stock optax but silently redefines the
noise model as a frozen random field — repeated evaluations at the
same point return the same draw, which breaks resampling-based noise
handling and the BBOB-noisy contract).
