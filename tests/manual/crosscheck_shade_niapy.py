"""Cross-check l2co's SHADE against NiaPy's reference implementation.

Not collected by pytest — run manually:

    uv pip install niapy   # into the project environment
    uv run python tests/manual/crosscheck_shade_niapy.py

Both implementations run canonical SHADE (NP = 50, H = NP, |A| = NP)
on the classic sphere / rastrigin / rosenbrock functions in
``[-5, 5]^10`` at an equal budget of 10,000 fitness evaluations, over
30 seeds each. The script reports per-function best-error medians and
IQRs and, when SciPy is available, a two-sided Mann-Whitney U test.

Interpretation: the implementations differ in RNG, float precision
(JAX float32 vs NumPy float64), pbest sampling (NiaPy uses a fixed
``pbest_factor``; we sample ``p_i ~ U[2/NP, 0.2]`` per the paper) and
the documented vectorization deviations (see
``l2co._src.optimizers.shade``) — with 30 seeds the U test *will*
resolve these (p ~ 0), so do not read the p-value as a pass/fail. The
pass criterion is: per-function medians within the same order of
magnitude, with no systematic inferiority across functions.

Observed on 2026-07-10 (CPU, jax 0.8, niapy 2.5):

    sphere     | l2co 1.7e-11 | niapy 2.3e-12  (gap is float32 vs
                 float64: with JAX_ENABLE_X64=1 l2co gives ~9e-12)
    rastrigin  | l2co 5.4e+00 | niapy 6.9e+00  (l2co better)
    rosenbrock | l2co 3.6e+00 | niapy 2.2e+00  (niapy better)

NiaPy was chosen as the reference because the original cross-check
target (xKuZz/pyade) has disappeared from both PyPI and GitHub. NiaPy's
``SuccessHistoryAdaptiveDifferentialEvolution`` follows Tanabe &
Fukunaga (2013) including the midpoint parent repair; its one visible
deviation is a fixed ``pbest_factor`` instead of the per-individual
``p_i ~ U[2/NP, 0.2]``, which we set to 0.11 (the midpoint).
"""

import numpy as np

N_DIMS = 10
N_SEEDS = 30
POPSIZE = 50
MAX_FEVALS = 10_000
BOUNDS = (-5.0, 5.0)


def sphere(x):
    return (x**2).sum()


def rastrigin(x):
    import numpy as xp

    if not isinstance(x, np.ndarray):
        import jax.numpy as xp
    return 10.0 * x.shape[-1] + (x**2 - 10.0 * xp.cos(2.0 * xp.pi * x)).sum()


def rosenbrock(x):
    return (100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2).sum()


FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
}


def run_l2co_shade(fn, seed: int) -> float:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from l2co_tasks import Task

    from l2co._src.optimizers.shade import shade_update

    task = Task(model=jnp.zeros(N_DIMS), loss_fn=fn, global_min=0.0)
    update_class = shade_update(
        task=task,
        opt_hash=1,
        bounded=BOUNDS,
        stop_fn=None,
        popsize=POPSIZE,
    )
    params, _ = eqx.partition(task.model, eqx.is_inexact_array)
    key_init, key_run = jr.split(jr.key(seed))
    population = jax.tree.map(
        lambda x: jr.uniform(
            key_init, (POPSIZE, *x.shape), minval=BOUNDS[0], maxval=BOUNDS[1]
        ),
        params,
    )
    state = update_class.init_fn(population, key_run)
    carry = (population, state, key_run)
    step = eqx.filter_jit(update_class.step_fn)
    # The adapter's first step re-evaluates the initial population, so
    # (1 + n_steps) * NP evaluations are spent in total.
    n_steps = MAX_FEVALS // POPSIZE - 1
    best = jnp.inf
    for _ in range(n_steps):
        carry, history = step(carry, {})
        best = jnp.minimum(best, jnp.min(history.loss))
    return float(best)


def run_niapy_shade(fn, seed: int) -> float:
    from niapy.algorithms.modified import (
        SuccessHistoryAdaptiveDifferentialEvolution,
    )
    from niapy.problems import Problem
    from niapy.task import Task as NiaTask

    class _Problem(Problem):
        def __init__(self):
            super().__init__(
                dimension=N_DIMS, lower=BOUNDS[0], upper=BOUNDS[1]
            )

        def _evaluate(self, x):
            return float(fn(np.asarray(x)))

    algorithm = SuccessHistoryAdaptiveDifferentialEvolution(
        population_size=POPSIZE,
        hist_mem_size=POPSIZE,  # H = NP
        extern_arc_rate=1.0,  # |A| = NP
        pbest_factor=0.11,
        seed=seed,
    )
    task = NiaTask(problem=_Problem(), max_evals=MAX_FEVALS)
    _, best_fitness = algorithm.run(task)
    return float(best_fitness)


def main() -> None:
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:  # pragma: no cover
        mannwhitneyu = None

    print(
        f"SHADE cross-check: l2co vs NiaPy | D={N_DIMS}, NP={POPSIZE}, "
        f"H=|A|={POPSIZE}, budget={MAX_FEVALS} fevals, {N_SEEDS} seeds"
    )
    for name, fn in FUNCTIONS.items():
        ours = np.array([run_l2co_shade(fn, seed) for seed in range(N_SEEDS)])
        reference = np.array(
            [run_niapy_shade(fn, seed) for seed in range(N_SEEDS)]
        )
        line = (
            f"{name:>10s} | l2co median {np.median(ours):.3e} "
            f"(IQR {np.percentile(ours, 25):.2e}-"
            f"{np.percentile(ours, 75):.2e}) | "
            f"niapy median {np.median(reference):.3e} "
            f"(IQR {np.percentile(reference, 25):.2e}-"
            f"{np.percentile(reference, 75):.2e})"
        )
        if mannwhitneyu is not None:
            p_value = mannwhitneyu(
                ours, reference, alternative="two-sided"
            ).pvalue
            line += f" | Mann-Whitney p={p_value:.3f}"
        print(line)


if __name__ == "__main__":
    main()
