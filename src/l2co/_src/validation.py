from numbers import Integral, Real

import numpy as np
import xarray as xr

from l2co._src.optimizerschedule import step_n_iterations

# =============================================================================

BUDGET_STEPS = 1000


def validation_fn(x: xr.Dataset,
                  budget_steps: int = BUDGET_STEPS) -> xr.Dataset:

    # Negate this validation if the dataset is already processed
    if 'output_mean' in x:
        return x

    popsizes = x.attrs['popsizes']

    if isinstance(popsizes, Integral):
        popsizes = [popsizes]

    budgets = x.attrs['budgets']

    if isinstance(budgets, Real):
        budgets = [budgets]

    n_iterations = x.attrs['n_iterations']

    iterations_per_step = step_n_iterations(n_iterations=n_iterations,
                                            budgets=budgets)

    bins = np.concatenate(([0], np.cumsum(
        [i*pop for
            pop, i in zip(popsizes, iterations_per_step)])))

    bins[-1] += 1

    y_bins = x['output'].groupby_bins(
        "budget", bins=bins, include_lowest=True, right=False)
    opt_step_bins = x['opt_step'].groupby_bins(
        "budget", bins=bins, include_lowest=True, right=False)

    y_coarsen = [y_group.coarsen(budget=pop)
                 for (_, y_group), pop in zip(y_bins, popsizes)]
    opt_step_coarsen = [opt_step_group.coarsen(budget=pop)
                        for (_, opt_step_group),
                        pop in zip(opt_step_bins, popsizes)]

    y_mean: xr.DataArray = xr.concat(
        [_y.mean() for _y in y_coarsen], dim='budget')
    y_std = xr.concat([_y.std() for _y in y_coarsen], dim='budget')
    y_min = xr.concat([_y.min() for _y in y_coarsen], dim='budget')
    opt_step = xr.concat(
        [_opt_step.min() for _opt_step in opt_step_coarsen],
        dim='budget')

    x = xr.Dataset(
        {
            'output_mean': y_mean,
            'output_std': y_std,
            'output_min': y_min,
            'opt_step': opt_step
        },
        coords={'budget': np.linspace(0., 1., n_iterations)})

    x = x.interp(budget=np.linspace(
        0., 1., budget_steps), method='nearest')

    return x
