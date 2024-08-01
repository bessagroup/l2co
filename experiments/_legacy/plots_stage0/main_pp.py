#                                                                       Modules
# =============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import xarray as xr
from f3dasm import ExperimentData
from hydra import compose, initialize
from hydra.utils import instantiate
from sklearn.model_selection import KFold

from l2co import LSTMMultiOutputClassifier
from l2co._src.collaborative_filtering import (collaborative_filtering,
                                               remove_performance_values)
from l2co._src.preprocessing import preprocessing_factory
from l2co._src.stage0_model import Stage0Model
from l2co._src.stage1_model import ELAClassifier
from l2co.data import DataLoader
from l2co.plotting import set_size
from l2co.trainer import Trainer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

plt.style.use(['science', 'bright'])
plt.rcParams.update({'svg.fonttype': 'none'})

filter_1 = {'noise': 0.0, 'convex': True, 'differentiable': True}
filter_2 = {'noise': 0.1, 'convex': True,
            'differentiable': True, 'dim': 100}
filter_3 = {'dim': 50, 'convex': False, 'noise': 0.1}

# =============================================================================


def filter_dataarray(data_array, filters):
    """
    Filter a DataArray based on feature-value pairs.

    Parameters:
        data_array (xarray.DataArray): The DataArray to filter.
        filters (dict): A dictionary containing feature-value pairs to filter on.

    Returns:
        xarray.DataArray: Filtered DataArray.
    """
    condition = None
    for feature, value in filters.items():
        if condition is None:
            condition = (data_array.X.sel(features=feature) == value)
        else:
            condition &= (data_array.X.sel(features=feature) == value)
    return data_array.X.sel(problem=condition.where(condition, drop=True).problem.values).problem

# =============================================================================


def plot_f(pp_selected: xr.Dataset):
    fig, ax = plt.subplots(1, 1, figsize=set_size(width=345, subplots=(1, 1)))
    x = pp_selected.f * 100 - 100
    for opt in pp_selected.data_vars:

        if opt == 'trainer_stage0':
            ax.plot(x, pp_selected[opt].mean('kfold'), label='stage0',
                    linestyle='--', color='k')
            # pp_selected[opt].mean('kfold').plot(ax=ax, label='stage0',
            #                                     linestyle='--', color='k')

            ax.fill_between(
                x,
                pp_selected[opt].mean('kfold') -
                pp_selected[opt].std('kfold'),
                pp_selected[opt].mean('kfold') +
                pp_selected[opt].std('kfold'),
                alpha=0.2,
                color='k')
        else:
            ax.plot(x, pp_selected[opt].mean('kfold'), label=opt
                    )
            # pp_selected[opt].mean('kfold').plot(ax=ax, label=opt)
            # plot std boundaries
            ax.fill_between(
                x,
                pp_selected[opt].mean('kfold') -
                pp_selected[opt].std('kfold'),
                pp_selected[opt].mean('kfold') +
                pp_selected[opt].std('kfold'),
                alpha=0.2)

    # ax.legend()
    ax.set_xlabel('Relative error [\%]')
    ax.set_ylabel('Fraction of solved problems')
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-10., 300)

    return fig, ax


# def plot_all(rtd_1: xr.Dataset, rtd_2: xr.Dataset, rtd_3: xr.Dataset):
#     plt.rcParams.update({'svg.fonttype': 'none'})
#     fig, ax = plt.subplots(1, 4, figsize=set_size(width=340, subplots=(1, 4)))
#     ax = ax.ravel()

#     for opt in rtd_1.data_vars:
#         rtd_1[opt].mean('kfold').plot(ax=ax[0], label=opt)
#         # plot std boundaries
#         ax[0].fill_between(
#             rtd_1.iterations,
#             rtd_1[opt].mean('kfold') - rtd_1[opt].std('kfold'),
#             rtd_1[opt].mean('kfold') + rtd_1[opt].std('kfold'),
#             alpha=0.2)

#         rtd_2[opt].mean('kfold').plot(ax=ax[1], label=opt)
#         # plot std boundaries
#         ax[1].fill_between(
#             rtd_2.iterations,
#             rtd_2[opt].mean('kfold') - rtd_2[opt].std('kfold'),
#             rtd_2[opt].mean('kfold') + rtd_2[opt].std('kfold'),
#             alpha=0.2)

#         rtd_3[opt].mean('kfold').plot(ax=ax[2], label=opt)
#         # plot std boundaries
#         ax[2].fill_between(
#             rtd_3.iterations,
#             rtd_3[opt].mean('kfold') - rtd_3[opt].std('kfold'),
#             rtd_3[opt].mean('kfold') + rtd_3[opt].std('kfold'),
#             alpha=0.2)

#     for a in ax:
#         a.legend()
#         a.set_xlabel('Function evaluations')
#         a.set_ylabel('Fraction of function, target pairs')
#         a.set_ylim(0.0, 1.0)

#     return fig, ax
# =============================================================================


with initialize(version_base=None, config_path="."):
    config = compose(config_name="config")

# Construct the classifiers
stage0_clf = instantiate(config.stage0_clf)
stage1_clf = LSTMMultiOutputClassifier(
    n_outputs=config.n_optimizers, length_sequence=config.t_switch,
    **config.stage1_clf)

preprocessing_step = preprocessing_factory(
    feature_types=config.feature_types)

path = Path(config.data.training_project_dir)

data = xr.open_dataset(
    path / "entire_dataset.nc", chunks='auto')
data_samples = xr.open_dataset(
    path / "entire_dataset_samples.nc", chunks='auto')
experimentdata = ExperimentData.from_file(path)


d = DataLoader.from_preprocessed_files(
    config={'t_switch': config.t_switch},
    experimentdata=experimentdata,
    data=data,
    data_samples=data_samples,)

d_train, d_test = d.train_test_split(config.test_size)

filter_1 = {'noise': 0.0, 'convex': True, 'differentiable': True}
filter_2 = {'noise': 0.1, 'convex': True,
            'differentiable': True, 'dim': 100}
filter_3 = {'dim': 50, 'convex': False, 'noise': 0.1}

kf = KFold(n_splits=config.folds, shuffle=True, random_state=config.seed)


pp_kfold, pp_lbfgsb, pp_adam, pp_pso = [], [], [], []

for index, (i_train, i_test) in enumerate(kf.split(range(d_train.problem.size))):
    print(f"Fold {index}")
    trainer = Trainer(
        data_train=d_train.select_problems(
            d_train.problem.isel(problem=i_train)),
        data_test=d_train.select_problems(
            d_train.problem.isel(problem=i_test)),
        stage0_clf=stage0_clf,
        stage1_clf=stage1_clf,
        iterations=config.iterations-1,
        t_switch=config.t_switch,
        t_start=config.t_start,
        rel_tolerance=config.rel_tolerance,
        preprocessing_step=preprocessing_step,
        epochs=config.epochs,
        batch_size=config.batch_size,
        name='trainer')

    trainer.train_stage0()

    trainer.stage0_prediction = trainer.predict_stage0()

    pp_p = trainer.data_test.select_single_optimizers().performance_profile(
        strategies=[trainer.stage0_prediction.sortby('problem')])

    sel_problems_1 = filter_dataarray(trainer.data_test, filter_1)
    sel_problems_2 = filter_dataarray(trainer.data_test, filter_2)
    sel_problems_3 = filter_dataarray(trainer.data_test, filter_3)

    lbfgsb = trainer.data_test.select_single_optimizers().select_problems(
        sel_problems_1).performance_profile(
        strategies=[trainer.stage0_prediction.sortby('problem').sel(
            problem=sel_problems_1)])

    adam = trainer.data_test.select_single_optimizers().select_problems(
        sel_problems_2).performance_profile(
        strategies=[trainer.stage0_prediction.sortby('problem').sel(
            problem=sel_problems_2)])

    pso = trainer.data_test.select_single_optimizers().select_problems(
        sel_problems_3).performance_profile(
        strategies=[trainer.stage0_prediction.sortby('problem').sel(
            problem=sel_problems_3)])

    pp_kfold.append(pp_p.compute())
    pp_lbfgsb.append(lbfgsb.compute())
    pp_adam.append(adam.compute())
    pp_pso.append(pso.compute())

print(f"Finished {index} folds")
xr_pp_kfold = xr.concat(pp_kfold, dim='kfold')
xr_pp_lbfgsb = xr.concat(pp_lbfgsb, dim='kfold')
xr_pp_adam = xr.concat(pp_adam, dim='kfold')
xr_pp_pso = xr.concat(pp_pso, dim='kfold')

print("Finished concatting")

# fig, _ = plot_all(xr_rtd_lbfgsb, xr_rtd_adam, xr_rtd_pso)
# fig.savefig('rtd_all.svg', bbox_inches='tight')

plt.rcParams.update({'svg.fonttype': 'none'})

fig, _ = plot_f(xr_pp_kfold)
fig.savefig('pp_kfold.svg', bbox_inches='tight')

fig, _ = plot_f(xr_pp_lbfgsb)
fig.savefig('pp_lbfgsb.svg', bbox_inches='tight')

fig, _ = plot_f(xr_pp_adam)
fig.savefig('pp_adam.svg', bbox_inches='tight')

fig, _ = plot_f(xr_pp_pso)
fig.savefig('pp_pso.svg', bbox_inches='tight')
