"""
"""

#                                                                       Modules
# =============================================================================

# Standard
from copy import deepcopy
from typing import Optional

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from pflacco.classical_ela_features import (calculate_dispersion,
                                            calculate_ela_distribution,
                                            calculate_ela_level,
                                            calculate_ela_meta,
                                            calculate_information_content,
                                            calculate_nbc)
from sklearn.decomposition import PCA
from tqdm import tqdm

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Planning'
# =============================================================================


ELA_FEATURE_LABELS = {
    'ela_distr.skewness': 'scaler',
    'ela_distr.kurtosis': 'scaler',
    'ela_distr.number_of_peaks': 'ordinal',
    'ela_distr.costs_runtime': 'scaler',
    'ela_meta.lin_simple.adj_r2': 'scaler',
    'ela_meta.lin_simple.intercept': 'scaler',
    'ela_meta.lin_simple.coef.min': 'scaler',
    'ela_meta.lin_simple.coef.max': 'scaler',
    'ela_meta.lin_simple.coef.max_by_min': 'scaler',
    'ela_meta.lin_w_interact.adj_r2': 'scaler',
    'ela_meta.quad_simple.adj_r2': 'scaler',
    'ela_meta.quad_simple.cond': 'scaler',
    'ela_meta.quad_w_interact.adj_r2': 'scaler',
    'ela_meta.costs_runtime': 'scaler',
    'ela_level.mmce_lda_10': 'scaler',
    'ela_level.mmce_qda_10': 'scaler',
    'ela_level.lda_qda_10': 'scaler',
    'ela_level.mmce_lda_25': 'scaler',
    'ela_level.mmce_qda_25': 'scaler',
    'ela_level.lda_qda_25': 'scaler',
    'ela_level.mmce_lda_50': 'scaler',
    'ela_level.mmce_qda_50': 'scaler',
    'ela_level.lda_qda_50': 'scaler',
    'ela_level.costs_runtime': 'scaler',
    'disp.ratio_mean_02': 'scaler',
    'disp.ratio_mean_05': 'scaler',
    'disp.ratio_mean_10': 'scaler',
    'disp.ratio_mean_25': 'scaler',
    'disp.ratio_median_02': 'scaler',
    'disp.ratio_median_05': 'scaler',
    'disp.ratio_median_10': 'scaler',
    'disp.ratio_median_25': 'scaler',
    'disp.diff_mean_02': 'scaler',
    'disp.diff_mean_05': 'scaler',
    'disp.diff_mean_10': 'scaler',
    'disp.diff_mean_25': 'scaler',
    'disp.diff_median_02': 'scaler',
    'disp.diff_median_05': 'scaler',
    'disp.diff_median_10': 'scaler',
    'disp.diff_median_25': 'scaler',
    'disp.costs_runtime': 'scaler',
    'ic.h_max': 'scaler',
    'ic.eps_s': 'scaler',
    'ic.eps_max': 'scaler',
    'ic.eps_ratio': 'scaler',
    'ic.m0': 'scaler',
    'ic.costs_runtime': 'scaler',
    'nbc.nn_nb.sd_ratio': 'scaler',
    'nbc.nn_nb.mean_ratio': 'scaler',
    'nbc.nn_nb.cor': 'scaler',
    'nbc.dist_ratio.coeff_var': 'scaler',
    'nbc.nb_fitness.cor': 'scaler',
    'nbc.costs_runtime': 'scaler',
}

empty_labels = {
    'ela_distr.skewness': np.nan,
    'ela_distr.kurtosis': np.nan,
    'ela_distr.number_of_peaks': np.nan,
    'ela_distr.costs_runtime': np.nan,
    'ela_meta.lin_simple.adj_r2': np.nan,
    'ela_meta.lin_simple.intercept': np.nan,
    'ela_meta.lin_simple.coef.min': np.nan,
    'ela_meta.lin_simple.coef.max': np.nan,
    'ela_meta.lin_simple.coef.max_by_min': np.nan,
    'ela_meta.lin_w_interact.adj_r2': np.nan,
    'ela_meta.quad_simple.adj_r2': np.nan,
    'ela_meta.quad_simple.cond': np.nan,
    'ela_meta.quad_w_interact.adj_r2': np.nan,
    'ela_meta.costs_runtime': np.nan,
    'ela_level.mmce_lda_10': np.nan,
    'ela_level.mmce_qda_10': np.nan,
    'ela_level.lda_qda_10': np.nan,
    'ela_level.mmce_lda_25': np.nan,
    'ela_level.mmce_qda_25': np.nan,
    'ela_level.lda_qda_25': np.nan,
    'ela_level.mmce_lda_50': np.nan,
    'ela_level.mmce_qda_50': np.nan,
    'ela_level.lda_qda_50': np.nan,
    'ela_level.costs_runtime': np.nan,
    'disp.ratio_mean_02': np.nan,
    'disp.ratio_mean_05': np.nan,
    'disp.ratio_mean_10': np.nan,
    'disp.ratio_mean_25': np.nan,
    'disp.ratio_median_02': np.nan,
    'disp.ratio_median_05': np.nan,
    'disp.ratio_median_10': np.nan,
    'disp.ratio_median_25': np.nan,
    'disp.diff_mean_02': np.nan,
    'disp.diff_mean_05': np.nan,
    'disp.diff_mean_10': np.nan,
    'disp.diff_mean_25': np.nan,
    'disp.diff_median_02': np.nan,
    'disp.diff_median_05': np.nan,
    'disp.diff_median_10': np.nan,
    'disp.diff_median_25': np.nan,
    'disp.costs_runtime': np.nan,
    'ic.h_max': np.nan,
    'ic.eps_s': np.nan,
    'ic.eps_max': np.nan,
    'ic.eps_ratio': np.nan,
    'ic.m0': np.nan,
    'ic.costs_runtime': np.nan,
    'nbc.nn_nb.sd_ratio': np.nan,
    'nbc.nn_nb.mean_ratio': np.nan,
    'nbc.nn_nb.cor': np.nan,
    'nbc.dist_ratio.coeff_var': np.nan,
    'nbc.nb_fitness.cor': np.nan,
    'nbc.costs_runtime': np.nan,
}

# =============================================================================


def create_ela_feature_1(data_samples: xr.Dataset, itemID: int,
                         iterations: Optional[int] = None) -> xr.DataArray:

    data_samples = data_samples.expand_dims({'itemID': [int(itemID)]})
    data_samples = data_samples.stack(problem=('realization', 'itemID'))

    ela_list = []

    for i in data_samples.problem:
        if iterations is not None:
            d = data_samples.sel(problem=i, iterations=np.arange(
                iterations)).dropna(dim='input_dim')
        else:
            d = data_samples.sel(problem=i).dropna(dim='input_dim')
        X = d.input.values
        y = d.output.values.ravel()

        if X.shape[1] > 20:
            pca = PCA(n_components=10)
            X = pca.fit_transform(X, y)

        features = deepcopy(empty_labels)

        try:
            features.update(calculate_ela_distribution(X, y))
            features.update(calculate_ela_meta(X, y))
            features.update(calculate_ela_level(X, y))
            features.update(calculate_dispersion(X, y))
            features.update(calculate_information_content(X, y))
            features.update(calculate_nbc(X, y))
        except Exception:
            pass

        # delete certain keys
        del features['ela_distr.costs_runtime']
        del features['ela_meta.costs_runtime']
        del features['ela_level.costs_runtime']
        del features['disp.costs_runtime']
        del features['ic.costs_runtime']
        del features['nbc.costs_runtime']

        ela_list.append(features)

    X = pd.DataFrame(ela_list)
    return xr.DataArray(
        X, coords={'problem': data_samples.problem,
                   'features': X.columns})


def create_ela_features(
        data_samples: xr.Dataset,
        iterations: Optional[int] = None) -> xr.DataArray:
    ela_list = []

    for i in tqdm(data_samples.problem):
        if iterations is not None:
            d = data_samples.sel(problem=i, iterations=np.arange(
                iterations)).dropna(dim='input_dim')
        else:
            d = data_samples.sel(problem=i).dropna(dim='input_dim')
        X = d.input.values
        y = d.output.values.ravel()
        ela_list.append(
            {**calculate_ela_distribution(X, y),
             **calculate_ela_meta(X, y),
             **calculate_ela_level(X, y),
             **calculate_dispersion(X, y),
             **calculate_information_content(X, y),
             **calculate_nbc(X, y)})

    X = pd.DataFrame(ela_list)
    return xr.DataArray(
        X, coords={'problem': data_samples.problem,
                   'features': X.columns})
