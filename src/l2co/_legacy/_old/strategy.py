"""Functions that"""
#                                                                       Modules
# =============================================================================

# Standard

import logging
import random
from copy import copy
from itertools import combinations
from typing import Dict, Protocol, Union

import f3dasm
# Third-party core
import pandas as pd
import xarray as xr
from f3dasm.design import ExperimentData
from sklearn.neural_network import MLPClassifier

# Local
from .._logging import logger
from ..performance import area_under_curve, margin_of_victory
from .rollout import determine_loser, determine_winner
from ..saving import load_netcdf

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


class Strategy(Protocol):
    def __call__(self, data: xr.Dataset, options: Union[None, dict]) -> str:
        """Strategy protocol function that determines the strategy for each problem

        Parameters
        ----------
        data
            raw data in the form of an xarray dataset

        Returns
        -------
            label of optimizer that is chosen for each problem
        """
        ...


def create_strategy(files_path: str, experimentdata: ExperimentData,
                    strategy_determiner: Strategy, label: str,
                    load_dataset: bool = True, parallelization: bool = False,
                    options: Union[None, dict] = None) -> pd.Series:
    """Create a strategy for each problem in the experimentdata"""
    recommender = f3dasm.experiment.run_operation_on_experiments(
        experimentdata, _create_one_strategy, parallel=parallelization, experimentdata=experimentdata,
        files_path=files_path, strategy_determiner=strategy_determiner, label=label,
        load_dataset=load_dataset, options=options)

    strategy_series = pd.Series(recommender, index=experimentdata.data.index, name=label)

    logger.info(f"Created strategy series for {label}")
    return strategy_series


def _create_one_strategy(index: int, value_input: dict,
                         value_output: dict, experimentdata: ExperimentData, files_path: str,
                         strategy_determiner: Strategy, label: str,
                         load_dataset: bool, options: Union[None, dict]) -> str:
    i = experimentdata.data.index[index]

    # if options is None, create an empty dict
    if options is None:
        options = {}

    # if options is a dict, update the key 'index' with the current i
    options.update({'index': i})

    if load_dataset:
        experimentdata: xr.Dataset = load_netcdf(f"{files_path}/raw/{value_output['status']}.cd")
    else:
        experimentdata = 0

    logging.debug(f"Created strategy for {i}")

    return strategy_determiner(experimentdata, options)


def best_mov_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = margin_of_victory(data)
    return determine_winner(mov)


def worst_mov_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = margin_of_victory(data)
    return determine_loser(mov)


def best_area_under_curve_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = area_under_curve(data)
    return determine_winner(mov)


def worst_area_under_curve_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    mov = area_under_curve(data)
    return determine_loser(mov)


def random_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    return random.choice(list(data.optimizer.values))


def lightfm_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    index: int = options['index']
    df_predictions: pd.DataFrame = options['df_predictions']

    optimizer_mapping = {i+1: opt for i, opt in enumerate(data.optimizer.values)}
    optimizer_number = df_predictions[df_predictions['itemID'] == index].sort_values(
        by=['prediction'], ascending=False).iloc[0]['userID']
    return optimizer_mapping[int(optimizer_number)]


def lightfm_strategy_reverse(data: xr.Dataset, options: Union[None, dict]) -> str:
    index: int = options['index']
    df_predictions: pd.DataFrame = options['df_predictions']

    optimizer_mapping = {i+1: opt for i, opt in enumerate(data.optimizer.values)}
    optimizer_number = df_predictions[df_predictions['userID'] == index].sort_values(
        by=['prediction'], ascending=True).iloc[0]['itemID']
    return optimizer_mapping[int(optimizer_number)]


def manual_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:
    index: int = options['index']
    features_functions: pd.DataFrame = options['features_functions']
    recommender_dict: dict = options['recommender_dict']

    features = features_functions[features_functions['itemID'] == index]['genres'].values[0].split(',')
    f_combinations = list(combinations(features, 1))
    f_combinations.extend(list(combinations(features, 2)))
    f = [",".join(f_combinations[ii]) for ii in range(len(f_combinations))]
    f_opt = [recommender_dict[feature] for feature in f]
    return max(set(f_opt), key=f_opt.count)


def mlp_strategy(data: xr.Dataset, options: Union[None, dict]) -> str:

    index: int = options['index']
    features_functions: pd.DataFrame = options['features_functions']
    unique_features: dict = options['unique_features']
    classifier: MLPClassifier = options['classifier']
    # # Reverse map
    # reverse_mapping = {value: key for key, value in output_map.items()}

    features = features_functions[features_functions['itemID'] == index]['genres'].values[0].split(',')

    features_dict = copy(unique_features)
    for value in features:
        features_dict[value] = 1

    mlp_input = pd.DataFrame([features_dict]).to_numpy()

    mlp_output = classifier.predict(mlp_input)
    return mlp_output  # reverse_mapping[mlp_output.item()]


# create a function that returns the similarity between two indices
def _similarity(row_i: pd.Series, row_j: pd.Series, min_max: Dict, p: int = 2):

    # # get the two rows of dg
    # row_i = dg.loc[i]
    # row_j = dg.loc[j]

    similarity = 0
    for feature_index, _ in enumerate(row_i):
        n, m = min_max[row_i.keys()[feature_index]]
        s = (1-(abs(row_i[feature_index] - row_j[feature_index])/(m - n)))**p
        similarity += s

    # divide by the number of features
    similarity /= len(row_i)

    return similarity


def _calculate_similarity_matrix(new_entry: pd.DataFrame, training_set: pd.DataFrame) -> pd.DataFrame:
    # find the lowest and highest value of each column and save in a dict
    min_max = {}
    for col in training_set.columns:
        min_max[col] = [training_set[col].min(), training_set[col].max()]

    # calculate the similarity of the first row with all other rows and put them in a dataframe
    similarity_df = pd.DataFrame(columns=['index', 'similarity'])
    for j in training_set.index:

        similarity_df.loc[j] = [j, _similarity(row_i=new_entry, row_j=training_set.loc[j], min_max=min_max)]

        # cast index column to int
    similarity_df['index'] = similarity_df['index'].astype(int)
    similarity_df.sort_values(by=['similarity'], ascending=False, inplace=True)
    return similarity_df


def _calculate_majority_vote(similarity_df, df_interactions, threshold: float = 0.8):
    # add a new column to similarity_df called rating
    similarity_df['rating'] = -2

    # for each itemID in similarity_df, find the rating in highest_ratings and save it to similarity_df
    for itemID in similarity_df['index']:
        similarity_df.loc[itemID, 'rating'] = df_interactions[df_interactions['itemID'] == itemID]['rating'].argmax()

    # take all the rows of similarity_df where the rating is not -1
    similarity_df = similarity_df[similarity_df['rating'] != -1]

    # Imbalanced dataset
    # from similarity_df, from each unique rating, sample 258 rows
    least_entries = min(similarity_df['rating'].value_counts())
    similarity_df_balanced = similarity_df.groupby('rating').apply(
        lambda x: x.sample(least_entries)).reset_index(drop=True)

    # take all the rows where similarity > threshold
    similarity_df_filtered = similarity_df_balanced[similarity_df_balanced['similarity'] >= threshold]

    # weighted sum of each of the unique ratings with weight similarity
    weighted_ratings = {i: 0 for i in similarity_df_filtered['rating'].unique()}
    for index, row in similarity_df_filtered.iterrows():
        weighted_ratings[row['rating']] += row['similarity']

    # take argmax of weighted_ratings
    return max(weighted_ratings, key=weighted_ratings.get)


def contentbasedrecommender(data: xr.Dataset, options: Union[None, dict]) -> str:
    optimizer_mapping = {i: opt for i, opt in enumerate(data.optimizer.values)}
    features = pd.DataFrame(columns=['dim', 'noise', 'convex', 'separable',
                            'differentiable', 'multimodal', 'randomized_term', 'parametric'])

    f = f3dasm.datageneration.functions.find_function(data.function_name)

    if data.function_noise == 0.0:
        noise = 0
    else:
        noise = 1

    features.loc[0] = [data.function_dimensionality, noise, f.convex, f.separable,
                       f.differentiable, f.multimodal, f.randomized_term, f.parametric]
    features = features.astype(int)

    optimizer_number = _calculate_majority_vote(_calculate_similarity_matrix(
        features.loc[0], options['df_training_features']), options['df_interactions'])
    return optimizer_mapping[int(optimizer_number)]
