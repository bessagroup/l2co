# Standard
import glob
import json
from copy import copy
from os import path
from typing import Dict, List, Union
import logging
# Third-party
import numpy as np
import pandas as pd
from f3dasm import create_optimizationresult_from_json
from f3dasm.run_optimization import OptimizationResult, calculate_mean_std
from sklearn import preprocessing


def margin_of_victory(results: List[OptimizationResult]) -> pd.DataFrame:
    """Calculate the margin of victory of optimizationresults

    Parameters
    ----------
    results
        OptimizationResults object

    Returns
    -------
        DataFrame containing the algorithms and the margin of victory
    """

    # Create df with all results
    df = pd.concat([calculate_mean_std(results[i])[0] for i, _ in enumerate(results)], axis=1)

    # Change columnnames
    optimizer_names = [results[i].optimizer.get_name() for i, _ in enumerate(results)]
    df.columns = optimizer_names

    # Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    # Reshape to 1D array
    df_numpy = df.values  # returns a numpy array
    df_numpy_reshaped = df_numpy.reshape(-1, 1)

    x_scaled = min_max_scaler.fit_transform(df_numpy_reshaped)

    # Transform back
    x_scaled = x_scaled.reshape(df_numpy.shape)
    df = pd.DataFrame(x_scaled)
    df.columns = optimizer_names

    # Calculate margin of victory
    mov = []
    for name in optimizer_names:
        df_dropped = df.drop(name, axis=1)
        mov.append(df_dropped.min(axis=1) - df[name])

    # Create df with all MoV
    df_margin_of_victory = pd.concat(mov, axis=1)

    # Change columnnames
    df_margin_of_victory.columns = optimizer_names

    return df_margin_of_victory


def load_results(filepath: str) -> List[OptimizationResult]:
    # Load results
    with open(filepath) as f:
        json_string = json.load(f)

    json_results = json.loads(json_string)
    results = [create_optimizationresult_from_json(r) for r in json_results]
    return results


def create_recommendation_results(path_of_files: str, recommendations: Dict[int, int],
                                  selection: Union[None, List[str]] = None) -> pd.Series:
    total_mov = None

    if selection is None:
        filelist: List[str] = glob.glob(path.join(path_of_files, '*.json'))

    else:
        filelist: List[str] = [glob.glob(path.join(path_of_files, f'*{_id}*.json'))[0] for _id in selection]

    for file in filelist:
        results = load_results(file)
        logging.inf(f"Loaded file: {file}")

        optimizer_number: dict = {index: r.optimizer.get_name() for index, r in enumerate(results)}
        mov = margin_of_victory(results)

        # Select recommendation
        function_id = results[0].seeds[0]
        recommendation_id = recommendations[function_id]
        recommendation = optimizer_number[recommendation_id]

        mov['Recommendation'] = mov[recommendation]

        if total_mov is None:
            total_mov = mov
            total_temp = copy(total_mov)

        else:
            total_temp = copy(total_mov)
            total_mov = sum((total_mov, mov))

        if total_mov.isnull().values.any():  # Check for NaN values of mov
            total_mov = total_temp

    return total_mov


def create_individual_optimizers_results(path_of_files: str, selection: Union[None, List[str]] = None) -> pd.DataFrame:
    total_mov = None

    if selection is None:
        filelist: List[str] = glob.glob(path.join(path_of_files, '*.json'))

    else:
        filelist: List[str] = [glob.glob(path.join(path_of_files, f'*{_id}*.json'))[0] for _id in selection]

    for file in filelist:
        results = load_results(file)

        mov = margin_of_victory(results)

        if total_mov is None:
            total_mov = mov
        else:
            total_mov += mov

    return total_mov


def create_random_strategy(path_of_files: str, selection: Union[None, List[str]] = None) -> Dict[int, int]:
    strategy_dict = {}

    if selection is None:
        filelist: List[str] = glob.glob(path.join(path_of_files, '*.json'))

    else:
        filelist: List[str] = [glob.glob(path.join(path_of_files, f'*{_id}*.json'))[0] for _id in selection]

    for file in filelist:
        results = load_results(file)
        optimizer_number: dict = {index: r.optimizer.get_name() for index, r in enumerate(results)}
        function_id = results[0].seeds[0]
        strategy_dict[function_id] = np.random.choice(list(optimizer_number.keys()))
    return strategy_dict
