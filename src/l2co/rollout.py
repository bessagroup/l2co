# Standard
import glob
import logging
from copy import copy
from os import path
from typing import Dict, List, Union

import pandas as pd

from l2co.margin_of_victory import margin_of_victory
from l2co.saving import load_results

# Third-party


def create_recommendation_results(path_of_files: str, recommendations: Dict[int, int],
                                  selection: Union[None, List[str]] = None) -> pd.Series:
    total_mov = None

    if selection is None:
        filelist: List[str] = glob.glob(path.join(path_of_files, '*.json'))

    else:
        filelist: List[str] = [glob.glob(path.join(path_of_files, f'*{_id}*.json'))[0] for _id in selection]

    for file in filelist:
        count_continue: int = 0
        results = load_results(file)
        logging.info(f"Loaded file: {file}")

        optimizer_number: dict = {index+1: r.optimizer.get_name() for index, r in enumerate(results)}
        mov = margin_of_victory(results)

        # Select recommendation
        function_id = results[0].seeds[0]
        try:
            recommendation_id = recommendations[function_id]
        except KeyError:  # No recommendation found
            count_continue += 1
            logging.info(f"Key {function_id} not found! Continue (count={count_continue})")
            continue

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
