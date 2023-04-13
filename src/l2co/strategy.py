import glob
from os import path
from typing import Dict, List, Union

import numpy as np

from l2co.saving import load_results


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
