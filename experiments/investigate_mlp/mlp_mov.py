#                                                                       Modules
# =============================================================================

# Standard
import os

# Third-party
import f3dasm
import matplotlib.pyplot as plt
import pandas as pd
from config import Config
from hydra.utils import get_original_cwd
from sklearn.neural_network import MLPClassifier

from l2co._logging import logger
# Local
from l2co.rollout import create_recommendation_results
from l2co._legacy.strategy import create_strategy, mlp_strategy

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# def convert_config_to_input(config: Config, function_name: str, dimensionality: str,
#                             noise: str, seed: int) -> List[dict]:
#     ...


def execute(config: Config, jobnumber: int, solver: str, activation: str, layers: int, neurons: int) -> str:

    filename = f"s-{solver}a-{activation}l-{layers}n-{neurons}"

    # '/media/martin/MARTIN USB/data/520508.hpc06.hpc'
    folder_train = os.path.join(get_original_cwd(), config.files.train_dirname_raw)
    # '/media/martin/MARTIN USB/data/520412.hpc06.hpc'
    folder_test = os.path.join(get_original_cwd(), config.files.test_dirname_raw)

    # recommender = pd.read_csv('/home/martin/Documents/functions_optimizer_map.csv', delimiter='\t')
    # data_train = f3dasm.ExperimentData.from_csv(filename=f"{folder_train}/benchmarkfunctions")
    data_test = f3dasm.ExperimentData.from_csv(filename=f"{folder_test}/benchmarkfunctions")
    # features_functions_train = pd.read_csv(f"{folder_train}/recommender/totalfeatures_functions.csv", index_col=0)
    features_functions_test = pd.read_csv(f"{folder_test}/recommender/totalfeatures_functions.csv", index_col=0)

    unique_features = {'convex': 0,
                       'differentiable': 0,
                       'high_dimensional': 0,
                       'low_dimensional': 0,
                       'medium_dimensional': 0,
                       'multimodal': 0,
                       'noiseless': 0,
                       'noisy': 0,
                       'parametric': 0,
                       'randomized_term': 0,
                       'seperable': 0}

    mlp_input_train = pd.read_csv(f"{folder_train}/mlp_input.csv", index_col=0)
    best_mov_test = pd.read_csv(f"{folder_test}/mlp_output.csv", index_col=0)['mov_best']
    best_mov_train = pd.read_csv(f"{folder_train}/mlp_output.csv", index_col=0)['mov_best1']

    clf = MLPClassifier(solver=solver, activation=activation,
                        alpha=config.mlp.alpha, hidden_layer_sizes=(layers, neurons),
                        random_state=config.mlp.seed)

    clf.fit(mlp_input_train, best_mov_train)

    mlp_dict = {'files_path': folder_test, 'experimentdata': data_test,
                'strategy_determiner': mlp_strategy, 'label': f'{filename}',
                'load_dataset': False, 'features_functions': features_functions_test,
                'unique_features': unique_features, 'classifier': clf}

    ds_mov_mlp = create_strategy(**mlp_dict)

    ds_mov_best = best_mov_test

    df_mov = create_recommendation_results(files_path=folder_test, experimentdata=data_test, strategies=[
                                           ds_mov_mlp, ds_mov_best])

    df_mov.to_csv(f"./{config.files.dirname_mov}/{filename}.csv")

    result: float = df_mov[filename].iloc[-1] / len(ds_mov_best)

    logger.info(f"Finished job and saved filename {filename}")

    # Plotting
    fig, ax = plt.figure(figsize=[10, 8]), plt.axes()
    ax.plot(df_mov / len(ds_mov_best), label=df_mov.columns)
    fig.legend(loc='upper left')
    fig.savefig(f'./{config.files.dirname_mov}/{filename}.png', dpi=300)

    return filename, result
