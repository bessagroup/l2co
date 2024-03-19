#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from time import perf_counter

# Third-party
import hydra
import matplotlib.pyplot as plt
import xarray as xr
from f3dasm import ExperimentData
from hydra.utils import instantiate
from sklearn.ensemble import GradientBoostingClassifier

from l2co import LSTMMultiOutputClassifier
from l2co import logger as l2co_logger
from l2co.data import DataLoader
from l2co.preprocessing import preprocessing_factory
from l2co.trainer import Trainer

# from sklearn.model_selection import train_test_split

# from l2co.content_based import (calculate_similarity_features, get_labels,
# predict_zero)
# from l2co.performance import collaborative_filtering, get_performance, query_t
# from l2co.trajectory import calculate_trajectory_similarity, get_trajectory

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@hydra.main(config_path=".", config_name="config")
def main(config):

    # Construct the classifiers
    stage0_clf = instantiate(config.stage0_clf)
    # stage0_clf = GradientBoostingClassifier(**config.stage0_clf)
    stage1_clf = LSTMMultiOutputClassifier(**config.stage1_clf)

    preprocessing_step = preprocessing_factory(
        feature_types=config.feature_types)

    training_project_dir = Path(config.data.training_project_dir)
    testing_project_dir = Path(
        config.data.testing_project_dir) if config.data.testing_project_dir is not None else None

    experimentdata_train = ExperimentData.from_file(training_project_dir)

    _, df_output = experimentdata_train.to_pandas()
    experimentdata_train.select(
        df_output.index[~(df_output == 'invalid').any(axis=1)])

    data_train = xr.open_dataset(
        training_project_dir / "entire_dataset.nc", chunks='auto')
    data_samples_train = xr.open_dataset(
        training_project_dir / 'entire_dataset_samples.nc', chunks='auto')

    t = perf_counter()
    d_train = DataLoader.from_preprocessed_files(
        config=0,
        experimentdata=experimentdata_train,
        data=data_train,
        data_samples=data_samples_train)

    l2co_logger.info(
        f"Loaded training data from {training_project_dir} (took {perf_counter() - t} seconds")

    if testing_project_dir is None:
        t = perf_counter()
        d_train, d_test = d_train.train_test_split(
            test_size=config.test_size, random_state=config.seed)

        l2co_logger.info((
            f"Split training data into training and testing data )"
            f"(with a test size of {config.test_size} and seed {config.seed} (took {perf_counter() - t} seconds")
        )
    else:
        experimentdata_test = ExperimentData.from_file(testing_project_dir)

        _, df_output = experimentdata_test.to_pandas()
        experimentdata_test.select(
            df_output.index[~(df_output == 'invalid').any(axis=1)])

        data_test = xr.open_dataset(
            testing_project_dir / "entire_dataset.nc", chunks='auto')
        data_samples_test = xr.open_dataset(
            testing_project_dir / 'entire_dataset_samples.nc', chunks='auto')
        t = perf_counter()
        d_test = DataLoader.from_preprocessed_files(config=0,
                                                    experimentdata=experimentdata_test,
                                                    data=data_test,
                                                    data_samples=data_samples_test)

        l2co_logger.info(
            f"Loaded testing data from {testing_project_dir} (took {perf_counter() - t} seconds")

    trainer = Trainer(
        data_train=d_train,
        data_test=d_test,
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

    print("Created trainer")

    trainer.train_stage0()

    t = perf_counter()
    if config.stage1_clf.path is not None:
        trainer.load_stage1(config.stage1_clf.path)

    else:
        trainer.train_stage1(save_model=True)

    l2co_logger.info(f"Trained trainer (took {perf_counter() - t} seconds")

    t = perf_counter()
    trainer.predict()
    l2co_logger.info(f"Predicted trainer (took {perf_counter() - t} seconds")

    trainer.create_performance_profile(strategies=[trainer.stage1_prediction])

    fig, axs = plt.subplots(
        2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))

    _ = trainer.plot_performance_profile(
        strategies=[trainer.stage1_prediction], strategy_names=['trainer_stage1'], axes=axs)

    trainer.performance_profile.to_netcdf('performance_profile.nc')

    fig.savefig('performance_profile.png', dpi=300, bbox_inches='tight')
    # experimentdata_test = ExperimentData.from_file(
    #     Path(config.data.test_project_dir))

    # l2co_logger.info(
    #     f"Loaded test data from {experimentdata_test.project_dir.stem}")

    # SEED = config.seed
    # T_SWITCH = config.t_switch
    # T_START = config.t_start

    # # Get performance
    # ds_train_performance_all = get_performance(
    #     experimentdata=experimentdata_train)

    # # Get labels

    # ds_test_labels = get_labels(experimentdata_test)
    # ds_train_labels = get_labels(experimentdata_train)

    # # SELECT ONE TEST PROBLEM

    # test_id = 0
    # realization = 0
    # budget = 529

    # """ STEP 0: GET THE OPTIMIZER PERFORMANCE AT THE BUDGET"""

    # ds_train_performance = query_t(
    #     performance_dataset=ds_train_performance_all, t=budget)

    # """ STEP 1: PREDICT OPTIMIZER WITH FEATURES """
    # ds_test_label = ds_test_labels.sel(itemID=test_id)

    # ds_features_similarity = calculate_similarity_features(
    #     ds_single=ds_test_label, ds_features=ds_train_labels)

    # prediction = predict_zero(ds_train_performance, ds_features_similarity)

    # l2co_logger.info(f"Predicted optimizer for stage 0: {prediction}")
    # """ STEP 2: Get the first part of the prediction optimizer"""

    # # split the string prediction at the + sign and take the first part
    # prediction = prediction.split('+')[0]

    # """ STEP 3: GET THE TRAJECTORY OF THE PREDICTED OPTIMIZER """

    # ds_test_trajectory = get_trajectory(experimentdata=experimentdata_test,
    #                                     optimizer=prediction,
    #                                     t_switch=T_SWITCH, t_start=T_START).sel(
    #                                         itemID=test_id,
    #                                         realization=realization,
    #                                         drop=True)

    # ds_train_trajectories = get_trajectory(experimentdata=experimentdata_train,
    #                                        optimizer=prediction,
    #                                        t_switch=T_SWITCH, t_start=T_START)

    # ds_trajectory_similarity = calculate_trajectory_similarity(
    #     ds_single=ds_test_trajectory, ds_similarity=ds_train_trajectories)

    # prediction = predict_one(ds_train_performance, ds_trajectory_similarity)

# 3


# project_dir: str = "path/to/project"
# test_id = 1
# T_SWITCH = 500
# T_START = 30
# SEED = 42

# # Load data

# experimentdata = ExperimentData.from_file(project_dir)

# # Split train test

# _train, _test = train_test_split(
#     experimentdata.index, test_size=0.2, random_state=SEED)

# experimentdata_train = experimentdata.select(_train)
# experimentdata_test = experimentdata.select(_test)


# # Get performance

# ds_test_performance_all = get_performance(experimentdata=experimentdata_test)
# ds_train_performance_all = get_performance(experimentdata=experimentdata_train)

# # Get labels

# ds_test_labels = get_labels(experimentdata_test)
# ds_train_labels = get_labels(experimentdata_train)

# # SELECT ONE TEST PROBLEM

# test_id = 0
# realization = 0
# budget = 600

# """ STEP 0: GET THE OPTIMIZER PERFORMANCE AT THE BUDGET"""

# ds_train_performance = query_t(
#     performance_dataset=ds_train_performance_all, t=budget)

# """ STEP 1: PREDICT OPTIMIZER WITH FEATURES """
# ds_test_label = ds_test_labels.sel(itemID=test_id)


# ds_features_similarity = calculate_similarity_features(
#     ds_single=ds_test_label, ds_features=ds_train_labels)


# prediction = predict_zero(ds_train_performance, ds_features_similarity)

# """ STEP 2: Get the first part of the prediction optimizer"""

# # split the string prediction at the + sign and take the first part
# prediction = prediction.split('+')[0]

# """ STEP 3: GET THE TRAJECTORY OF THE PREDICTED OPTIMIZER """

# ds_test_trajectory = get_trajectory(experimentdata=experimentdata_test,
#                                     optimizer=prediction,
#                                     t_switch=T_SWITCH, t_start=T_START).sel(
#                                         itemID=test_id,
#                                         realization=realization,
#                                         drop=True)
# ds_train_trajectories = get_trajectory(experimentdata=experimentdata_train,
#                                        optimizer=prediction,
#                                        t_switch=T_SWITCH, t_start=T_START)
# ds_trajectory_similarity = calculate_trajectory_similarity(
#     ds_single=ds_test_trajectory, ds_similarity=ds_train_trajectories)
# # prediction = predict_one(ds_train_performance, ds_trajectory_similarity)
if __name__ == "__main__":
    main()
