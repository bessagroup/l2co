import xarray as xr
from f3dasm import ExperimentData
from sklearn.model_selection import train_test_split

from l2co.content_based import (calculate_similarity_features, get_labels,
                                predict_zero)
from l2co.performance import collaborative_filtering, get_performance, query_t
from l2co.trajectory import calculate_trajectory_similarity, get_trajectory

project_dir: str = "path/to/project"
test_id = 1
T_SWITCH = 500
T_START = 30
SEED = 42

# Load data

experimentdata = ExperimentData.from_file(project_dir)

# Split train test

_train, _test = train_test_split(
    experimentdata.index, test_size=0.2, random_state=SEED)

experimentdata_train = experimentdata.select(_train)
experimentdata_test = experimentdata.select(_test)


# Get performance

ds_test_performance_all = get_performance(experimentdata=experimentdata_test)
ds_train_performance_all = get_performance(experimentdata=experimentdata_train)

# Get labels

ds_test_labels = get_labels(experimentdata_test)
ds_train_labels = get_labels(experimentdata_train)

# SELECT ONE TEST PROBLEM

test_id = 0
realization = 0
budget = 600

""" STEP 0: GET THE OPTIMIZER PERFORMANCE AT THE BUDGET"""

ds_train_performance = query_t(
    performance_dataset=ds_train_performance_all, t=budget)

""" STEP 1: PREDICT OPTIMIZER WITH FEATURES """
ds_test_label = ds_test_labels.sel(itemID=test_id)


ds_features_similarity = calculate_similarity_features(
    ds_single=ds_test_label, ds_features=ds_train_labels)


prediction = predict_zero(ds_train_performance, ds_features_similarity)

""" STEP 2: Get the first part of the prediction optimizer"""

# split the string prediction at the + sign and take the first part
prediction = prediction.split('+')[0]

""" STEP 3: GET THE TRAJECTORY OF THE PREDICTED OPTIMIZER """

ds_test_trajectory = get_trajectory(experimentdata=experimentdata_test,
                                    optimizer=prediction,
                                    t_switch=T_SWITCH, t_start=T_START).sel(
                                        itemID=test_id,
                                        realization=realization,
                                        drop=True)

ds_train_trajectories = get_trajectory(experimentdata=experimentdata_train,
                                       optimizer=prediction,
                                       t_switch=T_SWITCH, t_start=T_START)

ds_trajectory_similarity = calculate_trajectory_similarity(
    ds_single=ds_test_trajectory, ds_similarity=ds_train_trajectories)

prediction = predict_one(ds_train_performance, ds_trajectory_similarity)
