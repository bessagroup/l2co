#                                                                       Modules
# =============================================================================


# Standard
from copy import deepcopy
from typing import List

# Third-party
import f3dasm
import pandas as pd
import xarray as xr
from config import Config
from f3dasm import ExperimentSample
from f3dasm.datageneration import DataGenerator
from SALib.sample import latin

# Local
from l2co.create_recommender_dataset import RecommenderDatasetCreator
from l2co.run_optimization import experiment_two_optimizers

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class BenchmarkOptimizationRoutine(DataGenerator):
    def __init__(self, config: Config):
        self.config = config

    def execute(self) -> ExperimentSample:
        function_name = self.experiment_sample.get('function_name')
        dimensionality = self.experiment_sample.get('dimensionality')
        budget = int(self.experiment_sample.get('budget'))
        noise = self.experiment_sample.get('noise')
        seed = int(self.experiment_sample.get('seed'))

        optimizers = [opt_config['name'] for
                      opt_config in self.config.optimization.optimizers]

        results_xarray = experiment_two_optimizers(
            optimizers=optimizers,
            data_generator=function_name,
            sampler=self.config.optimization.sampler_name,
            realizations=self.config.optimization.realizations,
            n_samples=self.config.optimization.number_of_samples,
            parallelization=self.config.optimization.parallelization,
            seed=seed, iterations=budget, dimensionality=dimensionality,
            noise=noise)

        self.experiment_sample.store(
            object=results_xarray,
            name=self.config.files.dirname_raw, to_disk=True)

        if not self.config.recommender.create_recommender_dataset:
            return

        # Augment the designs
        augmented_designs = augment_designs(
            experiment_sample=self.experiment_sample,
            n_samples=self.config.recommender.n_augmented_samples,
            upper_bound_budget=budget,
            lower_bound_budget=self.config.recommender.lower_bound_budget)

        # Create the recommender dataset
        features_matrix_list: List[pd.DataFrame] = []
        # mov_list: List[pd.DataFrame] = []
        perf_profile_list: List[xr.DataArray] = []
        task_numbers: List[int] = []
        samples_y: List[xr.DataArray] = []
        perf_per_optimizer_list: List[xr.DataArray] = []

        for index, augmented_design in enumerate(augmented_designs):
            budget: int = int(augmented_design.get('budget'))
            task_number: int = int(
                self.experiment_sample.
                job_number * (
                    self.config.recommender.n_augmented_samples + 1
                                   ) + index)  # NOQA
            task_numbers.append(task_number)
            samples_y.append(results_xarray['output'].isel(iterations=slice(
                0, results_xarray.attrs['number_of_samples']), optimizer=0))
            recommender_dataset_creator = RecommenderDatasetCreator(
                results=results_xarray, jobnumber=task_number)

            features_matrix_list.append(
                recommender_dataset_creator.
                create_recommender_data_features_functions_csv(
                    budget=int(augmented_design.get('budget'))))

            # mov_list.append(margin_of_victory(
            #     results_xarray, budget=budget))

            perf_profile_list.append(perf_profile(
                results_xarray, budget=budget, itemID=task_number))
            perf_per_optimizer_list.append(
                create_performance_metric(results_xarray, budget=budget))

        df_features_matrix = pd.concat(features_matrix_list, ignore_index=True)

        # Create the xarray dataset
        df_features_matrix.set_index('itemID', inplace=True)
        xarr_all = xr.Dataset(df_features_matrix)

        xr_samples_y = xr.concat(
            samples_y, dim=xr.DataArray(task_numbers, dims='itemID'))
        xr_perf_per_optimizer = xr.concat(
            perf_per_optimizer_list, dim=xr.DataArray(
                task_numbers, dims='itemID'))
        # xarr_all['mov'] = c
        xarr_all['samples_output'] = xr_samples_y
        xarr_all['ranking'] = xr_perf_per_optimizer
        xarr_all['perf_profile'] = xr.concat(perf_profile_list, dim='itemID')

        self.experiment_sample.store(
            object=xarr_all, name=self.config.files.dirname_movxarr,
            to_disk=True)

        return


def augment_designs(
        experiment_sample: f3dasm.ExperimentSample, n_samples: int,
        lower_bound_budget: int,
        upper_bound_budget: int) -> List[ExperimentSample]:

    experiment_samples = []

    salib_problem = {
        'num_vars': 1,
        'names': ['budget'],
        'bounds': [[lower_bound_budget, upper_bound_budget]],
    }

    # add the original design
    design_0 = deepcopy(experiment_sample)
    design_0._jobnumber = design_0.job_number * (n_samples + 1)
    experiment_samples.append(design_0)

    samples = latin.sample(salib_problem, n_samples).flatten()

    for index, sample in enumerate(samples):
        design_ = deepcopy(design_0)
        design_._jobnumber = design_0.job_number + index + 1
        design_._dict_input['budget'] = int(sample)
        experiment_samples.append(design_)

    return experiment_samples


def perf_profile(data, budget: int, itemID: int) -> xr.DataArray:
    x: xr.DataArray = data['output'].sel(
        iterations=range(0, budget)).min(dim='iterations')

    # standard scale x
    x = (x - x.mean()) / x.std()
    x = x - x.min() + 1.0

    # add a new dimension named itemID and set to value itemID
    x = x.expand_dims({'itemID': [itemID]})

    return x
