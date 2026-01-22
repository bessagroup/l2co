# Data Validation

## Summary

The `data_validation` experiment is a post-processing routine designed to validate the optimization data generated in the `create_optimization_dataset_new` experiment. It ensures the integrity and consistency of the data by performing the following steps:

1. Loads each optimization dynamics file (`.nc`) and checks for any `np.nan` values in the input parameters or objective function values.
2. Concatenates all the optimization dynamics of all optimizers for each task.
3. Combines all tasks into a single `xarray.Dataset`.
4. Saves the resulting dataset as `entire_dataset.odata` in the designated directory.

This process ensures that the data is clean and ready for further analysis or training.

### Budget dimension

According to the `budget` function and `schedule` strategy used, each optimization trajectory may have a different number of function. The `data_validation` experiment handles this by creating a budget coordinate with a fixed resolution (accoring to the `budget_steps` parameter) in the resulting dataset. The mean (`output_mean`), best (`output_min`) and standard deviation (`output_std`) of the population for each iteration is calculated and projected on the number of budget steps. This means that an interpolation or extrapolation is performed.

### Missing combinations

Missing combinations of schedules and tasks are handled by the `data_validation` experiment. If a task is missing for a specific schedule, it is filled with `np.nan` values. This ensures that the dataset remains consistent and can be used for training or analysis without introducing errors due to missing data. The author is aware that this may not be the best approach, but it is a practical solution given the lack of handling sparse matrices with the `xarray` library.

## Configurations

The `conf` folder contains configurations for the data validation process. Below is a summary of the available configurations:

### Parameters
| Name         | Type   | Description |
|--------------|--------|-------------|
| `path`       | `str`  | Experiment ID to locate the `f3dasm.ExpermentData` files |
| `opt_dir`    | `str`  | Path to the parent directory containing the optimization data. |
| `budget_steps` | `int` | Number of budget steps to allocate. |

## Contents of this folder

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main script to run the data validation process. |
| `config.yaml` | Configuration file for the data validation experiment. |
| `README.md` | Explanation of this experiment. |
| `slurmjob_brown.sh` | SLURM job file to run the experiment on the OSCAR cluster. |
| `outputs/` | Folder where the validated dataset will be stored. |

> The `outputs/` folder is created when the experiment has been run for the first time.

## Usage

### Before running the experiment

1. Install the `l2co` package. See [here](https://github.com/bessagroup/L2CO#getting-started) for instructions.
2. Change the `config.yaml` file to your liking. See [here](#explanation-of-configyaml-parameters) for an explanation of the parameters.

### Running the experiment on your local machine

1. Navigate to this folder and run `python main.py path=<experiment_id>`

### Running the experiment on a cluster

1. Make sure you have a `conda` environment named `l2co` with the required packages installed.
2. Navigate to this folder and submit the job:

| Cluster | Command |
|---------|---------|
| [OSCAR](https://docs.ccv.brown.edu/oscar) | `sbatch slurmjob_brown.sh path=<experiment_id>` |

## Results

The validated dataset is stored in the same folder as the experiment chosen with the `path` parameter with the name `entire_dataset.odata`. If schedules are included, they are stored as `entire_dataset_schedules.osched`.
