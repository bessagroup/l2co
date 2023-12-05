# Recommend optimizer

## Contents of this folder

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main script to run the experiment |
| `config.yaml` | Configuration file for the experiment |
| `README.md` | Explanation of this experiment |
| `outputs/` | Folder with the results of running this experiment |

> The `outputs/` folder is created when the experiment has been run for the first time.

## Summary

> Note: This experiment is not yet fully implemented and remains a work-in-progress!.

## Usage

1. Install the `l2co` package. See [here](https://github.com/bessagroup/L2CO#getting-started) for instructions.
2. Navigate to this folder and run `python main.py`

## Results

Results are stored in a newly created `outputs` folder, with a subdirectory
indicating the current date (e.g. `2023-11-06`) and time (e.g. `13-50-14`).

The following subdirectories are created:

* `experiment_data`: Contains the input, output, domain and jobs to construct the `f3dasm.ExperimentData` object.
* `.hydra`: Contains the `config.yaml` file used to run the experiment.

Lastly, a log file `main` is created.

The folder structure is as follows:

```
outputs/
└── 2023-11-06/
    └── 13-50-14/
        ├── .hydra/
        ├── experiment_data/
        │   ├── domain.pkl
        │   ├── input.csv
        │   ├── output.csv
        │   └── jobs.pkl
        └── main.log
```


## Explanation of `config.yaml` parameters

| Name  | Type | Description  |
|-------|------|--------------|
| seed  | `int`  | Seed for the pseudo-random number generator |
| t_switch  | `int`  | Iteration number of the switching point |
| t_start  | `int`  | Iteration number of the start of the first optimizer |

### Data
| Name   | Type | Description  |
|--------|------|--------------|
| training_project_dir  | `str`  | Path to the `ExperimentData` project directory for the training data |
| test_project_dir | `str`  | Path to the `ExperimentData` project directory for the testing data |

### HPC
| Name   | Type | Description  |
|--------|------|--------------|
| jobid  | `int`  | Job ID of the array-job, automatically overwritten by scheduler bash script |

### Log Level
| Name      | Type | Description     |
|-----------|------|-----------------|
| log_level | `int`  | [Log level value](https://docs.python.org/3/library/logging.html#logging-levels) |
