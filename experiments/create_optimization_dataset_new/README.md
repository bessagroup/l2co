# Create trainingdata two optimizers


## Summary

The `main.py` script orchestrates the creation of a dataset for training and evaluating optimization algorithms. It performs the following steps:

1. **Configuration Loading**: Reads the experiment configuration using Hydra, which defines tasks, optimizers, schedules, budgets, and samplers.
2. **Task Generation**: Creates optimization tasks based on the specified configuration, including sampling parameters and generating task instances.
3. **Schedules Generation**: Defines how the optimization algorithms are scheduled, including the combination of optimizers and their execution order.
4. **Optimization Execution**: Runs the optimization schedules on the generated tasks, using the defined optimizers and budgets.
5. **Data Collection**: Collects and stores the results of the optimization process, including input parameters, outputs, and metadata.
6. **Output Management**: Saves the results as NetCDF files within the `outputs/` directory, ensuring reproducibility and traceability.

The script supports execution on local machines and HPC clusters, making it versatile for various computational setups.

## Configurations

In order to run the script, we need to define the components of the experiment in a configuration file. The configuration is done using [Hydra](https://hydra.cc/docs/intro/), which allows us to easily manage and override configurations:

- `tasks`: Defines the optimization tasks to be solved.
- `optimizers`: Defines the optimization algorithms to be used in the experiment.
- `schedules`: Defines how the optimization algorithms are scheduled during the optimization process.
- `budgets`: Defines the budget for each optimization task.
- `samplers`: Defines how the initial input parameters of the optimization tasks are sampled.

The `conf` folder contains various configurations for tasks, schedules, samplers, optimizers, and budgets. Below is a summary of the available configurations:

### Tasks

The configurations in the `conf/tasks/` folder define the optimization problems to be solved. Each configuration specifies how a distribution is sampled of optimization tasks, and which parameters can be altered.

Creating the tasks is a multi-step process that involves: 
* first we create *a parameterized domain* that contains the parameters of the optimization tasks,
* then we *sample* the parameters from the domain using a sampler,
* and finally, we *generate* the optimization tasks using a data generator function.

| Name             | Description                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------|
| `experimentdata` | Contains a parameterized domain for creating the optimization tasks.                           |
| `sampler`        | Determines the sampling strategy to create the optimization tasks.                             |
| `data_generator` | Defines how the optimization tasks are generated from domain parameters to a `Task` object.    |
| `task_kwargs`    | A dictionary of keyword arguments passed additionally to the `data_generator` function.        |

The following task configurations are available:
- `benchmark.yaml`: General benchmark functions.
- `benchmark_named.yaml`: Named benchmark functions.
- `cec2010.yaml`, `cec2013.yaml`, `cec2014.yaml`, `cec2015.yaml`, `cec2017.yaml`, `cec2019.yaml`: Configurations for CEC benchmark functions from different years.
- `gaussian_classification.yaml`: Gaussian classification tasks.
- `gaussian_meta.yaml`: Gaussian meta-learning tasks.
- `mnist1d.yaml`: MNIST-1D tasks.
- `pde.yaml`: Partial Differential Equation tasks.
- `quadratic.yaml`: Quadratic tasks.
- `spirals.yaml`: Spiral tasks.
- `continue.yaml`: Continue from an existing task.

### Optimizers

The configurations in the `conf/optimizers/` folder define the optimization algorithms to be used in the experiment. Each configuration specifies a list of optimizers that can be used to solve the optimization tasks:

Each list entry can have the following parameters:

| Name             | Description                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------|
| `optimizer`      | String name of the optimizer to use.                                                           |
| `hyperparameters`| A dictionary with the hyperparameters of the chosen optimizer.                                 |

The following optimizer configurations are available:
- `standard.yaml`: Standard optimizers like CMA-ES, Adam, PSO, and L-BFGS-B.
- `small.yaml`: A smaller set of optimizers.
- `pde.yaml`: Optimizers tailored for PDE tasks.
- `l2co.yaml`: L2CO optimizer configuration.

### Schedules

The configurations in the `conf/schedules/` folder define how the optimization algorithms are scheduled during the optimization process. Each configuration specifies how the optimizers are combined and when to switch between them.

| Name              | Description                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| `data_generator`  | Defines how the schedule is generated from the `schedules`.                                                   |
| `schedules`       | A list of schedules to be used in the experiment. You can mention `${optimizers}` to use the optimizers defined in the `optimizers` configuration. |
| `partition_factor`| Defines the fraction of all the schedules to be executed. If the fraction is less than `1.0`, the chosen schedules are sampled randomly. |
| `track_params`    | Indicates whether to track the input parameters of the optimization algorithms during the optimization process. If `False`, only the output of the optimization algorithms is tracked during execution of the schedule. |

The following schedule configurations are available:
- `single.yaml`: Single optimizer schedules.
- `combination.yaml`: Combination of two optimizers with a switch point.
- `l2co.yaml`: Schedules for the L2CO optimizer.

### Budgets

The configurations in the `conf/budgets/` folder define how the budget for each optimization task is determined. Each configuration specifies a budget strategy that can be used to limit the number of iterations or evaluations for each task.

Each configuration is a `functools.partial` function: a key with a `_partial_: True` and `_target_: <your function>` tag so that the configuration can be instantiated with [`hydra.instantiate`](https://hydra.cc/docs/advanced/instantiate_objects/overview/).

The following budget configurations are available:
- `fixed.yaml`: Fixed budget for all tasks.
- `fixed_pde.yaml`: Fixed budget for PDE tasks.
- `based_on_dim.yaml`: Budget based on the dimensionality of the task.

### Samplers

The configurations in the `conf/samplers/` folder define how the initial input parameters ($x_0$) of the optimization tasks are sampled. Each configuration specifies a sampling strategy that can be used to generate the parameters for the optimization tasks.

Each configuration is a `functools.partial` function: a key with a `_partial_: True` and `_target_: <your function>` tag so that the configuration can be instantiated with [`hydra.instantiate`](https://hydra.cc/docs/advanced/instantiate_objects/overview/).

The following sampler configurations are available:
- `random.yaml`: Random weight sampler.
- `grid.yaml`: Grid weight sampler.
- `normal.yaml`: Normal weight sampler.
- `xavier.yaml`: Xavier normal weight sampler.


### Default Configuration
The default configuration is defined in `conf/config.yaml`:
- Task: `gaussian_classification`
- Schedule: `single`
- Budget: `based_on_dim`
- Sampler: `normal`
- Optimizers: `standard`

You can modify the default configuration or use any of the above configurations by updating the `defaults` section in `conf/config.yaml`.

## Contents of this folder

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main script to run the experiment |
| `config.yaml` | Default configuration file for the experiment |
| `conf/` | Folder containing configurations for tasks, schedules, samplers, optimizers, and budgets |
| `README.md` | Explanation of this experiment |
| `img/` | Folder with images used in this file |
| `slurmjob.sh` | SLURM job file to run the experiment on the DelftBlue cluster |
| `slurmjob_brown.sh` | SLURM job file to run the experiment on the OSCAR cluster |
| `slurmjob_gpu.sh` | SLURM job file to run the experiment with GPU configuration on the DelftBlue cluster |
| `slurmjob_brown_gpu.sh` | SLURM job file to run the experiment on GPU nodes in the OSCAR cluster |
| `outputs/` | Folder with the results of running this experiment |

> The `outputs/` folder is created when the experiment has been run for the first time.

## Usage

### Before running the experiment

1. Install the `l2co` package. See [here](https://github.com/bessagroup/L2CO#getting-started) for instructions.
2. Make sure [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) is installed on your system.
3. Change the `config.yaml` file to your liking. You can also overwrite the parameters on the command line  See [here](#explanation-of-configyaml-parameters) for an explanation of the parameters.

### Running the experiment on your local machine

1. Navigate to this folder and run `mpirun -np <number of processes> python main.py`

The number of processes must be at least 2.

### Running the experiment on a TORQUE HPC cluster

1. Make sure you have an `conda` environment named `l2co` with the packages installed in the first step
2. Navigate to this folder and submit the job:

| Cluster | Command |
|---------|---------|
| [DelftBlue](https://www.tudelft.nl/dhpc/system) | `sbatch slurmjob.sh` |
| [OSCAR](https://docs.ccv.brown.edu/oscar) | `sbatch slurmjob_brown.sh` |



## Results

Results are stored in a newly created `outputs` folder, within a subdirectory:
* When running on a local machine, the output will be saved in a directory indicating the current time (e.g. `13-50-24`).
* When running on a cluster, the output will be saved in a
 directory indicating the current job ID (e.g. `1234567`).

The following subdirectories are created:

* `experiment_data`: Contains the input, output, domain, and jobs to construct the `f3dasm.ExperimentData` object.
    - `domain.json`, `input.csv`, `output.csv`, and `jobs.csv` files are the data for the `f3dasm.ExperimentData` object.
    - `/task/` contains the `.json` (task configuration) and `.npy` (model parameter structure) files to create each `Task`.
    - `/optimizer_schedule/` contains the `.json` files for each optimizer schedule.
    - `/loss_history/` contains the NetCDF (`.nc`) files with results for all realizations of the `OptimizationSchedule` on the `Task`.
* `.hydra`: Contains the `config.yaml` file used to run the experiment.
* `main.log`: Contains the log file of the experiment.

The folder structure is as follows:

```
outputs/
└── 2023-11-06/
    └── 13-50-14/
        ├── .hydra/
        │   └── config.yaml
        ├── experiment_data/
        │   ├── domain.json
        │   ├── input.csv
        │   ├── output.csv
        │   ├── jobs.csv
        │   ├── task/
        │   │   ├── 0.json
        │   │   ├── 0.npy
        │   │   └── ...
        │   ├── optimizer_schedule/
        │   │   ├── 0.json
        │   │   └── ...
        │   └── loss_history/
        │       ├── 0.ncs
        │       └── ...
        └── main.log
```

## Explanation of `config.yaml` parameters

### Defaults
The `defaults` section specifies the default configurations for tasks, schedules, budgets, samplers, and optimizers. These can be overridden by modifying the `config.yaml` file or passing parameters via the command line.

| Name         | Type   | Description |
|--------------|--------|-------------|
| `tasks`      | `str`  | Specifies the task configuration to use (e.g., `gaussian_classification`, `benchmark`, etc.). |
| `schedules`  | `str`  | Specifies the schedule configuration to use (e.g., `single`, `combination`, `l2co`). |
| `budgets`    | `str`  | Specifies the budget configuration to use (e.g., `fixed`, `fixed_pde`, `based_on_dim`). |
| `samplers`   | `str`  | Specifies the sampler configuration to use (e.g., `random`, `grid`, `normal`, `xavier`). |
| `optimizers` | `str`  | Specifies the optimizer configuration to use (e.g., `standard`, `small`, `pde`, `l2co`). |

#### General Parameters
| Name             | Type    | Description |
|------------------|---------|-------------|
| `n_realizations` | `int`   | Number of realizations (repeats) for each task. |
| `seed`           | `int`   | Random seed for reproducibility. |
| `mode`           | `str`   | Execution mode (`mpi`, `sequential`, or `cluster`). |
| `log_level`      | `int`   | Logging level (e.g., `10` for DEBUG, `20` for INFO). |
| `verbose`        | `bool`  | Whether to enable verbose logging. |
| `include_input`  | `bool`  | Whether to include input data in the output files. |

#### Paths
| Name          | Type   | Description |
|---------------|--------|-------------|
| `model_dir`   | `str`  | Path to the directory containing the trained L2CO model. |
| `project_dir` | `str`  | Path to the directory where the experiment outputs will be stored. |
| `project_job` | `str`  | Job ID for the current project (used to continue from another experiment). |
