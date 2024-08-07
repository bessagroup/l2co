# Create trainingdata two optimizers

<center>
<img src="./img/data-driven-process.png" alt="drawing" width=100%/>
</center>

## Summary

In order to train the recommender system, we need to create a dataset with the performance of different optimization algorithms on different optimization problems.

The dataset is created by solving a set of parametrized benchmark functions with a set of optimization algorithms. The optimization algorithms are combined in pairs, and the optimization problem is solved with each combination. The results of each optimization algorithm on each problem are stored in [NetCDF](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf) (`.nc`) files.

The benchmark functions are parametrized by their analytical formula, dimensionality, noise, and seed:

<center>
<img src="./img/problems.png" alt="drawing" width=50%/>
</center>

Given a set of optimization algorithms, combinations of two optimizers are created, where we can 'switch' from one optimizer to the other at a given iteration:

<center>
<img src="./img/optimizers.png" alt="drawing" width=50%/>
</center>

The benchmark function is optimized for a given number of iterations and repeated with different initial conditions (realizations). The results are stored in a [NetCDF](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf) (`.nc`) file.

<center>
<img src="./img/data.png" alt="drawing" width=50%/>
</center>

For every optimization problem, we solve the problems with a fraction of the total combination of optimization algorithms (controlled by the parameter `fraction_of_all_combinations`):

<center>
<img src="./img/data2.png" alt="drawing" width=50%/>
</center>

## Contents of this folder

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main script to run the experiment |
| `run_opt.py` | Custom `f3dasm.DataGenerator` to run the optimization process |
| `config.yaml` | Configuration file for the experiment |
| `config.py` | Configuration file type-hints for the experiment |
| `README.md` | Explanation of this experiment |
| `img/` | Folder with images used in this file |
| `pbsjob.sh` | TORQUE job file to run the experiment on the [TUD Delft HPC](https://hpcwiki.tudelft.nl/index.php?title=Main_Page) cluster |
| `slurmjob.sh` | SLURM job file to run the experiment on the [DelftBlue](https://www.tudelft.nl/dhpc/system) cluster|
| `slurmjob.sh` | SLURM job file to run the experiment on the [OSCAR](https://docs.ccv.brown.edu/oscar) cluster|
| `outputs/` | Folder with the results of running this experiment |

> The `outputs/` folder is created when the experiment has been run for the first time.

## Usage

### Before running the experiment

1. Install the `l2co` package. See [here](https://github.com/bessagroup/L2CO#getting-started) for instructions.
2. Change the `config.yaml` file to your liking. See [here](#explanation-of-configyaml-parameters) for an explanation of the parameters.

### Running the experiment on your local machine

1. Navigate to this folder and run `python main.py`

### Running the experiment on a TORQUE cluster

1. Make sure you have an `conda` environment named `f3dasm_env` with the packages installed in the first step
2. Navigate to this folder and submit the job:

| Cluster | Command |
|---------|---------|
| [TUD Delft HPC](https://hpcwiki.tudelft.nl/index.php?title=Main_Page)  | `qsub pbsjob.sh` |
| [DelftBlue](https://www.tudelft.nl/dhpc/system) | `sbatch slurmjob.sh` |
| [OSCAR](https://docs.ccv.brown.edu/oscar) | `sbatch slurmjob_brown.sh` |

You can run this experiment on multiple nodes by submitting an arrayjob (e.g. `qsub -t 0-10 pbsjob.sh` or `sbatch --array=0-10 slurmjob.sh`):
- The first node will (`jobid=0`) handle the creation of the `f3dasm.ExperimentData` object.
- The other nodes will (`jobid=1,2,...`) handle the `f3dasm.DataGenerator` process specified in `run_opt.py`.

## Results

Results are stored in a newly created `outputs` folder, with a subdirectory
indicating the current date (e.g. `2023-11-06`).

* When running on a local machine, the output will be saved in a directory indicating the current time (e.g. `13-50-14`).
* When running on a cluster, the output will be saved in a directory indicating the current job ID (e.g. `538734.hpc06.hpc`).

The following subdirectories are created:

* `experiment_data`: Contains the input, output, domain and jobs to construct the `f3dasm.ExperimentData` object.
* `random_samples`: Contains a [NetCDF](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf) (`.nc`) files with random samples for each optimization problem
* `<opt_1>+<opt2>`: Contains a [NetCDF](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf) (`.nc`) files with the optimization results for each optimization problem. The folder name is the concatenation of the two optimization algorithms used.
* `.hydra`: Contains the `config.yaml` file used to run the experiment.

Lastly, a log file `main.log` is created.

The folder structure is as follows:

```
outputs/
└── 2023-11-06/
    └── 13-50-14/
        ├── .hydra/
        ├── CMAES/
        ├── CMAES+Adam/
        ├── CMAES+PSO/
        │   ├── 0.nc
        │   ├── 1.nc
        │   └── 2.nc
        ├── random_samples/
        │   ├── 0.nc
        │   ├── 1.nc
        │   └── 2.nc
        ├── experiment_data/
        │   ├── domain.pkl
        │   ├── input.csv
        │   ├── output.csv
        │   └── jobs.pkl
        └── main.log
```


## Explanation of `config.yaml` parameters

### Domain
#### Function Name
| Name       | Type     | Description                |
|------------|----------|----------------------------|
| categories | `List[str]`     | List of benchmark functions ([reference](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/defaults.html#implemented-benchmark-functions)) |

#### Dimensionality
| Name       | Type     | Description                |
|------------|----------|----------------------------|
| categories | `List[int]`     | List of possible values of the function dimensionality |

#### Noise
| Name       | Type     | Description                |
|------------|----------|----------------------------|
| categories | `List[float]`     | List of possible values of the function noise standard deviation    |

#### Seed
| Name | Type | Description         |
|------|------|---------------------|
| low  | `int`  | Lower bound value of random seeds    |
| high | `int`  | Upper bound value of random seeds    |

#### Budget
| Name  | Type     | Description           |
|-------|----------|-----------------------|
| value | `int`      | Maximum number of iterations          |

### Experiment Data
#### From Sampling
| Name         | Type   | Description            |
|--------------|--------|------------------------|
| sampler      | `str` | Sampler name            |
| seed         | `int`    | Seed value              |
| n_samples    | `int`    | Number of samples       |
| domain       | `f3dasm.Domain` | `f3dasm` Domain object ([reference](https://f3dasm.readthedocs.io/en/latest/_autosummary/f3dasm.design.Domain.html#f3dasm.design.Domain))            |

### Mode
| Name  | Type   | Description |
|-------|--------|-------------|
| mode  | string | Evaluation mode of `f3dasm` ([reference](https://f3dasm.readthedocs.io/en/latest/_autosummary/f3dasm.datageneration.DataGenerator.html#f3dasm.datageneration.DataGenerator)) |

### Optimization
| Name                       | Type     | Description               |
|----------------------------|----------|---------------------------|
| lower_bound                | `float`    | Box-constraint lower bound for every dimension |
| upper_bound                | `float`    | Box-constraint upper bound for every dimension |
| sampler_name               | `str`   | Name of the sampling strategy for the first iterations ([reference](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/defaults.html#implemented-samplers)) |
| number_of_samples          | `int`      | Number of initial samples ($\vec{x}_0$)          |
| realizations               | `int`      | Number of realizations with different initial conditions    |
| parallelization            | `bool`  | True if realizations are computed on indivdiual cores, else for sequential |
| optimizers                 | `List[str]`     | List of optimizers ([from `f3dasm`](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/defaults.html#implemented-optimizers), [and `f3dasm_optimize`](https://bessagroup.github.io/f3dasm_optimize/rst_doc_files/optimizers.html#implemented-optimizers))        |
| fraction_of_all_combinations[^1] | `float`  | Fraction of combinations (optimizer + problem) to evaluate  |

[^1]: At least one optimizer will be evaluated for each problem.

### HPC
| Name   | Type | Description  |
|--------|------|--------------|
| jobid[^2]  | `int`  | Job ID of the array-job, automatically overwritten by scheduler bash script |

[^2]: When running on a local machine, this value will be left as the default: -1.

### Log Level
| Name      | Type | Description     |
|-----------|------|-----------------|
| log_level | `int`  | Log level value ([see `logging` module for more info](https://docs.python.org/3/library/logging.html#logging-levels)) |
