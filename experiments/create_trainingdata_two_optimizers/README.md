# Create trainingdata two optimizers

## Contents of this folder

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main script to run the experiment |
| `config.yaml` | Configuration file for the experiment |
| `config.py` | Configuration file type-hints for the experiment |
| `README.md` | Explanation of this experiment |
| `img/` | Folder with images used in this file |
| `pbsjob.sh` | TORQUE job file to run the experiment in a cluster |
| `outputs/` | Folder with the results of running this experiment |

> The `outputs/` folder is created when the experiment has been run for the first time.

## Summary

This experiment creates a set of  parametrized benchmark functions:

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

## Usage

1. Install the `l2co` package. See [here](https://github.com/bessagroup/L2CO#getting-started) for instructions.
1. Install `f3dasm_optimize` in your environment. See [here](https://bessagroup.github.io/f3dasm_optimize/rst_doc_files/getting_started.html) for instructions.
2. Navigate to this folder and run `python main.py`

## Results

Results are stored in a newly created `outputs` folder, with a subdirectory
indicating the current date (e.g. `2023-11-06`) and time (e.g. `13-50-14`).

The following subdirectories are created:

* `experiment_data`: Contains the input, output, domain and jobs to construct the `f3dasm.ExperimentData` object.
* `random_samples`: Contains a [NetCDF](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf) (`.nc`) files with random samples for each optimization problem
* `<opt_1>+<opt2>`: Contains a [NetCDF](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf) (`.nc`) files with the optimization results for each optimization problem. The folder name is the concatenation of the two optimization algorithms used.
* `.hydra`: Contains the `config.yaml` file used to run the experiment.

Lastly, a log file `main` is created.

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
| categories | `List[str]`     | List of benchmark functions ([reference](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/classes/datageneration/functions.html)) |

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
| domain       | `f3dasm.Domain` | `f3dasm` Domain object ([reference](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/classes/design/domain.html))            |

### Mode
| Name  | Type   | Description |
|-------|--------|-------------|
| mode  | string | Evaluation mode of `f3dasm` ([reference](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/classes/datageneration/datagenerator.html#)) |

### Optimization
| Name                       | Type     | Description               |
|----------------------------|----------|---------------------------|
| lower_bound                | `float`    | Box-constraint lower bound for every dimension |
| upper_bound                | `float`    | Box-constraint upper bound for every dimension |
| sampler_name               | `str`   | Name of the sampling strategy for the first iterations ([reference](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/classes/sampling/sampling.html#id2)) |
| number_of_samples          | `int`      | Number of initial samples ($\vec{x}_0$)          |
| realizations               | `int`      | Number of realizations with different initial conditions    |
| parallelization            | `bool`  | True if realizations are computed on indivdiual cores, else for sequential |
| optimizers                 | `List[str]`     | List of optimizers ([ref1](https://f3dasm.readthedocs.io/en/latest/rst_doc_files/classes/optimization/optimizers.html#implemented-optimizers), [ref2](https://bessagroup.github.io/f3dasm_optimize/rst_doc_files/optimizers.html#implemented-optimizers))        |
| fraction_of_all_combinations | `float`  | Fraction of combinations (optimizer + problem) to evaluate  |

### HPC
| Name   | Type | Description  |
|--------|------|--------------|
| jobid  | `int`  | Job ID of the array-job, automatically overwritten by scheduler bash script |

### Log Level
| Name      | Type | Description     |
|-----------|------|-----------------|
| log_level | `int`  | [Log level value](https://docs.python.org/3/library/logging.html#logging-levels) |
