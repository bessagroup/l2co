# Data validation

## Contents of this folder

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main script to run the experiment |
| `README.md` | Explanation of this experiment |

## Summary

This script is a post-processing routine for checking the validity of the data generated in the `create_trainingdata_two_optimizers` experiment:

1. It loads each optimization dynamics file (`.nc`) and checks if there are any `np.nan` values in the input parameters or objective function values
2. It concatenates all the optimization dynamics of all optimizers for each task
3. Lastly, it concatenates all the tasks into one big `xarray.Dataset`
4. The resulting `xarray.Dataset` is saved as a `entire_dataset.nc` file in the designated directory

## Usage

### Before running the experiment

1. Install the `l2co` package. See [here](https://github.com/bessagroup/L2CO#getting-started) for instructions.

### Running the experiment on your local machine

1. Navigate to this folder and run `python main.py --path <path/to/experiment>`

### Running the experiment on the TORQUE/DelftBlue/OSCAR cluster

This script is not yet compatible to be run on a cluster.

## Configuration parameters

The following command line arguments can be passed to the script:

| Name | Description |
|------|-------------|
| `--path` | Path to the experiment directory containing the optimization dynamics files |