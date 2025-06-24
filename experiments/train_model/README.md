# Train Model


## Summary

The `train_model` experiment is responsible for training the L2CO model. This model is trained using the dataset generated in the `create_optimization_dataset_new` experiment. The training process involves:

1. Loading the optimization dataset.
2. Extracting features and preparing the data for training.
3. Initializing and configuring the L2CO model.
4. Training the model using the specified optimizer, loss function, and hyperparameters.
5. Saving the trained model for use in downstream tasks.

## Configurations

The `conf` folder contains configurations for the training process. Below is a summary of the available configurations:

### Parameters
| Name             | Type   | Description |
|------------------|--------|-------------|
| `path`          | `str`  | Experiment ID to locate the `entire_dataset.odata` file. |
| `opt_dir`        | `str`  | Path to the directory containing the optimization dataset. |
| `seed`           | `int`  | Random seed for reproducibility. |
| `t_start`, `t`, `t_end` | `float` | Time intervals for feature extraction. |
| `n_optimizer_choices` | `int`  | Number of optimizer choices in the dataset. |
| `stage0_model`   | `dict` | Configuration for the Stage 0 model. |
| `stage1_model`   | `dict` | Configuration for the Stage 1 model. |
| `feature_types` | `list` | List of feature types to be extracted from the optimization tasks. |
| `optimizer`      | `dict` | Configuration for the optimizer used in training. |
| `epochs`         | `int`  | Number of training epochs. |
| `batch_size`     | `int`  | Batch size for training. |
| `loss_fn`        | `dict` | Configuration for the loss function. Partial function |

## Contents of this folder

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main script to train the L2CO model. |
| `config.yaml` | Configuration file for the training experiment. |
| `README.md` | Explanation of this experiment. |
| `outputs/` | Folder where the trained model and logs will be stored. |

> The `outputs/` folder is created when the experiment has been run for the first time.

## Usage

### Before running the experiment

1. Install the `l2co` package. See [here](https://github.com/bessagroup/L2CO#getting-started) for instructions.
2. Change the `config.yaml` file to your liking. See [here](#explanation-of-configyaml-parameters) for an explanation of the parameters.

### Running the experiment on your local machine

1. Navigate to this folder and run `python main.py path=<experiment_id>`. Replace `<experiment_id>` with the ID of the experiment you want to run.

### Running the experiment on a cluster

1. Make sure you have a `conda` environment named `l2co` with the required packages installed.
2. Submit the job using your cluster's job scheduler.

## Results

The trained model is saved in the `outputs` folder. The folder structure is as follows:

```
outputs/
└── model/
    ├── stage0_model.pkl
    ├── stage1_model.pkl
    └── l2co_model.pkl
└── logs/
    └── training.log
```