from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DesignConfig:
    input_space: List[Dict]
    output_space: List[Dict]


@dataclass
class ExperimentConfig:
    name: str
    sampler: str
    seed: int
    number_of_samples: int


@dataclass
class MLPConfig:
    alpha: float
    seed: int


# @dataclass
# class OptimizationConfig:
#     lower_bound: float
#     upper_bound: float
#     sampler_name: str
#     number_of_samples: int
#     iterations: int
#     realizations: int
#     parallelization: bool
#     optimizers: List[Dict]


@dataclass
class HPCConfig:
    jobid: int
    jobqueue_filename: str


@dataclass
class FilesConfig:
    train_dirname_raw: str
    test_dirname_raw: str
    dirname_mov: str


@dataclass
class Config:
    design: DesignConfig
    experimentdata: ExperimentConfig
    # optimization: OptimizationConfig
    mlp: MLPConfig
    hpc: HPCConfig
    files: FilesConfig
