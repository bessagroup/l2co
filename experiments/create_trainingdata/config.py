from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class DesignConfig:
    input_space: List[Dict]
    output_space: List[Dict]


@dataclass
class ExperimentConfig:
    existing_data_path: Union[bool, str]
    name: str
    sampler: str
    seed: int
    number_of_samples: int


@dataclass
class OptimizationConfig:
    lower_bound: float
    upper_bound: float
    sampler_name: str
    number_of_samples: int
    iterations: int
    realizations: int
    parallelization: bool
    optimizers: List[Dict]


@dataclass
class HPCConfig:
    jobid: int
    jobqueue_filename: str


@dataclass
class FilesConfig:
    dirname_raw: str
    dirname_interactions: str
    dirname_features: str
    dirname_recommender: str


@dataclass
class Config:
    design: DesignConfig
    experimentdata: ExperimentConfig
    optimization: OptimizationConfig
    hpc: HPCConfig
    files: FilesConfig
