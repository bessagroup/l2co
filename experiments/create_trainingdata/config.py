from dataclasses import dataclass
from typing import List


@dataclass
class DesignConfig:
    name: str
    functions: List[str]
    dimensionality_list: List[str]
    noise_list: List[str]
    sampler: str
    seed: int
    number_of_samples: int


@dataclass
class OptimizationConfig:
    optimizers_names: List[str]
    lower_bound: float
    upper_bound: float
    sampler_name: str
    number_of_samples: int
    iterations: int
    realizations: int
    parallelization: bool


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
    optimization: OptimizationConfig
    hpc: HPCConfig
    files: FilesConfig
