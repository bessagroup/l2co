from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DomainConfig:
    input_space: Dict


@dataclass
class ExperimentConfig:
    from_sampling: dict


@dataclass
class OptimizationConfig:
    lower_bound: float
    upper_bound: float
    sampler_name: str
    number_of_samples: int
    realizations: int
    parallelization: bool
    optimizers: List[Dict]


@dataclass
class RecommenderConfig:
    create_recommender_dataset: bool
    lower_bound_budget: int
    n_augmented_samples: int
    scaler: str


@dataclass
class HPCConfig:
    jobid: int


@dataclass
class FilesConfig:
    dirname_raw: str
    dirname_movxarr: str


@dataclass
class Config:
    domain: DomainConfig
    experimentdata: ExperimentConfig
    optimization: OptimizationConfig
    hpc: HPCConfig
    files: FilesConfig
    recommender: RecommenderConfig
    log_level: int
    mode: str
