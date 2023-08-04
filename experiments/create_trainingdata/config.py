from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class DesignConfig:
    input_space: Dict
    output_space: Dict


@dataclass
class ExperimentConfig:
    existing_data_path: Union[bool, str]
    name: str
    mode: str


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
    dirname_recommender_interactions: str
    dirname_recommender_features: str


@dataclass
class Config:
    design: DesignConfig
    experimentdata: ExperimentConfig
    optimization: OptimizationConfig
    hpc: HPCConfig
    files: FilesConfig
    recommender: RecommenderConfig
