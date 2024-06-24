from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DomainConfig:
    input_space: Dict
    # output_space: Dict


@dataclass
class ExperimentConfig:
    name: str
    from_sampling: dict
    # existing_data_path: Union[bool, str]
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
    fraction_of_all_combinations: float
    combinations: bool

# @dataclass
# class RecommenderConfig:
#     create_recommender_dataset: bool
#     lower_bound_budget: int
#     n_augmented_samples: int
#     scaler: str


@dataclass
class HPCConfig:
    jobid: int


# @dataclass
# class FilesConfig:
#     dirname_raw: str
#     dirname_movxarr: str


@dataclass
class Config:
    domain: DomainConfig
    experimentdata: ExperimentConfig
    optimization: OptimizationConfig
    hpc: HPCConfig
    # files: FilesConfig
    # recommender: RecommenderConfig
    log_level: int
    mode: str
