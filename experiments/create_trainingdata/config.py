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
class HPCConfig:
    jobid: int


@dataclass
class FilesConfig:
    dirname_raw: str


@dataclass
class Config:
    design: DesignConfig
    experimentdata: ExperimentConfig
    optimization: OptimizationConfig
    hpc: HPCConfig
    files: FilesConfig
