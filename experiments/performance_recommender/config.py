from dataclasses import dataclass


@dataclass
class FileConfig:
    raw_data: str
    experimentdata_file: str
    interactions_file: str
    training_features_file: str
    output_filename: str
    output_dir: str
    output_experimentdata_file: str
    output_master_file: str


@dataclass
class ExperimentConfig:
    mode: str


@dataclass
class HPCConfig:
    jobid: int


@dataclass
class Config:
    files: FileConfig
    hpc: HPCConfig
    experiment: ExperimentConfig
