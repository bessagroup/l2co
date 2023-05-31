from dataclasses import dataclass


@dataclass
class FileConfig:
    dirname: str
    dirname_raw: str
    dirname_interactions: str
    dirname_features: str
    dirname_recommender: str
    output_filename: str


@dataclass
class ExperimentConfig:
    existing_data_path: str
    name: str


@dataclass
class HPCConfig:
    jobid: int


@dataclass
class Config:
    files: FileConfig
    hpc: HPCConfig
    experiment: ExperimentConfig
