from dataclasses import dataclass
from typing import Dict


@dataclass
class DomainConfig:
    domain: Dict


@dataclass
class FileConfig:
    raw_data: str
    experimentdata_file: str


@dataclass
class ExperimentConfig:
    mode: str
    from_sampling: Dict


@dataclass
class HPCConfig:
    jobid: int


@dataclass
class Config:
    domain: DomainConfig
    files: FileConfig
    hpc: HPCConfig
    experimentdata: ExperimentConfig
