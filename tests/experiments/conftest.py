from pathlib import Path

import pytest
from hydra import compose, initialize


def _load_config(config_name: Path):

    with initialize(config_path="."):
        cfg = compose(config_name=config_name)
    return cfg


# add a fixture which loads a hydra file
@pytest.fixture(scope='package')
def config_create_trainingdata_two_optimizers():
    return _load_config("config_create_trainingdata_two_optimizers.yaml")
