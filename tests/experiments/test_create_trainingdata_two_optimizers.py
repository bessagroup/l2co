import pytest
from omegaconf import DictConfig

# pytestmark = pytest.mark.smoke


def test_load_hydra(config_create_trainingdata_two_optimizers: DictConfig):
    assert isinstance(config_create_trainingdata_two_optimizers, DictConfig)
    assert config_create_trainingdata_two_optimizers.mode == 'cluster'
