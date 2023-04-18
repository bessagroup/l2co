

import f3dasm
import hydra
# Third-party
from config import Config
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    print(config.optimization)
    opt = []
    for o in config.optimization.optimizers:
        hyperparameters = o['hyperparameters'] if 'hyperparameters' in o else {}
        opt.append(f3dasm.optimization.find_optimizer(o['class'])(
            data=0, hyperparameters=hyperparameters, name=o['name']))

    print(opt)


if __name__ == "__main__":
    main()
