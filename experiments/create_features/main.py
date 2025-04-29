from pathlib import Path

import hydra

from l2co import DataLoader
from l2co import logger as l2co_logger
from l2co._src.data_loader import compute_parameters, compute_X_trajectory


@hydra.main(config_path=".", config_name="config")
def main(config):
    path = Path(config.opt_dir) / str(config.path)

    l2co_logger.info(f"Loading data from {path} ..")

    d = DataLoader.from_file(path)
    l2co_logger.info(f"Data loaded from {path}")

    X_trajectories = compute_X_trajectory(
        d.select_t(t_start=config.t_start, t=config.t).data)

    l2co_logger.info(f"X trajectories computed from {path}")

    X_params = compute_parameters(X_trajectories.isel(problem=range(3)))

    l2co_logger.info(f"X parameters computed from {path}")

    X_params.to_netcdf('X_params.nc')

    l2co_logger.info("X parameters stored in X_params.nc")
    l2co_logger.info("End of the program")


if __name__ == '__main__':
    main()
