from l2co._src.data_loader import compute_parameters, compute_X_trajectory
from l2co import logger as l2co_logger
from l2co import DataLoader
import jax
import hydra
from pathlib import Path

jax.config.update("jax_disable_jit", True)


# logging.basicConfig(level=logging.ERROR)


@hydra.main(config_path=".", config_name="config")
def main(config):
    import logging
    logging.getLogger().setLevel(logging.ERROR)  # Root logger
    logging.getLogger("cola").setLevel(logging.ERROR)  # Suppress cola logs

    path = Path(config.opt_dir) / str(config.path)

    l2co_logger.info(f"Loading data from {path} ..")

    d = DataLoader.from_file(path)
    l2co_logger.info(f"Data loaded from {path}")

    X_trajectories = compute_X_trajectory(
        d.select_t(t_start=config.t_start, t=config.t).data)

    l2co_logger.info(f"X trajectories computed from {path}")

    X_params = compute_parameters(X_trajectories.isel(problem=range(10)))

    l2co_logger.info(f"X parameters computed from {path}")

    X_params.to_netcdf('X_params.nc')

    l2co_logger.info("X parameters stored in X_params.nc")
    l2co_logger.info("End of the program")


if __name__ == '__main__':
    main()
