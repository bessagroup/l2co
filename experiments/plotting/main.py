
import time
from glob import glob
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
from config import Config
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from l2co._logging import logger
from l2co.saving import load_netcdf
from l2co.visualization import plot_four_figures


@hydra.main(config_path=".", config_name="config")
def main(config: Config):

    start_time = time.perf_counter()

    # orig_cwd = Path(get_original_cwd())

    # Load experiment data
    raw_data = Path(config.files.dirname) / Path(config.files.dirname_raw)


    # Get a list of filenames that end with .cd
    filenames = glob(f"{raw_data}/*.cd")
    print(filenames)

    for file in filenames:
        results = load_netcdf(file_path=file)
        fig: plt.Figure = plot_four_figures(results)

        # Convert to Path object and get only the filename
        file = Path(file).name

        # Save the figure
        fig.savefig(f"{file}.png", dpi=300)

    end_time = time.perf_counter()
    logger.info(f"Finished in {end_time - start_time:0.4f} seconds")


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
