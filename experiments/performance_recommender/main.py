import os

import hydra
from config import Config
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from l2co.lightfm import create_predictions_df_with_lightfm
from l2co.margin_of_victory import create_recommendation_results


@hydra.main(config_path=".", config_name="config")
def main(cfg: Config):
    PATH_OF_FILES = os.path.join(get_original_cwd(), cfg.files.raw)
    FOLDER_TO_CSV = os.path.join(get_original_cwd(), cfg.files.recommender)

    df = create_predictions_df_with_lightfm(folder_to_files=FOLDER_TO_CSV)

    unique_itemids = list(set(df['itemID'].values))
    recommendations = {_id: int(df[df['itemID'] == _id].sort_values(
        by=['prediction'], ascending=False).iloc[0]['userID']) for _id in unique_itemids}

    df = create_recommendation_results(
        path_of_files=PATH_OF_FILES, recommendations=recommendations, selection=unique_itemids)

    df.to_csv(f'{cfg.files.outputfilename}.csv')


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

if __name__ == "__main__":
    main()
