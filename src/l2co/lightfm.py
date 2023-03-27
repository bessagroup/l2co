#                                                                       Modulesset
# =============================================================================

# Standard
from itertools import chain

# Third-party
import numpy as np
import pandas as pd
from lightfm import LightFM, cross_validation
from lightfm.data import Dataset
from recommenders.models.lightfm.lightfm_utils import prepare_all_predictions

# Local
from .train_test_no_shuffle import no_shuffle_train_test_split

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# default number of recommendations
K = 10
# percentage of data used for testing
TEST_PERCENTAGE = 0.25
# model learning rate
LEARNING_RATE = 0.25
# no of latent factors
NO_COMPONENTS = 20
# no of epochs to fit model
NO_EPOCHS = 20
# no of threads to fit model
NO_THREADS = 32
# regularisation for both user and item features
ITEM_ALPHA = 1e-6
USER_ALPHA = 1e-6

# seed for pseudonumber generations
SEED = 42


def create_predictions_df_with_lightfm(folder_to_files: str) -> pd.DataFrame:
    # Read datasets
    data_interactions = pd.read_csv(f'{folder_to_files}/totalinteractions.csv', index_col=0)
    data_userfeatures = pd.read_csv(f'{folder_to_files}/totalfeatures_optimizers.csv', index_col=0)
    data_itemfeatures = pd.read_csv(f'{folder_to_files}/totalfeatures_functions.csv', index_col=0)

    # Combine to one dataframe 'data':
    features = pd.DataFrame([data_itemfeatures[data_itemfeatures['itemID'] == d]
                            ['genres'].values for d in data_interactions['itemID']])
    occupation = pd.DataFrame([data_userfeatures[data_userfeatures['userID'] == d]
                               ['occupation'].values for d in data_interactions['userID']])
    data_interactions['genres'] = features
    data_interactions['occupation'] = occupation

    # retrieve the all the unique categories in the data
    movie_genre = [x.split(',') for x in data_itemfeatures['genres']]
    occupations = [x.split(',') for x in data_interactions['occupation']]
    all_movie_genre = sorted(list(set(chain.from_iterable(movie_genre))))
    all_occupations = sorted(list(set(chain.from_iterable(occupations))))

    # Create Dataset object
    dataset2 = Dataset()
    dataset2.fit(data_interactions['userID'],
                 data_interactions['itemID'],
                 item_features=all_movie_genre,
                 user_features=all_occupations)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data_interactions.itemID, movie_genre))
    user_features = dataset2.build_user_features((x, y) for x, y in zip(data_interactions.userID, occupations))

    interactions2, weights2 = dataset2.build_interactions(data_interactions.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = no_shuffle_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    uids, iids, interaction_data = cross_validation._shuffle(
        interactions2.row,
        interactions2.col,
        interactions2.data,
        random_state=np.random.RandomState(SEED)
    )

    # cutoff = int((1.0 - TEST_PERCENTAGE) * len(uids))
    # test_idx = slice(cutoff, None)

    uid_map, ufeature_map, iid_map, ifeature_map = dataset2.mapping()

    all_predictions2 = prepare_all_predictions(data_interactions, uid_map, iid_map,
                                               interactions=train_interactions2,
                                               user_features=user_features,
                                               item_features=item_features,
                                               model=model2,
                                               num_threads=NO_THREADS)
    return all_predictions2
