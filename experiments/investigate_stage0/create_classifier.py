from f3dasm import ExperimentSample
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                              ExpSineSquared, Matern)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def create_classifier(es: ExperimentSample):

    classifier_name = es.get('classifier')

    if classifier_name == 'DecisionTree':
        return create_decisiontree(es)

    elif classifier_name == 'SVC':
        return create_svc(es)

    elif classifier_name == 'KNN':
        return create_knn(es)

    elif classifier_name == 'GPC':
        return create_gpc(es)

    elif classifier_name == 'GPR':
        return create_gpr(es)

    elif classifier_name == 'GradientBoostingRegressor':
        return create_gradientboostingregressor(es)


def create_gpr(es: ExperimentSample):
    kernel = es.get('kernel')

    if kernel == 'RBF':
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            RBF(1.0, length_scale_bounds="fixed")

    elif kernel == 'ExpSineSquared':
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            ExpSineSquared(1.0, length_scale_bounds="fixed",
                           periodicity_bounds="fixed")

    elif kernel == 'Matern':
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            Matern(1.0, length_scale_bounds="fixed", nu=1.5)

    return MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel))


def create_gradientboostingregressor(es: ExperimentSample):
    learning_rate = es.get('learning_rate')
    n_estimators = es.get('n_estimators')
    return MultiOutputRegressor(
        GradientBoostingRegressor(random_state=0,
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate))


def create_decisiontree(es: ExperimentSample):
    criterion = es.get('criterion')
    return DecisionTreeClassifier(criterion=criterion)


def create_gpc(es: ExperimentSample):
    return GaussianProcessClassifier()


def create_knn(es: ExperimentSample):
    n_neighbors = es.get('n_neighbors')
    weights = es.get('weights')
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)


def create_svc(es: ExperimentSample):
    kernel = es.get('kernel')
    clf = SVC(kernel=kernel)
    return create_multiclass_classifier(es, clf)


def create_multiclass_classifier(es: ExperimentSample, single_clf):
    multiclass_mode = es.get('multiclass_mode')

    if multiclass_mode == 'one-vs-rest':
        return OneVsRestClassifier(single_clf)

    if multiclass_mode == 'one-vs-one':
        return OneVsOneClassifier(single_clf)
