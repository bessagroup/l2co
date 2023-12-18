from f3dasm import ExperimentSample
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
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
