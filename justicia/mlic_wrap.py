from pyrulelearn.imli import imli
import justicia.utils as utils
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import metrics
import numpy as np
import os
import pickle


def init(dataset, repaired=False, verbose=False, compute_equalized_odds=False, thread=0, remove_column=None):

    df = dataset.get_df(repaired=repaired)

    if(remove_column is not None):
        assert isinstance(remove_column, str)
        df = df.drop([remove_column], axis=1)
        if(remove_column in dataset.continuous_attributes):
            dataset.continuous_attributes.remove(remove_column)

    # discretize
    data = utils.get_discretized_df(
        df, columns_to_discretize=dataset.continuous_attributes, verbose=verbose)

    # get X,y
    X = data.drop(['target'], axis=1)
    y = data['target']

    # one-hot
    X = utils.get_one_hot_encoded_df(X, X.columns.to_list(), verbose=verbose)

    skf = KFold(n_splits=5, shuffle=True, random_state=10)
    skf.get_n_splits(X, y)

    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    clfs = []
    clf_negs = []

    os.system("mkdir -p data/model/")
    cnt = 0

    for train, test in skf.split(X, y):

        X_trains.append(X.iloc[train])
        y_trains.append(y.iloc[train])
        X_tests.append(X.iloc[test])
        y_tests.append(y.iloc[test])

        if(remove_column is None):
            store_file = "data/model/CNF_" + dataset.name + "_" + \
                str(dataset.config) + "_" + str(cnt) + ".pkl"
        else:
            store_file = "data/model/CNF_" + dataset.name + "_remove_" + \
                remove_column.replace(" ", "_") + "_" + \
                str(dataset.config) + "_" + str(cnt) + ".pkl"

        if(not os.path.isfile(store_file)):
            os.system("mkdir -p data/temp_" + str(thread))
            clf = imli(num_clause=2, data_fidelity=10, work_dir="data/temp_" +
                       str(thread), rule_type="CNF", verbose=False)
            clf.fit(X_trains[-1].values, y_trains[-1].values)
            os.system("rm -r data/temp_" + str(thread))

            # save the classifier
            with open(store_file, 'wb') as fid:
                pickle.dump(clf, fid)

        else:
            # Load the classifier
            with open(store_file, 'rb') as fid:
                clf = pickle.load(fid)

        clfs.append(clf)

        if(verbose):
            print("\nFeatures: ", X_trains[-1].columns.to_list())
            print("Number of features:", len(X_trains[-1].columns.to_list()))
            print("\nlearned rule:")
            print(clf.get_rule(X_trains[-1].columns.to_list()))

        if(verbose):
            print("\nTrain Accuracy Score: ", metrics.accuracy_score(clf.predict(
                X_trains[-1].values), y_trains[-1].values), "positive ratio: ", y_trains[-1].mean())
            print("Test Accuracy Score: ", metrics.accuracy_score(clf.predict(
                X_tests[-1].values), y_tests[-1].values), "positive ratio: ", y_tests[-1].mean())

        cnt += 1

    if(compute_equalized_odds):
        return clfs,  X_trains, X_tests, dataset.known_sensitive_attributes, y_trains, y_tests

    return clfs,  X_trains, X_tests, dataset.known_sensitive_attributes
