
# wrapper for logistic regression
from pysat.pb import *
from pysat.solvers import Glucose3
from fractions import gcd
from functools import reduce
# from fairness_comparison.fairness.data.objects.ProcessedData import ProcessedData
# from fairness_comparison.fairness.data.objects.Adult import Adult
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.datasets
import numpy as np
from sklearn.datasets import load_iris
import justicia.utils as utils
from sklearn import metrics
import numpy as np
import os
import pickle
from datetime import datetime
import random


def predict_lin(X, weights, bias):
    X = np.array(X)
    dot_result = np.dot(X, np.array(weights))
    return (dot_result >= -1 * bias).astype(int)


class linear_classifier_wrap_Wrapper():

    # TODO
    # We are given a list of weight and a bias :  w = weights of features, b = bias
    # assumption: attributes are Boolean.

    def _find_gcd(self, list):
        x = reduce(gcd, list)
        return x

    def __init__(self, weights, attributes, sensitive_attributes, bias, converted_data, original_model_prediction, convert_to_cnf, negate=False, verbose=True):
        self.weights = weights
        self.bias = bias
        self.classifier = []
        self.num_attributes = len(weights)
        self.sensitive_attributes = sensitive_attributes
        self.attributes = attributes
        self.auxiliary_variables = []
        self._store_benchmark = False
        if(self._store_benchmark):
            os.system("mkdir -p pseudo_Boolean_benchmarks")

        self._convert_weights_to_integers(
            converted_data, original_model_prediction)
        if(convert_to_cnf):
            self._get_cnf(negate)
        if(verbose):
            print("Expression: ", " ".join(
                map(str, self.weights)), "?=", -1*self.bias)
            # print("clauses: ", self.classifier)

    def predict(self, X, weights, bias):
        X = np.array(X)
        dot_result = np.dot(X, np.array(weights))
        return (dot_result >= -1 * bias).astype(int)

    def _convert_weights_to_integers(self, converted_data, original_model_prediction):

        # convert w, bias to integer
        _max = abs(np.array(self.weights + [self.bias])).max()
        best_accuracy = -1
        best_weights = None
        best_bias = None
        best_multiplier = None

        # gridsearch to find best multiplier
        for multiplier in range(101):
            _weights = [int(float(weight/_max)*multiplier)
                        for weight in self.weights]
            _bias = int(float(self.bias/_max)*multiplier)
            measured_accuracy = metrics.f1_score(self.predict(
                converted_data, _weights, _bias), original_model_prediction)
            if(measured_accuracy > best_accuracy):
                best_weights = _weights
                best_bias = _bias
                best_best_multiplier = multiplier
                best_accuracy = measured_accuracy
        assert best_weights is not None
        assert best_bias is not None
        self.weights = best_weights
        self.bias = best_bias
        # print(accuracy, best_multiplier, self.weights, self.bias)

    def _get_cnf(self, negate=False):
        # apply pseudo-Boolean encoding to get the CNF of the classifier
        lits = [i+1 for i in range(len(self.attributes) + len(
            [_var for _group in self.sensitive_attributes for _var in _group]))]
        if(negate):
            cnf = PBEnc.atmost(lits=lits, weights=self.weights,
                               bound=-1 * self.bias - 1, top_id=self.num_attributes)

            if(self._store_benchmark):
                benchmark_file = "Justicia_" + str(len(self.weights)) + datetime.now(
                ).strftime('_%Y_%m_%d_%H_%M_%S_') + str(random.randint(0, 100000))
                s = "* #variable= " + \
                    str(len(self.weights)) + " #constraint= 1\n"
                s += (" ").join([b + " " + a for a, b in list(zip(["x" + str(i) for i in lits], ["+" + str(_weight) if _weight >
                                                                                                 0 else "-" + str(-1*_weight) for _weight in self.weights]))] + ["<= " + str(-1 * self.bias)]) + ";\n"
                fout = open("pseudo_Boolean_benchmarks/" +
                            benchmark_file + ".opb", "w")
                fout.write(s)
                fout.close()
                cnf.to_file("pseudo_Boolean_benchmarks/" +
                            benchmark_file + ".cnf")

        else:
            cnf = PBEnc.atleast(lits=lits, weights=self.weights,
                                bound=-1 * self.bias, top_id=self.num_attributes)

            if(self._store_benchmark):
                benchmark_file = "Justicia_" + str(len(self.weights)) + datetime.now(
                ).strftime('_%Y_%m_%d_%H_%M_%S_') + str(random.randint(0, 100000))
                s = "* #variable= " + \
                    str(len(self.weights)) + " #constraint= 1\n"
                s += (" ").join([b + " " + a for a, b in list(zip(["x" + str(i) for i in lits], ["+" + str(_weight) if _weight >
                                                                                                 0 else "-" + str(-1*_weight) for _weight in self.weights]))] + [">= " + str(-1 * self.bias)]) + ";\n"
                fout = open("pseudo_Boolean_benchmarks/" +
                            benchmark_file + ".opb", "w")
                fout.write(s)
                fout.close()
                cnf.to_file("pseudo_Boolean_benchmarks/" +
                            benchmark_file + ".cnf")

        self.classifier = cnf.clauses
        self.auxiliary_variables = [i for i in range(
            self.num_attributes + 1, cnf.nv + 1)]
        self.num_attributes = max(cnf.nv, self.num_attributes)

    def check_assignment(self):
        # check whether there is a satisfying assignment to the classifier.
        g = Glucose3()
        for clause in self.classifier:
            g.add_clause(clause)

        print("is SAT? ", g.solve())
        print("assignment: ", g.get_model())


# supporting functions for learning weights and bias


def init_iris():
    """ 
    Returns weights, bias, features (including sensitive features), and sensitive features
    """

    # loading dataset

    target = "target"
    dataset = load_iris()
    dataset[target] = np.where(dataset[target] == 2, 0, dataset[target])

    # get df

    data_df = utils.sklearn_to_df(dataset)

    # discretize
    data = utils.get_discretized_df(
        data_df, columns_to_discretize=data_df.columns.to_list())

    # get X,y
    X = data.drop(['target'], axis=1)
    y = data['target']

    # one-hot
    X = utils.get_one_hot_encoded_df(X, X.columns.to_list())

    # split into train_test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    known_sensitive_attributes = [['sepal length (cm)_1']]
    attributes, sensitive_attributes, probs = utils.get_statistics_from_df(
        X_train, known_sensitive_attributes)

    #  For linear classifier, we use Logistic regression model of sklearn
    clf = LogisticRegression(random_state=0)
    clf = clf.fit(X_train, y_train)

    print("\nFeatures: ", X_train.columns.to_list())
    print("\nWeights: ", clf.coef_)
    print("\nBias:", clf.intercept_[0])
    assert len(clf.coef_[0]) == len(
        X_train.columns), "Error: wrong dimension of features and weights"

    print("Train Accuracy Score: ", clf.score(
        X_train, y_train), "positive ratio: ", y_train.mean())
    print("Test Accuracy Score: ", clf.score(
        X_test, y_test), "positive ratio: ", y_test.mean())

    return clf.coef_[0], clf.intercept_[0], attributes, sensitive_attributes, probs


# process adult dataset


def init_synthetic():
    filename = "data/sample.csv"

    if(os.path.isfile(filename)):
        dataframe = pd.read_csv(filename)
    else:
        cols = 10
        rows = 200

        matrix = np.random.randint(2, size=(rows, cols))
        dataframe = pd.DataFrame.from_records(matrix)
        dataframe.columns = ['col-' + str(i)
                             for i in range(cols-1)] + ['target']
        dataframe.to_csv(filename, index=False)

    # get X,y
    X = dataframe.drop(['target'], axis=1)
    y = dataframe['target']

    # one-hot
    X = utils.get_one_hot_encoded_df(X, X.columns.to_list())

    # split into train_test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    known_sensitive_attributes = ['col-0']
    attributes, sensitive_attributes, probs = utils.get_statistics_from_df(
        X_train, known_sensitive_attributes)

    #  For linear classifier, we use Logistic regression model of sklearn
    clf = LogisticRegression(random_state=0)
    clf = clf.fit(X_train, y_train)

    print("\nFeatures: ", X_train.columns.to_list())
    print("\nWeights: ", clf.coef_)
    print("\nBias:", clf.intercept_[0])
    assert len(clf.coef_[0]) == len(
        X_train.columns), "Error: wrong dimension of features and weights"

    # print("Train Accuracy Score: ", clf.score(X_train, y_train), "positive ratio: ",y_train.mean())
    # print("Test Accuracy Score: ", clf.score(X_test, y_test), "positive ratio: ",y_test.mean())
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)

    print("Train accuracy:", metrics.accuracy_score(
        y_train, predict_train), "positive ratio: ", y_train.mean())
    print("Test accuracy:", metrics.accuracy_score(
        y_test, predict_test), "positive ratio: ", y_test.mean())
    print("Train set positive prediction", predict_train.mean())
    print("Test set positive prediction", predict_test.mean())
    print()

    return clf.coef_[0], clf.intercept_[0], attributes, sensitive_attributes, probs


def init(dataset, classifier="lr", repaired=False, verbose=False, compute_equalized_odds=False, remove_column=None, fraction=0.5):
    df = dataset.get_df(repaired=repaired)

    random.seed(10)

    # discretize
    # df =  utils.get_discretized_df(df, columns_to_discretize=dataset.continuous_attributes)

    # get X,y
    X = df.drop(['target'], axis=1)
    y = df['target']

    if(remove_column is not None):
        assert isinstance(remove_column, str)
        X = X.drop([remove_column], axis=1)

    if(fraction != 1):
        non_sensitive_attributes = [
            attribute for attribute in X.columns if attribute not in dataset.known_sensitive_attributes]
        sub_columns = random.sample(non_sensitive_attributes, int(
            fraction * len(non_sensitive_attributes)))
        X = X[sub_columns + dataset.known_sensitive_attributes]

    # one-hot
    X = utils.get_one_hot_encoded_df(X, dataset.categorical_attributes)
    # X = utils.get_one_hot_encoded_df(X,X.columns.to_list())

    skf = KFold(n_splits=5, shuffle=True, random_state=10)
    skf.get_n_splits(X, y)

    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    clfs = []

    cnt = 0
    os.system("mkdir -p data/model/")

    for train, test in skf.split(X, y):

        X_trains.append(X.iloc[train])
        y_trains.append(y.iloc[train])
        X_tests.append(X.iloc[test])
        y_tests.append(y.iloc[test])

        clf = None

        if(classifier == "lr"):
            if(remove_column is None):
                store_file = "data/model/LR_" + dataset.name + "_" + \
                    str(dataset.config) + "_" + str(cnt) + \
                    "_" + str(fraction) + ".pkl"
            else:
                store_file = "data/model/LR_" + dataset.name + "_remove_" + remove_column.replace(
                    " ", "_") + "_" + str(dataset.config) + "_" + str(cnt) + "_" + str(fraction) + ".pkl"

            if(not os.path.isfile(store_file)):
                #  For linear classifier, we use Logistic regression model of sklearn
                clf = LogisticRegression(
                    class_weight='balanced', solver='liblinear', random_state=0)
                clf.fit(X_trains[-1], y_trains[-1])

                # save the classifier
                with open(store_file, 'wb') as fid:
                    pickle.dump(clf, fid)

            else:
                # Load the classifier
                with open(store_file, 'rb') as fid:
                    clf = pickle.load(fid)

        elif(classifier == "svm-linear"):
            if(remove_column is None):
                store_file = "data/model/SVM_" + dataset.name + "_" + \
                    str(dataset.config) + "_" + str(cnt) + \
                    "_" + str(fraction) + ".pkl"
            else:
                store_file = "data/model/SVM_" + dataset.name + "_remove_" + remove_column.replace(
                    " ", "_") + "_" + str(dataset.config) + "_" + str(cnt) + "_" + str(fraction) + ".pkl"
            if(not os.path.isfile(store_file)):
                #  For linear classifier, we use Logistic regression model of sklearn
                clf = SVC(kernel="linear")
                clf.fit(X_trains[-1], y_trains[-1])

                # save the classifier
                with open(store_file, 'wb') as fid:
                    pickle.dump(clf, fid)

            else:
                # Load the classifier
                with open(store_file, 'rb') as fid:
                    clf = pickle.load(fid)

        else:
            raise ValueError(classifier)

        clfs.append(clf)

        if(verbose):
            print("\nFeatures: ", X_trains[-1].columns.to_list())
            print("Number of features:", len(X_trains[-1].columns.to_list()))
            print("\nWeights: ", clf.coef_[0])
            print("\nBias:", clf.intercept_[0])
            assert len(clf.coef_[0]) == len(
                X_trains[-1].columns), "Error: wrong dimension of features and weights"

            print("Train Accuracy Score: ", clf.score(
                X_trains[-1], y_trains[-1]), "positive ratio: ", y_trains[-1].mean())
            print("Test Accuracy Score: ", clf.score(
                X_tests[-1], y_tests[-1]), "positive ratio: ", y_tests[-1].mean())

        cnt += 1

    if(compute_equalized_odds):
        return clfs, X_trains, X_tests, dataset.known_sensitive_attributes, y_trains, y_tests

    return clfs, X_trains, X_tests, dataset.known_sensitive_attributes
