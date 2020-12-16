
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
from data.objects import titanic as titanic_
import numpy as np

class linear_classifier_wrap_Wrapper():

    # TODO
    # We are given a list of weight and a bias :  w = weights of features, b = bias
    # assumption: attributes are Boolean. 

    def _find_gcd(self, list):
        x = reduce(gcd, list)
        return x


    def __init__(self, weights, attributes, sensitive_attributes, bias, negate = False, verbose = True):
        self.weights = weights
        self.bias = bias
        self.classifier = None
        self.multiplier = 30
        self.num_attributes = len(weights)
        self.sensitive_attributes = sensitive_attributes
        self.attributes = attributes


        self._convert_weights_to_integers()
        self._get_cnf(negate)
        if(verbose):
            print("Expression: ", " ".join(map(str,self.weights)), "?=" , -1*self.bias)
            # print("clauses: ", self.classifier)
            
         
        

    def _convert_weights_to_integers(self):
        # convert w, bias to integer
        _max = abs(np.array(self.weights + [self.bias])).max()
        self.weights = [int(float(weight/_max)*self.multiplier) for weight in self.weights]
        self.bias = int(float(self.bias/_max)*self.multiplier)
        



    def _get_cnf(self, negate = False):
        # apply pseudo-Boolean encoding to get the CNF of the classifier
        lits = [i+1 for i in range(len(self.attributes) + len([_var for _group in self.sensitive_attributes for _var in _group]))]
        if(negate):
            cnf = PBEnc.atmost(lits = lits, weights = self.weights,  bound = -1 * self.bias - 1, top_id = self.num_attributes)
        else:
            cnf = PBEnc.atleast(lits = lits, weights = self.weights,  bound = -1 * self.bias, top_id = self.num_attributes)
        self.classifier = cnf.clauses
        self.auxiliary_variables = [i for i in range(self.num_attributes + 1, cnf.nv + 1)]
        self.num_attributes = cnf.nv



    def check_assignment(self):
        # check whether there is a satisfying assignment to the classifier. 
        g = Glucose3() 
        for clause in self.classifier:
            g.add_clause(clause)

        print("is SAT? ",g.solve())
        print("assignment: ",g.get_model())





# supporting functions for learning weights and bias


def init_iris():
    """ 
    Returns weights, bias, features (including sensitive features), and sensitive features
    """

    # loading dataset

    target = "target"
    dataset = load_iris()
    dataset[target] = np.where(dataset[target]==2, 0, dataset[target])

    # get df

    data_df = utils.sklearn_to_df(dataset)


    # discretize
    data =  utils.get_discretized_df(data_df, columns_to_discretize=data_df.columns.to_list())

    # get X,y
    X = data.drop(['target'], axis=1)
    y = data['target']

    
    # one-hot
    X = utils.get_one_hot_encoded_df(X,X.columns.to_list())

    # split into train_test
    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.3, random_state=0)


    known_sensitive_attributes = [['sepal length (cm)_1']]
    attributes, sensitive_attributes, probs = utils.get_statistics_from_df(X_train, known_sensitive_attributes)



    #  For linear classifier, we use Logistic regression model of sklearn
    clf = LogisticRegression(random_state=0)
    clf = clf.fit(X_train, y_train)


    print("\nFeatures: ", X_train.columns.to_list())
    print("\nWeights: ", clf.coef_)
    print("\nBias:", clf.intercept_[0])
    assert len(clf.coef_[0]) == len(X_train.columns), "Error: wrong dimension of features and weights"


    print("Train Accuracy Score: ", clf.score(X_train, y_train), "positive ratio: ",y_train.mean())
    print("Test Accuracy Score: ", clf.score(X_test, y_test), "positive ratio: ",y_test.mean())


    return clf.coef_[0], clf.intercept_[0], attributes, sensitive_attributes, probs



# process adult dataset




def init_synthetic():
    filename = "data/sample.csv"

    if( os.path.isfile(filename)): 
        dataframe = pd.read_csv(filename)
    else:
        cols = 10
        rows = 200

        matrix = np.random.randint(2, size=(rows, cols))
        dataframe = pd.DataFrame.from_records(matrix)
        dataframe.columns = ['col-' + str(i) for i in range(cols-1)] + ['target']
        dataframe.to_csv(filename, index=False)


    # get X,y
    X = dataframe.drop(['target'], axis=1)
    y = dataframe['target']


    # one-hot
    X = utils.get_one_hot_encoded_df(X,X.columns.to_list())

    # split into train_test
    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.3, random_state=0)


    known_sensitive_attributes = ['col-0']    
    attributes, sensitive_attributes, probs = utils.get_statistics_from_df(X_train, known_sensitive_attributes)



    #  For linear classifier, we use Logistic regression model of sklearn
    clf = LogisticRegression(random_state=0)
    clf = clf.fit(X_train, y_train)


    print("\nFeatures: ", X_train.columns.to_list())
    print("\nWeights: ", clf.coef_)
    print("\nBias:", clf.intercept_[0])
    assert len(clf.coef_[0]) == len(X_train.columns), "Error: wrong dimension of features and weights"
    

    # print("Train Accuracy Score: ", clf.score(X_train, y_train), "positive ratio: ",y_train.mean())
    # print("Test Accuracy Score: ", clf.score(X_test, y_test), "positive ratio: ",y_test.mean())
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)

    
    print("Train accuracy:",metrics.accuracy_score(y_train, predict_train), "positive ratio: ",y_train.mean())
    print("Test accuracy:",metrics.accuracy_score(y_test, predict_test), "positive ratio: ",y_test.mean())
    print("Train set positive prediction",predict_train.mean())
    print("Test set positive prediction",predict_test.mean())
    print()
    

    return clf.coef_[0], clf.intercept_[0], attributes, sensitive_attributes, probs



def init(dataset, classifier = "lr", repaired=False, verbose = False, compute_equalized_odds = False):
    df = dataset.get_df(repaired=repaired)

    # discretize
    # df =  utils.get_discretized_df(df, columns_to_discretize=dataset.continuous_attributes)

    # get X,y
    X = df.drop(['target'], axis=1)
    y = df['target']

    # one-hot
    X = utils.get_one_hot_encoded_df(X,dataset.categorical_attributes)
    # X = utils.get_one_hot_encoded_df(X,X.columns.to_list())
    

    skf = KFold(n_splits=5, shuffle=True, random_state=10)
    skf.get_n_splits(X, y)

    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    clfs = []
    
    for train, test in skf.split(X, y):

        X_trains.append(X.iloc[train])
        y_trains.append(y.iloc[train])
        X_tests.append(X.iloc[test])
        y_tests.append(y.iloc[test])


        clf = None

        if(classifier == "lr"):
            #  For linear classifier, we use Logistic regression model of sklearn
            clf = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0)
        elif(classifier == "svm-linear"):
            clf = SVC(kernel="linear")
        else:
            raise ValueError(classifier)

        clf = clf.fit(X_trains[-1], y_trains[-1])
        clfs.append(clf)

        if(verbose):
            print("\nFeatures: ", X_trains[-1].columns.to_list())
            print("Number of features:", len(X_trains[-1].columns.to_list()))
            print("\nWeights: ", clf.coef_[0])
            print("\nBias:", clf.intercept_[0])
            assert len(clf.coef_[0]) == len(X_trains[-1].columns), "Error: wrong dimension of features and weights"


            print("Train Accuracy Score: ", clf.score(X_trains[-1], y_trains[-1]), "positive ratio: ",y_trains[-1].mean())
            print("Test Accuracy Score: ", clf.score(X_tests[-1], y_tests[-1]), "positive ratio: ",y_tests[-1].mean())

    
    if(compute_equalized_odds):
        return clfs, X_trains, X_tests, dataset.known_sensitive_attributes, y_trains, y_tests
    

    return clfs, X_trains, X_tests, dataset.known_sensitive_attributes











