from pyrulelearn.imli import imli
import justicia.utils as utils
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import metrics
import numpy as np


def init(dataset, repaired=False, verbose = False, negated = False, compute_equalized_odds = False):

    df = dataset.get_df(repaired=repaired)

    # discretize
    data =  utils.get_discretized_df(df, columns_to_discretize=dataset.continuous_attributes, verbose=verbose)

    # get X,y
    X = data.drop(['target'], axis=1)
    y = data['target']


    # one-hot
    X = utils.get_one_hot_encoded_df(X,X.columns.to_list(), verbose=verbose)


    skf = KFold(n_splits=5, shuffle=True, random_state=10)
    skf.get_n_splits(X, y)

    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    clfs = []
    clf_negs = []

    for train, test in skf.split(X, y):

        X_trains.append(X.iloc[train])
        y_trains.append(y.iloc[train])
        X_tests.append(X.iloc[test])
        y_tests.append(y.iloc[test])

        

        #  For linear classifier, we use Logistic regression model of sklearn
        clf = imli(num_clause=2, data_fidelity=10, work_dir="data/", rule_type="CNF", verbose=False)
            
        clf.fit(X_trains[-1].values, y_trains[-1].values)


        clf_neg = None
        if(negated):
            clf_neg = imli(num_clause=2, data_fidelity=10, work_dir="data/", rule_type="CNF", verbose=False)
            clf_neg.fit(X_trains[-1].values, np.array([1 - val for val in y_trains[-1].values]))


        clfs.append(clf)
        clf_negs.append(clf_neg)


        if(verbose):
            print("\nFeatures: ", X_trains[-1].columns.to_list())
            print("Number of features:", len(X_trains[-1].columns.to_list()))
            print("\nlearned rule:")
            print(clf.get_rule(X_trains[-1].columns.to_list()))
        
        
            
        if(verbose):
            print("\nTrain Accuracy Score: ", metrics.accuracy_score(clf.predict(X_trains[-1].values, y_trains[-1].values),y_trains[-1].values) , "positive ratio: ",y_trains[-1].mean())
            print("Test Accuracy Score: ", metrics.accuracy_score(clf.predict(X_tests[-1].values, y_tests[-1].values),y_tests[-1].values), "positive ratio: ",y_tests[-1].mean())
        
    if(compute_equalized_odds):
        return clfs, clf_negs, X_trains, X_tests, dataset.known_sensitive_attributes, y_trains, y_tests
       
    return clfs, clf_negs, X_trains, X_tests, dataset.known_sensitive_attributes

