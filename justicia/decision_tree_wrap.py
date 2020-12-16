# this script learns a decision tree classifier, 
# generate a DNF rule for the positive class (in the standard binary classification setting),
# negate the DNF to CNF


import sklearn.datasets
from sklearn import tree
import numpy as np
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics 
# from fairness_comparison.fairness.data.objects.ProcessedData import ProcessedData
# from fairness_comparison.fairness.data.objects.Adult import Adult
import pandas as pd
import justicia.utils as utils
import os
from data.objects import titanic as titanic_   
from data.objects import adult as adult_
from data.objects import ricci as ricci_
# from fairness_comparison.fairness.data.objects import Ricci
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV



class dtWrapper():

    def __init__(self, DecisionTreeClassifier, data_features_original, sensitive_attributes, negate = False,  verbose = True):

        self.nodes = []
        self.paths = [] # a DNF formula, only capture when the leaf is 1
        self.tree_to_code(DecisionTreeClassifier, data_features_original, negate, verbose = verbose)

        self.num_attributes = len(self.nodes)
        self.attributes = [] # does not include variables related to sensitive attributes
        self.sensitive_attributes = []
        self.attribute_variable_map = {}
        self.variable_attribute_map = {}
        self.auxiliary_variables = []
        self.classifier = []
        
        self._do_map_from_node_to_attribute_index(sensitive_attributes, verbose)
        self._construct_CNF()


    def _do_map_from_node_to_attribute_index(self, sensitive_attributes, verbose):
        """  
        TODO make it more efficient
        """
        cnt = 1
        _sensitive_attributes = [_var for _group in sensitive_attributes for _var in _group]
        
        self.sensitive_attributes = [[] for _ in sensitive_attributes]
        # appearance of sensitive attributes
        _visited_sensitive_attributes = []
        for node in self.nodes:
            feature, threshold = node
            # if feature is one-hot encoded, it may contain specific prefix


            if(feature in _sensitive_attributes): # when sensitive feature is binary
                _visited_sensitive_attributes.append(feature)
                self.attribute_variable_map[(feature, threshold)] = cnt
                self.variable_attribute_map[cnt] = (feature, threshold)
                for i in range(len(sensitive_attributes)):
                    if(feature in sensitive_attributes[i]):
                        self.sensitive_attributes[i].append(cnt)
                        
    
                cnt += 1
            else:
                self.attribute_variable_map[(feature, threshold)] = cnt
                self.variable_attribute_map[cnt] = (feature, threshold)
                self.attributes.append(cnt)
                cnt += 1

        
        # It may happen that all sensitive attributes are not preset in the decision tree
        for _sensitive_attribute in _sensitive_attributes:
            if(_sensitive_attribute not in _visited_sensitive_attributes):
                self.attribute_variable_map[(_sensitive_attribute, 0.5)] = cnt
                self.variable_attribute_map[cnt] = (_sensitive_attribute, 0.5)
                for i in range(len(sensitive_attributes)):
                    if(_sensitive_attribute in sensitive_attributes[i]):
                        self.sensitive_attributes[i].append(cnt)
                self.num_attributes += 1
                cnt += 1
            # else:
            #     print(_sensitive_attribute, "with assigned variable" ,self.attribute_variable_map[(_sensitive_attribute, 0.5)], "is in the tree")
            
        self._max_attribute_index = cnt - 1

        if(verbose):
            print("\nattribute to variable map ->")
            print(self.attribute_variable_map)
            print("\nvarible to attribute map ->")
            print(self.variable_attribute_map)
            
        assert self.num_attributes == len(self.attribute_variable_map), "Error in mapping attributes to index"

    def _construct_CNF(self):
        """  
        Applying De Morgan's law
        """
        # first construct a DNF formula from paths leading to 0 of the decision trees and then negate it to construct a CNF
        # self.path is already in DNF for class = 0

        for path in self.paths:
            clause = []
            for (name, threshold, flag) in path:
                clause.append(self.attribute_variable_map[(name, threshold)]  * -1 * flag) # -1 is used to negate the variable
            self.classifier.append(clause)


    def compute_probability(self, data, verbose = False):
        """
        inputs:
            data is a dataframe
        outputs:
            probabilities of attributes: a dictionary
        """
        if(verbose):
            print("\nCalculated probability")
        probs = {}
        for key in self.attribute_variable_map:
            feature, threshold = key
            probs[self.attribute_variable_map[key]] = round((data[feature] <= threshold).mean(),3)
            if(probs[self.attribute_variable_map[key]] == 0):
                probs[self.attribute_variable_map[key]] = 0.001
            elif(probs[self.attribute_variable_map[key]] == 1):
                probs[self.attribute_variable_map[key]] = 0.999
            if(verbose):
                print(feature, "<=" ,threshold, " has probability: " , probs[self.attribute_variable_map[key]], " and assigned variable is:", self.attribute_variable_map[key])
        return probs


    def compute_pair_wise_correlation(self, data, probs, verbose = True):


        attribute_cnt = self._max_attribute_index
        constraints = []
        
        for key_1 in self.attribute_variable_map:
            # if(self.attribute_variable_map[key_1] not in self.attributes):
            #         continue
                
            for key_2 in self.attribute_variable_map:   
                if(key_1 == key_2):
                    continue
                
                # if(self.attribute_variable_map[key_2] not in self.attributes):
                #     continue
                

                feature_1, threshold_1 = key_1
                feature_2, threshold_2 = key_2
                
                attribute_cnt += 1
                self.attributes.append(attribute_cnt)
                constraints.append((attribute_cnt, self.attribute_variable_map[key_1], self.attribute_variable_map[key_2]))
                probs[attribute_cnt] = round(((data[feature_1] <= threshold_1) & (data[feature_2] <= threshold_2)).mean(),3)

                attribute_cnt += 1
                self.attributes.append(attribute_cnt)
                constraints.append((attribute_cnt, self.attribute_variable_map[key_1], -1 * self.attribute_variable_map[key_2]))
                probs[attribute_cnt] = round(((data[feature_1] <= threshold_1) & (data[feature_2] > threshold_2)).mean(),3)

                attribute_cnt += 1
                self.attributes.append(attribute_cnt)
                constraints.append((attribute_cnt, -1 * self.attribute_variable_map[key_1], self.attribute_variable_map[key_2]))
                probs[attribute_cnt] = round(((data[feature_1] > threshold_1) & (data[feature_2] <= threshold_2)).mean(),3)

                attribute_cnt += 1
                self.attributes.append(attribute_cnt)
                constraints.append((attribute_cnt, -1 * self.attribute_variable_map[key_1], -1 * self.attribute_variable_map[key_2]))
                probs[attribute_cnt] = round(((data[feature_1] > threshold_1) & (data[feature_2] > threshold_2)).mean(),3)


        # for constraint in correlation_constraints:
        #     self.formula += self._construct_clause([-1 * constraint[0], constraint[1]])
        #     self.formula += self._construct_clause([-1 * constraint[0], constraint[2]])
        #     # self.formula += self._construct_clause([constraint[0], -1 * constraint[1], -1 * constraint[2]])


            #     break
            # break

        # print(constraints)
        # print("\n")
        # print(probs)
        
        return probs, constraints


             

        





    def tree_to_code(self, tree, feature_names, negate, verbose = True):

        
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        if(verbose):
            print("\nLearned tree -->\n")
            print("def tree({}):".format(", ".join(feature_names)))

        def recurse(node, depth, path):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                if((name, threshold) not in self.nodes):
                    self.nodes.append((name,  threshold)) # stores the description of the node (a Boolean variable)
                if(verbose):
                    print("{}if {} <= {}:".format(indent, name, threshold))
                recurse(tree_.children_left[node], depth + 1, path + [(name,  threshold, 1)])
                if(verbose):
                    print("{}else:  # NEG if {} <= {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1, path + [(name, threshold, -1)])
            else:
                class_label = np.argmax(tree_.value[node][0])
                if(class_label == 0 and not negate):
                    self.paths.append(path)
                if(class_label == 1 and negate):
                    self.paths.append(path)
                
                if(verbose):
                    print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))

        recurse(0, 1, [])

        
        if(verbose):
            print("\n\n\nnodes ->")
            for node in self.nodes:
                print(node)

            print("\n\n\nDNF paths ->")            
            for path in self.paths:
                print(path)
        

    


# supporting functions for learning decision trees

def _get_sensitive_attibutes(known_sensitive_features, features):
    """ 
    Return sensitive attributes in appropriate format
    """

    # Extract new names of sensitive attributes
    _sensitive_attributes = {} # it is a map because each entry contains all one-hot encoded variables
    for _column in features:
        if("_" in _column and _column.split("_")[0] in known_sensitive_features):
            if(_column.split("_")[0] not in _sensitive_attributes):
                _sensitive_attributes[_column.split("_")[0]] = [_column]
            else:
                _sensitive_attributes[_column.split("_")[0]].append(_column)
        elif(_column in known_sensitive_features):
            if(_column not in _sensitive_attributes):
                _sensitive_attributes[_column] = [_column]
            else:
                _sensitive_attributes[_column].append(_column)


    # Finally make a 2d list
    sensitive_attributes = []
    for key in _sensitive_attributes:
        sensitive_attributes.append(_sensitive_attributes[key])


    return sensitive_attributes
    

def init_iris():

    # dataset.data is a np matrix, 
    # dataset.target is a np array
    # dataset['features] is the list of features in the original dataset

    # prepare iris dataset for binary classification
    target = "target"
    dataset = sklearn.datasets.load_iris()
    dataset[target] = np.where(dataset[target]==2, 0, dataset[target])

    # get df
    dataset = utils.sklearn_to_df(dataset)

    index_of_sensitive_features = 0

    # discretize sensitive attributes
    data =  utils.get_discretized_df(dataset, columns_to_discretize=[dataset.columns.to_list()[index_of_sensitive_features]])

    # get X,y
    X = data.drop(['target'], axis=1)
    y = data['target']


    # one-hot
    X = utils.get_one_hot_encoded_df(X,X.columns.to_list())

    # split into train_test
    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.3, random_state=0)


 
    
    # Extract new names of sensitive attributes
    _sensitive_attributes = {} # it is a map because each entry contains all one-hot encoded variables
    for _column in X_train.columns.to_list():
        if("_" in _column and _column.split("_")[0] in dataset.columns.to_list()[index_of_sensitive_features]):
            if(_column.split("_")[0] not in _sensitive_attributes):
                _sensitive_attributes[_column.split("_")[0]] = [_column]
            else:
                _sensitive_attributes[_column.split("_")[0]].append(_column)
        elif(_column in dataset.columns.to_list()[index_of_sensitive_features]):
            if(_column not in _sensitive_attributes):
                _sensitive_attributes[_column] = [_column]
            else:
                _sensitive_attributes[_column].append(_column)
    
    # Finally make a 2d list
    sensitive_attributes = []
    for key in _sensitive_attributes:
        sensitive_attributes.append(_sensitive_attributes[key])
        
        
    

    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)

    
    print("Train accuracy:",metrics.accuracy_score(y_train, predict_train), "positive ratio: ",y_train.mean())
    print("Test accuracy:",metrics.accuracy_score(y_test, predict_test), "positive ratio: ",y_test.mean())
    print("Train set positive prediction",predict_train.mean())
    print("Test set positive prediction",predict_test.mean())
    
    
    return clf, X_train.columns.to_list(), sensitive_attributes, X_train, X_test





def init_synthetic():
    filename = "data/sample.csv"

    if( os.path.isfile(filename)): 
        dataframe = pd.read_csv(filename)
    else:

        cols = 4
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


    sensitive_attributes = _get_sensitive_attibutes(['col-0'],X_train.columns.to_list())

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)

    
    print("Train accuracy:",metrics.accuracy_score(y_train, predict_train), "positive ratio: ",y_train.mean())
    print("Test accuracy:",metrics.accuracy_score(y_test, predict_test), "positive ratio: ",y_test.mean())
    print("Train set positive prediction",predict_train.mean())
    print("Test set positive prediction",predict_test.mean())
    


    return clf, sensitive_attributes, X_train, X_test





def init(dataset, repaired=False, verbose = False, compute_equalized_odds = False):
    
    df = dataset.get_df(repaired=repaired)



    # get X,y
    X = df.drop(['target'], axis=1)
    y = df['target']



    # one-hot
    X = utils.get_one_hot_encoded_df(X,dataset.categorical_attributes, verbose=verbose)


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
    
    

        # apply gridsearch
        param_grid = {'max_depth': np.arange(2, 10)}
        grid_tree = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
        grid_tree.fit(X_trains[-1], y_trains[-1])
        tree_preds = grid_tree.predict_proba(X_tests[-1])[:, 1]
        tree_performance = roc_auc_score(y_tests[-1], tree_preds)
        clf = grid_tree.best_estimator_
        clfs.append(clf)


        # clf = tree.DecisionTreeClassifier()
        # clf = clf.fit(X_train, y_train)
        predict_train = clf.predict(X_trains[-1])
        predict_test = clf.predict(X_tests[-1])
    

        if(verbose):
            print("\nTraining result =>")
            print('dt: Area under the ROC curve = {}'.format(tree_performance))
            
            # getting the best models:
            print(grid_tree.best_params_)
            
            print("\nTrain accuracy:",metrics.accuracy_score(y_trains[-1], predict_train), "positive ratio: ",y_trains[-1].mean())
            print("Test accuracy:",metrics.accuracy_score(y_tests[-1], predict_test), "positive ratio: ",y_tests[-1].mean())
            print("Train set positive prediction",predict_train.mean())
            print("Test set positive prediction",predict_test.mean())


       




    if(compute_equalized_odds):
        return clfs, X_trains, X_tests, dataset.known_sensitive_attributes, y_trains, y_tests
    
    return clfs, X_trains, X_tests, dataset.known_sensitive_attributes
    




















    



