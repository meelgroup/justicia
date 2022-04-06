from sklearn.neighbors import KernelDensity
from scipy.stats import moment
import math
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
import matplotlib
import pandas as pd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from feature_engine import discretisers as dsc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import _tree
from scipy.stats import norm


# pgmpy
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

# notears
import igraph as ig
from numpy import genfromtxt
import networkx as nx

# draw
import matplotlib.pyplot as plt


def call_pgmpy(data):
    print(data)
    est = PC(data=data)
    estimated_model = est.estimate(variant='orig', max_cond_vars=5)
    return estimated_model


def call_notears(data, regularizer=0.01, timeout=100, verbose=True, filename="temp"):
    """
        Learn a DAG from data.
        Output is a list of tuples (a,b) where b is conditionally dependent on a
    """
    assert isinstance(data, pd.DataFrame), "data must be a Pandas dataframe"

    filename = "./" + filename

    data.to_csv(filename + "X.csv", index=False, header=None)
    cmd = "timeout " + str(timeout) + " notears_linear --lambda1 " + str(
        regularizer) + " --W_path " + filename + "W_est.csv " + filename + "X.csv"
    os.system(cmd)

    if(not os.path.isfile(filename + 'W_est.csv')):
        print("Notears_linear did not terminate")
        return []

    # construct weighted graphs
    adjacency_matrix = genfromtxt(filename + 'W_est.csv', delimiter=',')

    # remove temp files
    os.system("rm " + filename + "W_est.csv " + filename + "X.csv")

    # graph
    iG = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist())

    edges = []
    for a, b in iG.get_edgelist():
        edges.append((data.columns[a], data.columns[b]))

    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


def select_encoding(model_name, encoding, verbose):

    if(model_name in ["linear-model", "decision-tree"]):
        forwarded_encoding_dic = {
            'Learn': 'Learn-efficient',
            'Learn-dependency': 'Learn-efficient-dependency',
            'Learn-correlation': 'Learn-efficient-correlation'
        }

        if(encoding in forwarded_encoding_dic):
            if(verbose):
                print(encoding, "encoding is solved based on",
                      forwarded_encoding_dic[encoding], "encoding")
            encoding = forwarded_encoding_dic[encoding]
    elif(model_name in ["CNF"]):
        forwarded_encoding_dic = {
            'Learn-efficient': 'Learn',
            'Learn-efficient-correlation': 'Learn-correlation',
            'Learn-efficient-dependency': 'Learn-dependency'
        }

        if(encoding in forwarded_encoding_dic):
            if(verbose):
                print(encoding, "encoding is solved based on",
                      forwarded_encoding_dic[encoding], "encoding")
            encoding = forwarded_encoding_dic[encoding]

    return encoding


def get_population_model_fairsquare_format(data, sensitive_attribute):

    divide = True

    s = 'def popModel():\n'
    mean = data[sensitive_attribute].mean()
    mean = round(mean, 7)
    s += "\t" + sensitive_attribute + \
        " = step([(0,0.5," + str(round(1-mean, 7)) + \
        "), (0.5,1," + str(mean) + ")])\n"

    if(divide):
        s += "\tif " + str(sensitive_attribute) + " < 0.5:\n"
        for column in data.columns:
            if(column == sensitive_attribute):
                continue
            if(len(data[column].unique()) == 2):
                mean = data[data[sensitive_attribute] < 0.5][column].mean()
                mean = round(mean, 7)
                s += "\t\t" + column + \
                    " = step([(0,0.5," + str(round(1-mean, 7)) + \
                    "), (0.5,1," + str(mean) + ")])\n"
            else:
                mu, std = norm.fit(
                    data[data[sensitive_attribute] < 0.5][column])
                mu = round(mu, 7)
                std = round(std, 7)
                if(std == 0):
                    std = 0.00001
                s += "\t\t" + column + \
                    " = gaussian(" + str(mu) + "," + str(std) + ")\n"

        s += "\telse:\n"
        for column in data.columns:
            if(column == sensitive_attribute):
                continue
            if(len(data[column].unique()) == 2):
                mean = data[data[sensitive_attribute] >= 0.5][column].mean()
                mean = round(mean, 7)
                s += "\t\t" + column + \
                    " = step([(0,0.5," + str(round(1-mean, 7)) + \
                    "), (0.5,1," + str(mean) + ")])\n"
            else:
                mu, std = norm.fit(
                    data[data[sensitive_attribute] >= 0.5][column])
                mu = round(mu, 7)
                std = round(std, 7)
                if(std == 0):
                    std = 0.00001
                s += "\t\t" + column + \
                    " = gaussian(" + str(mu) + "," + str(std) + ")\n"
    else:
        for column in data.columns:
            if(column == sensitive_attribute):
                continue
            if(len(data[column].unique()) == 2):
                mean = data[column].mean()
                mean = round(mean, 7)
                s += "\t" + column + \
                    " = step([(0,0.5," + str(round(1-mean, 7)) + \
                    "), (0.5,1," + str(mean) + ")])\n"
            else:
                mu, std = norm.fit(data[column])
                mu = round(mu, 7)
                std = round(std, 7)
                if(std == 0):
                    std = 0.00001
                s += "\t" + column + \
                    " = gaussian(" + str(mu) + "," + str(std) + ")\n"
    s += "\tsensitiveAttribute(" + sensitive_attribute + " < 0.5)\n\n\n\n"
    return s


def get_population_model_verifair_format(data, sensitive_attribute):

    divide = True

    s = "from ..benchmarks.fairsquare.helper import *\n\n\ndef sample(flag):\n\n"

    mean = data[sensitive_attribute].mean()
    mean = round(mean, 7)
    s += "\t" + sensitive_attribute + \
        " = step([(0,1," + str(round(1-mean, 7)) + "), (1,2," + str(mean) + ")])\n"

    if(divide):
        s += "\tif " + str(sensitive_attribute) + " < 0.5:\n"
        for column in data.columns:
            if(column == sensitive_attribute):
                continue
            if(len(data[column].unique()) == 2):
                mean = data[data[sensitive_attribute] < 0.5][column].mean()
                mean = round(mean, 7)
                s += "\t\t" + column + \
                    " = step([(0,1," + str(round(1-mean, 7)) + \
                    "), (1,2," + str(mean) + ")])\n"
            else:
                mu, std = norm.fit(
                    data[data[sensitive_attribute] < 0.5][column])
                mu = round(mu, 7)
                std = round(std, 7)
                if(std == 0):
                    std = 0.00001
                s += "\t\t" + column + \
                    " = gaussian(" + str(mu) + "," + str(std) + ")\n"

        s += "\telse:\n"
        for column in data.columns:
            if(column == sensitive_attribute):
                continue
            if(len(data[column].unique()) == 2):
                mean = data[data[sensitive_attribute] >= 0.5][column].mean()
                mean = round(mean, 7)
                s += "\t\t" + column + \
                    " = step([(0,1," + str(round(1-mean, 7)) + \
                    "), (1,2," + str(mean) + ")])\n"
            else:
                mu, std = norm.fit(
                    data[data[sensitive_attribute] >= 0.5][column])
                mu = round(mu, 7)
                std = round(std, 7)
                if(std == 0):
                    std = 0.00001
                s += "\t\t" + column + \
                    " = gaussian(" + str(mu) + "," + str(std) + ")\n"
    else:
        for column in data.columns:
            if(column == sensitive_attribute):
                continue
            if(len(data[column].unique()) == 2):
                mean = data[column].mean()
                mean = round(mean, 7)
                s += "\t" + column + \
                    " = step([(0,1," + str(round(1-mean, 7)) + \
                    "), (1,2," + str(mean) + ")])\n"
            else:
                mu, std = norm.fit(data[column])
                mu = round(mu, 7)
                std = round(std, 7)
                if(std == 0):
                    std = 0.00001
                s += "\t" + column + \
                    " = gaussian(" + str(mu) + "," + str(std) + ")\n"
    s += "\tsensitiveAttribute(" + sensitive_attribute + " < 1, flag)\n\n\n\n"
    return s


def attribute_raname_fairsquare_format(data):
    attributes = []
    for column in data.columns:
        column = column.strip().replace(" ", "_")
        column = column.replace("-", "_")
        column = column.replace("=", "_eq_")
        column = column.replace(">", "_g_")
        column = column.replace("<", "_l_")
        column = column.replace("(", "_lpar_")
        column = column.replace(")", "_rpar_")
        column = column.replace("/", "_or_")
        attributes.append(column)
    data.columns = attributes
    return data


def linear_model_fairsquare_format(clf, dataset_name, feature_names):
    coef = clf.coef_[0]
    bias = clf.intercept_[0]
    assert len(coef) == len(
        feature_names), "Error dimension of input attributes"
    s = "def F():\n\tt = "
    total_features = len(feature_names)
    for idx in range(total_features):
        s += str(round(coef[idx], 7)) + " * " + feature_names[idx] + " + "
    s += str(round(bias, 7))

    s += "\n\n\n"
    if(dataset_name == "titanic" or dataset_name == "adult" or dataset_name == "german"):
        s += "\tfairnessTarget(t < 0)\n\n"
    else:
        s += "\tfairnessTarget(t >= 0)\n\n"

    return s


def linear_model_verifair_format(clf, dataset_name, feature_names):
    coef = clf.coef_[0]
    bias = clf.intercept_[0]
    assert len(coef) == len(
        feature_names), "Error dimension of input attributes"
    s = "\tt = "
    total_features = len(feature_names)
    for idx in range(total_features):
        s += str(round(coef[idx], 7)) + " * " + feature_names[idx] + " + "
    s += str(round(bias, 7))

    s += "\n\n\n"
    if(dataset_name == "titanic" or dataset_name == "adult" or dataset_name == "german"):
        s += "\treturn int(t < 0)\n\tfairnessTarget(t < 0)\n\n"
    else:
        s += "\treturn int(t >= 0)\n\tfairnessTarget(t >= 0)\n\n"

    return s


def tree_to_code_fairsquare_format(tree, dataset_name, feature_names):

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    s = "def tree({}):".format(", ".join(feature_names)) + "\n\n"
    s = "def F():\n"
    # print("\nLearned tree -->\n")
    # print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth, s):
        indent = "\t" * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node], 7)
            s = s + "{}if {} <= {}:".format(indent, name, threshold) + "\n"
            # print("{}if {} <= {}:".format(indent, name, threshold))
            s = recurse(tree_.children_left[node], depth + 1, s)
            s = s + "{}else:".format(indent) + "\n"
            # print("{}else:".format(indent))
            s = recurse(tree_.children_right[node], depth + 1, s)
        else:
            s = s + "{}t = {}".format(indent,
                                      np.argmax(tree_.value[node][0])) + "\n"
            # print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))

        return s

    s = recurse(0, 1, s)
    if(dataset_name == "titanic" or dataset_name == "adult" or dataset_name == "german"):
        s += "\tfairnessTarget(t < 0.5)\n\n"
    else:
        s += "\tfairnessTarget(t >= 0.5)\n\n"

    return s


def tree_to_code_verifair_format(tree, dataset_name, feature_names):

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    s = "def tree({}):".format(", ".join(feature_names)) + "\n\n"
    s = ""
    # print("\nLearned tree -->\n")
    # print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth, s):
        indent = "\t" * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node], 7)
            s = s + "{}if {} <= {}:".format(indent, name, threshold) + "\n"
            # print("{}if {} <= {}:".format(indent, name, threshold))
            s = recurse(tree_.children_left[node], depth + 1, s)
            s = s + "{}else:".format(indent) + "\n"
            # print("{}else:".format(indent))
            s = recurse(tree_.children_right[node], depth + 1, s)
        else:
            s = s + "{}t = {}".format(indent,
                                      np.argmax(tree_.value[node][0])) + "\n"
            # print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))

        return s

    s = recurse(0, 1, s)
    if(dataset_name == "titanic" or dataset_name == "adult" or dataset_name == "german"):
        s += "\treturn int(t < 0.5)\n\tfairnessTarget(t < 0.5)\n\n"
    else:
        s += "\treturn int(t > 0.5)\n\tfairnessTarget(t >= 0.5)\n\n"

    return s


def aif360_cv_split(dataset, num_of_splits, shuffle=False, seed=None):

    n = dataset.features.shape[0]

    kf = KFold(n_splits=num_of_splits, shuffle=shuffle, random_state=10)
    kf.get_n_splits(dataset.features, dataset.labels)

    folds_train = [dataset.copy() for _ in range(num_of_splits)]
    folds_test = [dataset.copy() for _ in range(num_of_splits)]
    cnt = 0
    for train, test in kf.split(dataset.features, dataset.labels):
        fold = folds_train[cnt]
        fold.features = dataset.features[train]
        fold.labels = dataset.labels[train]
        fold.scores = dataset.scores[train]
        fold.protected_attributes = dataset.protected_attributes[train]
        fold.instance_weights = dataset.instance_weights[train]
        fold.instance_names = list(
            map(str, np.array(dataset.instance_names)[train]))
        fold.metadata = fold.metadata.copy()
        fold.metadata.update({
            'transformer': '{}.split'.format(type(dataset).__name__),
            'params': {'num_of_splits': num_of_splits,
                       'shuffle': shuffle},
            'previous': [dataset]
        })

        fold = folds_test[cnt]
        fold.features = dataset.features[test]
        fold.labels = dataset.labels[test]
        fold.scores = dataset.scores[test]
        fold.protected_attributes = dataset.protected_attributes[test]
        fold.instance_weights = dataset.instance_weights[test]
        fold.instance_names = list(
            map(str, np.array(dataset.instance_names)[test]))
        fold.metadata = fold.metadata.copy()
        fold.metadata.update({
            'transformer': '{}.split'.format(type(dataset).__name__),
            'params': {'num_of_splits': num_of_splits,
                       'shuffle': shuffle},
            'previous': [dataset]
        })

        cnt += 1

    return folds_train, folds_test


def get_scaled_df(X):
    # scale the feature values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X


def sklearn_to_df(sklearn_dataset):
    """ 
    Convert sklearn dataset to a dataframe, the class-label is renamed to "target"
    """
    df = pd.DataFrame(sklearn_dataset.data,
                      columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


def get_discretized_df(data, columns_to_discretize=None, verbose=False):
    """ 
    returns train_test_splitted and discretized df
    """

    if(columns_to_discretize is None):
        columns_to_discretize = data.columns.to_list()

    if(verbose):
        print("Applying discretization\nAttribute bins")
    for variable in columns_to_discretize:
        bins = min(4, len(data[variable].unique()))
        if(verbose):
            print(variable, bins)

        # set up the discretisation transformer
        disc = dsc.EqualWidthDiscretiser(bins=bins, variables=[variable])

        # fit the transformer
        disc.fit(data)

        if(verbose):
            print(disc.binner_dict_)

        # transform the data
        data = disc.transform(data)
        if(verbose):
            print(data[variable].unique())

    return data


def get_one_hot_encoded_df(df, columns_to_one_hot, verbose=False):
    """  
    Apply one-hot encoding on categircal df and return the df
    """
    if(verbose):
        print("\n\nApply one-hot encoding on categircal attributes")
    for column in columns_to_one_hot:
        if(column not in df.columns.to_list()):
            if(verbose):
                print(column, " is not considered in classification")
            continue

        # Apply when there are more than two categories or the binary categories are string objects.
        unique_categories = df[column].unique()
        if(len(unique_categories) > 2):
            one_hot = pd.get_dummies(df[column])
            if(verbose):
                print(column, " has more than two unique categories",
                      one_hot.columns.to_list())

            if(len(one_hot.columns) > 1):
                one_hot.columns = [column + "_" +
                                   str(c) for c in one_hot.columns]
            else:
                one_hot.columns = [column for c in one_hot.columns]
            df = df.drop(column, axis=1)
            df = df.join(one_hot)
        else:
            # print(column, unique_categories)
            if(0 in unique_categories and 1 in unique_categories):
                if(verbose):
                    print(column, " has categories 1 and 0")

                continue
            df[column] = df[column].map(
                {unique_categories[0]: 0, unique_categories[1]: 1})
            if(verbose):
                print("Applying following mapping on attribute", column, "=>",
                      unique_categories[0], ":",  0, "|", unique_categories[1], ":", 1)
    if(verbose):
        print("\n")
    return df


def get_statistics_from_df(data, known_sensitive_attributes, verbose=False):
    # return non-sensitive attributes, sensitive attributes, probabilities of different attributes

    probs = {}
    sensitive_attributes = [[] for _ in known_sensitive_attributes]
    attributes = []  # contains non-sensitive attributes
    attribute_variable_map = {}
    variable_attribute_map = {}

    column_info = {}
    for idx, column in enumerate(data.columns):
        _max_value = data[column].max()
        column_info[idx] = ("Bin", column, _max_value)
        probs[idx + 1] = round((data[column] == _max_value).mean(), 3)
        if(probs[idx + 1] == 0):
            probs[idx + 1] = 0.001
        elif(probs[idx + 1] == 1):
            probs[idx + 1] = 0.999
        elif(math.isnan(probs[idx + 1])):
            probs[idx + 1] = 0.5
        assert 0 <= probs[idx + 1] and probs[idx +
                                             1] <= 1, "Error in calculating probabilities"

        # check for sensitive attributes
        _is_sensitive_attribute = False
        for group_idx in range(len(known_sensitive_attributes)):
            if(data.columns[idx].split("_")[0] in known_sensitive_attributes[group_idx]):
                sensitive_attributes[group_idx].append(idx + 1)
                _is_sensitive_attribute = True
                variable_attribute_map[idx +
                                       1] = (data.columns[idx], _max_value)
                attribute_variable_map[(
                    data.columns[idx], _max_value)] = idx + 1

                break

            elif(data.columns[idx] in known_sensitive_attributes[group_idx]):
                sensitive_attributes[group_idx].append(idx + 1)
                variable_attribute_map[idx +
                                       1] = (data.columns[idx], _max_value)
                attribute_variable_map[(
                    data.columns[idx], _max_value)] = idx + 1
                _is_sensitive_attribute = True
                break

        # otherwise non-sensitive attributes
        if(not _is_sensitive_attribute):
            attributes.append(idx + 1)
            variable_attribute_map[idx + 1] = (data.columns[idx], _max_value)
            attribute_variable_map[(data.columns[idx], _max_value)] = idx + 1

    if(verbose > 1):
        print("\nvariable to attribute map ->")
        print(variable_attribute_map)
        print("\nattribute to variable map ->")
        print(attribute_variable_map)
        print("\n\n\n")

    return attributes, sensitive_attributes, probs, variable_attribute_map, column_info


def calculate_probs_linear_classifier_wrap(data, column_info, selected_columns=[]):
    # print(selected_columns)
    compute_for_all = True
    if(len(selected_columns) != 0):
        compute_for_all = False

    probs = {}
    _attributes_cnt = 1
    for idx in range(len(data.columns)):

        if(column_info[idx][0] == "Bin"):
            if(not compute_for_all):
                if(_attributes_cnt not in selected_columns):
                    _attributes_cnt += 1
                    continue

            probs[_attributes_cnt] = round(
                (data[data.columns[idx]] == column_info[idx][2]).mean(), 3)
            if(probs[_attributes_cnt] == 0):
                probs[_attributes_cnt] = 0.001
            elif(probs[_attributes_cnt] == 1):
                probs[_attributes_cnt] = 0.999
            elif(math.isnan(probs[_attributes_cnt])):
                probs[_attributes_cnt] = 0.5
            assert 0 <= probs[_attributes_cnt] and probs[_attributes_cnt] <= 1, "Error in calculating probabilities " + str(
                probs[_attributes_cnt]) + " : " + str(probs[_attributes_cnt])

            _attributes_cnt += 1

        else:

            bins, _binner_dict, _ = column_info[idx]

            # print()
            # print(data.columns[idx])
            # mu, sigma = norm.fit(data[data.columns[idx]])
            # print("mu ", mu, "sigma ", sigma)
            # print("max:", data[data.columns[idx]].max() )
            # print("min", data[data.columns[idx]].min() )
            # print()

            for i in range(bins):

                if(not compute_for_all):
                    if(_attributes_cnt not in selected_columns):
                        _attributes_cnt += 1
                        continue

                if(i == bins - 1):
                    # for the last bin, include the max boundary
                    probs[_attributes_cnt] = round(((_binner_dict[i] <= data[data.columns[idx]]) & (
                        data[data.columns[idx]] <= _binner_dict[i+1])).mean(), 3)
                else:
                    probs[_attributes_cnt] = round(((_binner_dict[i] <= data[data.columns[idx]]) & (
                        data[data.columns[idx]] < _binner_dict[i+1])).mean(), 3)

                # probs[_attributes_cnt] = round(((_binner_dict[i] <= data[data.columns[idx]]) & (data[data.columns[idx]] < _binner_dict[i+1])).mean(),3)
                if(probs[_attributes_cnt] == 0):
                    probs[_attributes_cnt] = 0.001
                elif(probs[_attributes_cnt] == 1):
                    probs[_attributes_cnt] = 0.999
                elif(math.isnan(probs[_attributes_cnt])):
                    probs[_attributes_cnt] = 0.5
                _attributes_cnt += 1

    return probs


def n_weighted_moment(values, weights, n):

    assert n > 0 & (values.shape == weights.shape)
    w_avg = np.average(values, weights=weights)
    w_var = np.sum(weights * (values - w_avg)**2)/np.sum(weights)

    if n == 1:
        return w_avg
    elif n == 2:
        return w_var
    else:
        w_std = np.sqrt(w_var)
        return np.sum(weights * ((values - w_avg)/w_std)**n)/np.sum(weights)


def draw_dependency(metric):
    if(metric._graph_edges is None):
        print("No Bayesian network is provided or learned")
        return

    G = nx.DiGraph()
    edges = []
    edges = metric._graph_edges
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=1)
    for node in sorted(list(G.nodes)):
        label = None
        if(type(metric._variable_attribute_map[node]) == tuple):
            if(metric._model_name in ["CNF"]):
                label = metric._variable_attribute_map[node]
            elif(metric._model_name == "decision-tree"):
                feature, threshold = metric._variable_attribute_map[node]
                label = feature + " <= " + str(round(threshold, 2))
            elif(metric._model_name == "linear-model"):
                feature, comparator_1, threshold_1, comparator_2, threshold_2 = metric._variable_attribute_map[
                    node]
                # if(comparator_2 == "=="):
                #     label = feature + " $\geq 40$"
                if(comparator_1 is None):
                    label = feature + " " + comparator_2 + \
                        " " + str(round(threshold_1, 2))
                else:

                    label = str(round(threshold_1, 2)) + r" $\leq$ " + feature + \
                        " " + comparator_2 + " " + str(round(threshold_2, 2))

            else:
                raise ValueError
        else:
            label = metric._variable_attribute_map[node]

        nx.draw_networkx_nodes(G, pos, node_color="white",  nodelist=[
                               node], label=str(node) + ": " + label)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos=pos)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    return plt


def draw_nx(G):
    pos = nx.spring_layout(G, k=1)
    dic = {}
    for idx, node in enumerate(sorted(list(G.nodes))):
        dic[node] = idx
        nx.draw_networkx_nodes(G, pos, node_color="white",  nodelist=[
                               node], label=str(idx) + ": " + node)
    nx.draw_networkx_labels(G, pos, labels=dic)
    nx.draw_networkx_edges(G, pos=pos)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    return plt


def get_statistics_for_linear_classifier_wrap(data, weights, known_sensitive_attributes, discretizer="equalWidth", num_bins=16, verbose=False):
    """
        Data contains either binary attributes or real-valued attributes. 
        If it is a binary attribute, the probability calculation is trivial. 
        For real-valued attributes, we need to discretize it. 
        let w be the coefficient and x be a real-valued feature. 
        now x has to be discretized to, say three bins z1, z2 and z3. 
        We apply the following transformation
        wx = w * sum_i=1^3 (t_i * z_i) where t_i is the mean value of the bin z_i
    """

    calculated_weights = []
    probs = {}
    sensitive_attributes = [[] for _ in known_sensitive_attributes]
    attributes = []  # contains non-sensitive attributes
    column_info = {}
    attribute_variable_map = {}
    variable_attribute_map = {}
    converted_data = pd.DataFrame()

    _attributes_cnt = 1
    _discretized_attributes_group = []
    for idx in range(len(data.columns)):

        _uniques = data[data.columns[idx]].unique()
        _num_uniques = len(_uniques)
        assert _num_uniques > 0
        _max_value = _uniques.max()
        # assert _num_uniques != 1, "Error:  " + data.columns[idx] + " feature contains single feature-value"
        if(_num_uniques <= 2):

            # check for sensitive attributes
            _is_sensitive_attribute = False
            for group_idx in range(len(known_sensitive_attributes)):
                """
                    We check for sensitive attribute based ontwo conditions.
                        1. The column name matches with a known_sensitive_attribute
                        2. The attribute is one-hot encoded, in which case known_sensitive_attribute has new suffix with
                           following pattern: <column>_<id>. For example, race = {0, 1, 2} has one-hot encoded
                           columns as race_0, race_1, race_2 etc.
                """
                if(known_sensitive_attributes[group_idx].startswith(data.columns[idx].split("_")[0])):
                    sensitive_attributes[group_idx].append(_attributes_cnt)
                    _is_sensitive_attribute = True
                    variable_attribute_map[_attributes_cnt] = (
                        data.columns[idx], None, _max_value, "==", None)
                    attribute_variable_map[(
                        data.columns[idx], None, _max_value, "==", None)] = _attributes_cnt
                    break

                elif(known_sensitive_attributes[group_idx].startswith(data.columns[idx])):
                    sensitive_attributes[group_idx].append(_attributes_cnt)
                    _is_sensitive_attribute = True
                    variable_attribute_map[_attributes_cnt] = (
                        data.columns[idx], None, _max_value, "==", None)
                    attribute_variable_map[(
                        data.columns[idx], None, _max_value, "==", None)] = _attributes_cnt
                    break

            # otherwise non-sensitive attributes
            if(not _is_sensitive_attribute):
                attributes.append(_attributes_cnt)
                variable_attribute_map[_attributes_cnt] = (
                    data.columns[idx], None, _max_value, "==", None)
                attribute_variable_map[(
                    data.columns[idx], None, _max_value, "==", None)] = _attributes_cnt

            # For binary features, weights remain same as provided
            calculated_weights.append(weights[idx])
            probs[_attributes_cnt] = round(
                (data[data.columns[idx]] == _max_value).mean(), 3)
            _attributes_cnt += 1
            column_info[idx] = ("Bin", data.columns[idx], _max_value)
            converted_data[data.columns[idx]] = data[data.columns[idx]]

        else:
            bins = min(num_bins, _num_uniques)
            if(discretizer == "equalWidth"):

                # set up the discretisation transformer
                disc = dsc.EqualWidthDiscretiser(
                    bins=bins, variables=[data.columns[idx]])

                # fit the transformer
                disc.fit(data)
            elif(discretizer == "equalFreq"):
                disc = dsc.EqualFrequencyDiscretiser(
                    q=bins, variables=[data.columns[idx]])
                # fit the transformer
                disc.fit(data)

            else:
                raise ValueError(discretizer + " not defined")

            # contains binning information
            _binner_dict = disc.binner_dict_[data.columns[idx]]

            # print()
            # print(data.columns[idx])
            # mu, sigma = norm.fit(data[data.columns[idx]])
            # print("mu ", mu, "sigma ", sigma)
            # print()

            assert _binner_dict[0] == float(
                "-inf"), "Error: discretization error"
            assert _binner_dict[-1] == float(
                "inf"), "Error: discretization error"
            assert len(_binner_dict) == bins + \
                1, "Error: wrong number of bins in discretization"

            # replace min and max values by column min and column max
            _binner_dict[0] = data[data.columns[idx]].min()
            _binner_dict[-1] = data[data.columns[idx]].max()
            _discretized_attributes_group.append([])

            """
            histogram_mean = []
            histogram_prob = []
            histogram_mean_data_dependent = []
            """

            for i in range(bins):
                attributes.append(_attributes_cnt)
                _threshold = float(
                    (_binner_dict[i+1] + _binner_dict[i])/2.0)  # mean of bin
                # histogram_mean.append(_threshold)
                if(i == bins - 1):
                    # for the last bin, include the max boundary
                    variable_attribute_map[_attributes_cnt] = (
                        data.columns[idx], ">=", _binner_dict[i], "<=", _binner_dict[i+1])
                    attribute_variable_map[(
                        data.columns[idx], ">=", _binner_dict[i], "<=", _binner_dict[i+1])] = _attributes_cnt
                    probs[_attributes_cnt] = round(((_binner_dict[i] <= data[data.columns[idx]]) & (
                        data[data.columns[idx]] <= _binner_dict[i+1])).mean(), 3)
                    # histogram_mean_data_dependent.append(data[data.columns[idx]][(_binner_dict[i] <= data[data.columns[idx]]) & (data[data.columns[idx]] <= _binner_dict[i+1])].mean())
                    converted_data[str(_binner_dict[i]) + " <= " + data.columns[idx] + " <= " + str(_binner_dict[i+1])] = (
                        (data[data.columns[idx]] >= _binner_dict[i]) & (data[data.columns[idx]] <= _binner_dict[i + 1])).astype(int)
                else:
                    variable_attribute_map[_attributes_cnt] = (
                        data.columns[idx], ">=", _binner_dict[i], "<", _binner_dict[i+1])
                    attribute_variable_map[(
                        data.columns[idx], ">=", _binner_dict[i], "<", _binner_dict[i+1])] = _attributes_cnt
                    probs[_attributes_cnt] = round(((_binner_dict[i] <= data[data.columns[idx]]) & (
                        data[data.columns[idx]] < _binner_dict[i+1])).mean(), 3)
                    # histogram_mean_data_dependent.append(data[data.columns[idx]][(_binner_dict[i] <= data[data.columns[idx]]) & (data[data.columns[idx]] < _binner_dict[i+1])].mean())
                    converted_data[str(_binner_dict[i]) + " <= " + data.columns[idx] + " < " + str(_binner_dict[i+1])] = (
                        (data[data.columns[idx]] >= _binner_dict[i]) & (data[data.columns[idx]] < _binner_dict[i + 1])).astype(int)
                # histogram_prob.append(probs[_attributes_cnt])

                """
                # sample mean within bin-->
                _threshold = histogram_mean_data_dependent[-1]
                if(math.isnan(_threshold)):
                    # if histogram_mean_data_dependent is nan, which means there is no point within that bin. 
                    # In that case, we use histogram mean.
                    _threshold = histogram_mean[-1]
                    pass
                """

                # multiplied by the mean
                calculated_weights.append(weights[idx] * _threshold)
                _discretized_attributes_group[-1].append(_attributes_cnt)
                _attributes_cnt += 1

            column_info[idx] = (bins, _binner_dict, data.columns[idx])

            # assertion
            _sum = 0
            for _attribute in _discretized_attributes_group[-1]:
                _sum += probs[_attribute]
            if(len(_discretized_attributes_group[-1]) >= 2):
                assert abs(_sum - 1) <= 0.1, "Error in calculating probabilities of discretized variables. Computed value is " + \
                    str(abs(_sum - 1)) + " in group " + \
                    str(_discretized_attributes_group[-1])

            # print()
            # print(data.columns[idx])
            # # print(data[data.columns[idx]].values)
            # x = data[data.columns[idx]].values
            # params = {'bandwidth': np.logspace(-1, 1, 20)}
            # grid = GridSearchCV(KernelDensity(), params, cv=5)
            # grid.fit(x.reshape(-1,1))
            # kde = grid.best_estimator_
            # # vals = np.linspace(x.min(), x.max(), 100*bins)
            # vals = np.array(_binner_dict)
            # density = np.exp(kde.score_samples(vals.reshape(-1,1)))

            # kde_sm = sm.nonparametric.KDEUnivariate(np.array(x))
            # kde_sm.fit()

            # closest_indices = []
            # for q in range(len(vals)):
            #     closest_indices.append(kde_sm.support.tolist().index(min(kde_sm.support, key=lambda x:abs(x-vals[q]))))

            # cdf = kde_sm.cdf
            # histogram_prob_kde = []
            # for q in range(len(closest_indices)-1):
            #     if(q == 0):
            #         histogram_prob_kde.append(cdf[closest_indices[q+1]])
            #     elif(q == len(closest_indices)-2):
            #         histogram_prob_kde.append(1 - cdf[closest_indices[q]])
            #     else:
            #         histogram_prob_kde.append(cdf[closest_indices[q+1]] - cdf[closest_indices[q]])

            # # print(vals)
            # # print(density)
            # print("histogram_prob", histogram_prob)
            # print("histogram_prob_kde", histogram_prob_kde)
            # print("histogram_mean", histogram_mean)
            # print("histogram_mean_data", histogram_mean_data_dependent)
            # for i in range(1, 5):
            #     print()
            #     print(f'Sample         : Moment {i} value is {n_weighted_moment(np.array(x), np.array([1 for _ in x]), i)}')
            #     print(f'Mean           : Moment {i} value is {n_weighted_moment(np.array(histogram_mean), np.array(histogram_prob), i)}')
            #     print(f'Filtered sample: Moment {i} value is {n_weighted_moment(np.array(histogram_mean_data_dependent), np.array(histogram_prob), i)}')
            #     print(f'Filtered (kde) : Moment {i} value is {n_weighted_moment(np.array(histogram_mean_data_dependent), np.array(histogram_prob_kde), i)}')

            #     print()
            # # print((np.array(histogram_prob) * np.array(histogram_mean)).sum())
            # # print(np.array(xx).mean())

    if(verbose > 1):
        print(converted_data)
        print("\nvariable to attribute map ->")
        print(variable_attribute_map)
        print("\nattribute to variable map ->")
        print(attribute_variable_map)
        print("\n\n\n")

    return attributes, sensitive_attributes, probs, np.array(calculated_weights), variable_attribute_map, column_info, converted_data, _discretized_attributes_group


def get_sensitive_attibutes(known_sensitive_features, features):
    """ 
    Return sensitive attributes in appropriate format
    """

    # Extract new names of sensitive attributes
    # it is a map because each entry contains all one-hot encoded variables
    _sensitive_attributes = {}
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
