
import pandas as pd
import numpy as np
import os
import networkx as nx
from numpy import genfromtxt
import itertools
from operator import add
from pgmpy.estimators import ParameterEstimator
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, BicScore
import igraph as ig
import warnings
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from time import time


# Generate CNF from dependency graph,
# Introduce auxiliary variables whenever necessay,


class Dependency():
    def __init__(self, edges, edge_weights, probs, variable_header=None):
        # vertices : participating features
        # edges: set of directed edge (a,b) denoting dependency of vertix a on b
        self.edges = edges  # list of tuples
        self.edge_weights = edge_weights
        # dictionary object, key is a variable and value is the satisfying probability of that variable
        self.probs = probs
        self.variable_header = variable_header
        self.attributes = []  # these will be randomized quantified
        self.auxiliary_variables = []  # these will be existential quantified
        self.CNF = []  # 2d list
        # keep track of variable to introduced variable
        self.introduced_variables_map = {}
        self._prepare_CNF()

    def __str__(self):
        s = ""
        s += "\nDependency model: "
        s += "CNF\n" + str(self.CNF) + "\n\n"
        s += "Attributes (random) quantified:\n" + \
            str(self.attributes) + "\n\n"
        s += "Auxiliary (Existential quantified):\n" + \
            str(self.auxiliary_variables) + "\n\n"
        s += "Variable header: " + str(self.variable_header) + "\n\n"
        return s

    def _prepare_CNF(self):

        for idx in range(len(self.edges)):
            (a, b) = self.edges[idx]

            # introduce two constraints
            #  a & theta_{b|a} -> b
            #  a & ~theta_{b|a} -> ~b

            # when a is True
            self.variable_header += 1  # introduce new variable
            self.attributes.append(self.variable_header)
            if(isinstance(a, tuple)):
                # An assignment of parent variables
                self.CNF.append([-1 * var for var in a] +
                                [-1 * self.variable_header, b])
                self.CNF.append([-1 * var for var in a] +
                                [self.variable_header, -1 * b])
            else:
                self.CNF.append([-1 * a, -1 * self.variable_header, b])
                self.CNF.append([-1 * a, self.variable_header, -1 * b])
            # assign probabilities to the newly introduced variable
            self.probs[self.variable_header] = self.edge_weights[(a, b)]

            # For tracking
            if(b not in self.introduced_variables_map):
                self.introduced_variables_map[b] = [self.variable_header]
            else:
                self.introduced_variables_map[b].append(self.variable_header)

            # variable b should get existential quantified and appear in the inner-most level
            if(b not in self.auxiliary_variables):
                self.auxiliary_variables.append(b)
            if(b in self.attributes):
                self.attributes.remove(b)

            # remove variable that will be existentially quantified
            if(b in self.probs):
                self.probs.pop(b)


def do_combinations(edges):
    """
    Input is a list of directed edge
    Output is a form that is digestable by Justicia, generally applied combinations on parents
    """
    dependency_dic = {}
    for a, b in edges:
        # direction a -> b
        if(b not in dependency_dic):
            dependency_dic[b] = [a]
        else:
            dependency_dic[b].append(a)
    result = []

    for key in dependency_dic:
        for x in itertools.product([1, -1], repeat=len(dependency_dic[key])):
            # print(x)
            parent = tuple(a*b for a, b in list(zip(dependency_dic[key], x)))
            result.append((parent, key))
        # print(result)
    return result


def default_dependency(sensitive_attributes, non_sensitive_attributes):

    edges = [(4, 5), (5, 7), (7, 9)]

    return do_combinations(edges), edges


def compound_group_correlation(sensitive_attributes, non_sensitive_attributes, disretized_group_attributes):
    """
    Is used for encoding sensitive group correlation in learning encoding.
    """

    result = []
    for attribute in non_sensitive_attributes:
        for group in sensitive_attributes:
            for var in group:
                result.append((var, attribute))

    # for discretized_group in disretized_group_attributes:
    #     len_ = len(discretized_group)
    #     for i in range(len_):
    #         for j in range(i+1, len_):
    #             result.append((discretized_group[i], discretized_group[j]))

    return do_combinations(result), result


def call_pgmpy(metrics_object):
    # data is a pandas object
    print(metrics_object.data)
    print(metrics_object._variable_attribute_map)


def _call_pgmpy(data, sensitive_attributes, verbose=True, threshold=0.33):
    if(verbose):
        print("\n---------------------------\nStarting DAG learning")
    data.columns = [str(var) for var in data.columns]
    result_edges = []
    flatten_sensitive_attributes = [
        str(abs(_var)) for _group in sensitive_attributes for _var in _group]
    timeout = 200
    start_time = time()
    black_list = [(all_var, var)
                  for var in flatten_sensitive_attributes for all_var in data.columns]

    # fixed_edges = [('25', '24'), ('20', '24'), ('30', '24')]
    fixed_edges = []

    for max_iter in [1, 5, 10, 50, 100, 200, 500, 1000, 5000, 10000]:
        if(time() - start_time > timeout):
            break

        est = HillClimbSearch(data=data)
        estimated_model = est.estimate(max_iter=max_iter, max_indegree=5,
                                       fixed_edges=fixed_edges, black_list=black_list, show_progress=False)
        network_nodes = []
        for a, b in list(estimated_model.edges()):
            network_nodes.append(a)
            network_nodes.append(b)
            if(b in flatten_sensitive_attributes):
                raise RuntimeError("Black_listing is not working")

        network_nodes = set(network_nodes)
        if(verbose > 1):
            print("\nAfter modification")
            print(estimated_model)

        edges = estimated_model.edges()

        if(tuple(edges) == tuple(result_edges)):
            if(verbose):
                print("Producing same DAG")
            break

        if(verbose):
            print("Maximum iteration", max_iter, " edges:",
                  len(edges), "nodes", len(network_nodes))
            print(edges)

        # If 25% nodes are covered, we return results
        if((len(set(network_nodes)) >= len(data.columns) * threshold) and threshold != 1):
            break

        result_edges = edges

    result_edges = [(int(a), int(b)) for a, b in result_edges]
    # print(result_edges)

    return do_combinations(result_edges), result_edges, True

    # return estimated_model


def _call_notears(data, sensitive_attributes, verbose=True, filename="temp"):
    """returns a Dependency structure: list of dependencies of the form a -> b where a is a tuple of parents"""

    flatten_sensitive_attributes = [
        abs(_var) for _group in sensitive_attributes for _var in _group]
    filename = "./" + filename
    data.to_csv(filename + "X.csv", index=False, header=None)

    result_edges = []
    for regularizer in [10, 1, 0.1, 0.01, 0.001, 0.0001]:

        # flatten to 1d list
        cmd = "timeout 200 notears_linear --lambda1 " + \
            str(regularizer) + " --W_path " + filename + \
            "W_est.csv " + filename + "X.csv"
        os.system(cmd)
        if(not os.path.isfile(filename + 'W_est.csv')):
            return [], [], False
        # construct weighted graphs
        adjacency_matrix = genfromtxt(filename + 'W_est.csv', delimiter=',')
        # graph
        iG = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist())
        """
        Check if there is an incident edge on a sensitive variable.
        If there is, then reverse the edge and check if the graph is still a DAG, or revert changes.
        """
        temp = iG.copy()
        if(verbose > 1):
            print("\nLearned graph")
            print(iG)
        for a, b in iG.get_edgelist():
            if(b + 1 in flatten_sensitive_attributes):
                iG.delete_edges([(a, b)])
                if(a + 1 in flatten_sensitive_attributes):
                    continue
                else:
                    if(verbose):
                        print((a, b), " found edges incident on sensitive variable")

                iG.add_edge(b, a)

        if(verbose > 1):
            print("\nAfter modification")
            print(iG)

        os.system("rm " + filename + "W_est.csv ")

        edges = []
        for a, b in iG.get_edgelist():
            # increment for consistency with Justicia's declared variables
            a += 1
            b += 1
            edges.append((a, b))

        if(len(edges) > len(result_edges)):
            result_edges = edges

        if(len(edges) > 0):
            result_edges = edges
            if(verbose):
                print("optimal regularizer", regularizer,
                      " generates edges of length", len(edges))
            break

    # remove temp files
    os.system("rm " + filename + "X.csv")

    return do_combinations(result_edges), result_edges, True


def Bayesian_estimate(data, dependency_structure, graph_edges):
    # data.columns = [i + 1 for i in range(data.shape[1])]
    model = BayesianModel(graph_edges)
    model.fit(data, state_names={var: [0, 1] for var in data.columns})
    probs = {}

    # print(graph_edges)
    for parent, child in dependency_structure:
        cpd = model.get_cpds(node=child)

        # print("\n")
        # print(cpd)

        index = [cpd.variables.index(
            var) - 1 if var > 0 else cpd.variables.index(-1 * var) - 1 for var in parent]
        ordered_parent = [x for _, x in sorted(zip(index, parent))]
        assert cpd.variable_card == 2
        value = cpd.values[1]
        for var in ordered_parent:
            value = value[0] if var < 0 else value[1]
        if(value == 0):
            value = 0.001
        elif(value == 1):
            value = 0.999
        probs[(parent, child)] = round(value, 3)

        # print((parent, child), probs[(parent, child)])

    # quit()
    return probs


def refine_dependency_constraints(sensitive_attributes, edges):
    """
    If there is an incidenet edge to a sensitive variables, it should be removed. Because sensitive 
    variables are already existentially quantified, so no randomness is involved
    """
    # flatten to 1d list
    all_sensitive_attributes = [
        abs(_var) for _group in sensitive_attributes for _var in _group]

    result = []
    for a, b in edges:
        if(b in all_sensitive_attributes or -1 * b in all_sensitive_attributes):
            # print((a,b), "is removed")
            continue
        result.append((a, b))

    return result
