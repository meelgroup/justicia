
import pandas as pd 
import numpy as np
import os
import networkx as nx
from numpy import genfromtxt
import itertools
from operator import add
from pgmpy.estimators import ParameterEstimator
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import igraph as ig
import warnings



# Generate CNF from dependency graph,
# Introduce auxiliary variables whenever necessay,



class Dependency():
    def __init__(self, edges, edge_weights, probs, variable_header = None):
        # vertices : participating features
        # edges: set of directed edge (a,b) denoting dependency of vertix a on b
        self.edges = edges # list of tuples
        self.edge_weights = edge_weights
        self.probs = probs # dictionary object, key is a variable and value is the satisfying probability of that variable
        self.variable_header = variable_header
        self.attributes = [] # these will be randomized quantified
        self.auxiliary_variables = [] # these will be existential quantified
        self.CNF = [] # 2d list
        self.introduced_variables_map = {} # keep track of variable to introduced variable
        self._prepare_CNF()

    def __str__(self):
        s = ""
        s += "\nDependency model: "
        s += "CNF\n" + str(self.CNF) + "\n\n"
        s += "Attributes (random) quantified:\n" + str(self.attributes) + "\n\n"
        s += "Auxiliary (Existential quantified):\n" + str(self.auxiliary_variables) + "\n\n"
        s += "Variable header: " + str(self.variable_header)   + "\n\n"
        return s


    def _prepare_CNF(self):

        for idx in range(len(self.edges)):
            (a,b) = self.edges[idx] 
            
            
            # introduce two constraints
                #  a & theta_{b|a} -> b
                #  a & ~theta_{b|a} -> ~b
                
            # when a is True
            self.variable_header += 1 # introduce new variable
            self.attributes.append(self.variable_header)
            if(isinstance(a, tuple)):
                # An assignment of parent variables
                self.CNF.append([-1 * var for var in a] + [ -1 * self.variable_header, b])
                self.CNF.append([-1 * var for var in a] + [self.variable_header, -1 * b])
            else:
                self.CNF.append([-1 * a, -1 * self.variable_header, b])
                self.CNF.append([-1 * a, self.variable_header, -1 * b])
            self.probs[self.variable_header] = self.edge_weights[(a,b)]  # assign probabilities to the newly introduced variable

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
    for a,b in edges:
        # direction a -> b
        if(b not in dependency_dic):
            dependency_dic[b] = [a]
        else:
            dependency_dic[b].append(a)

    result = []

    for key in dependency_dic:
        for x in itertools.product([1, -1], repeat=len(dependency_dic[key])):
            parent = tuple(a*b for a,b in list(zip(dependency_dic[key], x)))
            result.append((parent, key))

    return result

def default_dependency(sensitive_attributes, non_sensitive_attributes):
    
    edges = [(3,1),(1,4),(2,4),(4,2)]

    return do_combinations(edges), edges


def compound_group_correlation(sensitive_attributes, non_sensitive_attributes):

    """
    Is used for encoding sensitive group correlation in learning encoding.
    """
    
    _sensitive_attributes = []
    for _group in sensitive_attributes:
        if(len(_group)==1):
            _group = [_group[0], -1*_group[0]]
        _sensitive_attributes.append(_group)    

    combinations = list(itertools.product(*_sensitive_attributes))
    result = []
    for attribute in non_sensitive_attributes:
        for combination in combinations:
            result.append((combination, attribute))

    return result
    
def call_pgmpy(metrics_object):
    # data is a pandas object
    print(metrics_object.data)
    print(metrics_object._variable_attribute_map)


def _call_notears(data, sensitive_attributes, regularizer = 0.01, verbose=True, filename="temp"):
    """returns a Dependency structure: list of dependencies of the form a -> b where a is a tuple of parents"""
    
    # flatten to 1d list
    flatten_sensitive_attributes = [abs(_var) for _group in sensitive_attributes for _var in _group]

    filename = "./" + filename
    
    data.to_csv(filename + "X.csv", index=False, header=None)
    cmd = "timeout 100 notears_linear --lambda1 " + str(regularizer) + " --W_path " + filename + "W_est.csv " + filename +  "X.csv"
    os.system(cmd)

    if(not os.path.isfile(filename + 'W_est.csv')):
        return [], [], False


    
    # construct weighted graphs
    adjacency_matrix = genfromtxt(filename + 'W_est.csv', delimiter=',')

    # remove temp files
    os.system("rm " + filename + "W_est.csv " + filename +  "X.csv")

    
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
    for a,b in iG.get_edgelist():
        if(b + 1 in flatten_sensitive_attributes):
            if(verbose):
                print((a,b), " found edges incident on sensitive variable")
            iG.delete_edges([(a,b)])
            iG.add_edge(b,a)

    """
        Update: DAG is not a requirement in Justicia, hence all changes are invoked
    """    
    if(not iG.is_dag()):
        pass
        # if(verbose):
        #     print("Inverting edge direction does not produce a DAG")
        # iG = temp.copy()
    else:
        if(verbose):
            print("Inverting edge direction still produces a DAG")
        
    
    if(verbose > 1):
        print("\nAfter modification")
        print(iG)
        
    
        



    edges = []
    for a,b in iG.get_edgelist():
        # increment for consistency with Justicia's declared variables
        a += 1
        b += 1
        edges.append((a,b))

        
    return do_combinations(edges), edges, True
    


def Bayesian_estimate(data, dependency_structure, graph_edges):
    data.columns = [i + 1 for i in range(data.shape[1])]
    print(data)
    model = BayesianModel(graph_edges)
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu") # default equivalent_sample_size=5

    for column in data.columns:
        print(column, data[column].unique())
    probs = {}

    for parent, child in dependency_structure:
        cpd = model.get_cpds(node=child)
        print()
        print(cpd)
        print(cpd.variable_card)
        index = [cpd.variables.index(var) - 1 if var > 0 else cpd.variables.index(-1 * var) - 1 for var in parent]
        ordered_parent = [x for _,x in sorted(zip(index,parent))]
        print(cpd.values)
        if(cpd.variable_card == 1):
            if(data[child].unique()[0] == 0):
                value = 1 - cpd.values[0]
            else:
                value = cpd.values[0]
        else:
            assert cpd.variable_card == 2
            value = cpd.values[1]
        print(value)
        for var in ordered_parent:
            value = value[0] if var < 0 else value[1]
        probs[(parent, child)] =  value
        print((parent, child), probs[(parent, child)])
    
    return probs



def refine_dependency_constraints(sensitive_attributes, edges):
    """
    If there is an incidenet edge to a sensitive variables, it should be removed. Because sensitive 
    variables are already existentially quantified, so no randomness is involved
    """
    # flatten to 1d list
    all_sensitive_attributes = [abs(_var) for _group in sensitive_attributes for _var in _group]

    result = []
    for a,b in edges:
        if(b in all_sensitive_attributes or -1 * b in all_sensitive_attributes):
            # print((a,b), "is removed")
            continue
        result.append((a,b))
    
    return result


