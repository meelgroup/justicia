from justicia import utils
from justicia.linear_classifier_wrap import linear_classifier_wrap_Wrapper, predict_lin
import itertools
from justicia.ssat_wrap import Fairness_verifier
from justicia.decision_tree_wrap import dtWrapper
import numpy as np
import pandas as pd
from justicia import dependency_utils
from itertools import chain
from time import time
from sklearn.metrics import accuracy_score, f1_score


from justicia.subsetsum import SubSetSumCount
# from justicia.subsetsum_backup import SubSetSumCount


# supporting classifiers
from pyrulelearn.imli import imli
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from justicia.poison_attack_wrap import Poison_Model


class Metric():

    def __init__(self, model, data, sensitive_attributes, dependency_graph=None, mediator_attributes=[], major_group={}, encoding="Enum", filename="sample.sdimacs", verbose=True, feature_threshold=0, timeout=900, discretization_bins=10, dag_threshold=0.001):
        self.encoding = encoding
        self.model = model
        self._model_name = self._retrieve_model_name()
        self._data = data
        self.given_sensitive_attributes = sensitive_attributes
        if(dependency_graph is None):
            self._given_dependency_graph = None
        else:
            self._given_dependency_graph = dependency_graph.copy()
        self._filename = filename
        self._verbose = verbose
        self._meta_sensitive_groups = None
        self._correlation_constraints = []
        self._timeout = timeout
        self.sensitive_group_statistics = []
        self._time_taken_notears = 0
        self._feature_threshold = feature_threshold
        self._graph_edges = None
        self._num_variables = []
        self._num_clauses = []
        self._dag_threshold = dag_threshold
        self._discretization_bins = discretization_bins
        """
        Path-specific causal fairness
        """
        # These attributes have constant probabilities across different compound groups.
        self.given_mediator_attributes = mediator_attributes
        # each item is a map from feature to binary feature-value
        self.given_major_group = major_group

        """
        Auto select encoding based on classifiers
        """
        self._encoding_selection()

        # preprocessing
        start_time = time()
        self.encoding = utils.select_encoding(
            self._model_name, self.encoding, self._verbose)
        self._get_required_params()
        self.time_taken = time() - start_time

    def _encoding_selection(self):
        if(self.encoding == "best"):
            if(self._model_name == "linear-model"):
                self.encoding = "Learn"
            elif(self._model_name in ['CNF', 'decision-tree']):
                self.encoding = "Enum"
            else:
                raise ValueError((self._model_name))
        elif(self.encoding == "best-correlated"):
            if(self._model_name == "linear-model"):
                self.encoding = "Learn-dependency"
            elif(self._model_name in ['CNF', 'decision-tree']):
                self.encoding = "Enum-dependency"
            else:
                raise ValueError((self._model_name))
        else:
            pass

    def __repr__(self):
        if(self._model_name == "CNF"):
            return "\nJusticia\n - model: pyrulelearn\n" + '\n'.join(" - %s: %s" % (item, value) for (item, value) in vars(self).items() if (not item.startswith("_") and item != "model"))

        return "\nJusticia\n" + '\n'.join(" - %s: %s" % (item, value) for (item, value) in vars(self).items() if not item.startswith("_"))

    def compute(self):
        start_time = time()
        self._compute()
        self.time_taken += time() - start_time

        return self

    def compute_eqo(self, y):
        """
        In order to compute equalized odds, we instantiate two Metric computing, one for y = 1 and other other for y = 0
        """
        y = np.array(y)
        assert np.array_equal(y, y.astype(bool))

        # manipulate variables
        full_data = self._data.copy()
        self.time_taken = 0
        _time_taken_notears = 0

        result_DI = []
        result_SPD = []
        result_sensitive_group_statistics = []
        result_most_favored_group = []
        result_least_favored_group = []

        start_time = time()

        # when y = 1
        self._data = full_data[y == 1]
        self._get_required_params()
        self.compute()
        result_DI.append(self.disparate_impact_ratio)
        result_SPD.append(self.statistical_parity_difference)
        result_sensitive_group_statistics.append(
            self.sensitive_group_statistics)
        result_most_favored_group.append(self.most_favored_group)
        result_least_favored_group.append(self.least_favored_group)
        _time_taken_notears += self._time_taken_notears

        # when y = 0
        self._data = full_data[y == 0]
        self.sensitive_group_statistics = []
        self._get_required_params()
        self.compute()
        result_DI.append(self.disparate_impact_ratio)
        result_SPD.append(self.statistical_parity_difference)
        result_sensitive_group_statistics.append(
            self.sensitive_group_statistics)
        result_most_favored_group.append(self.most_favored_group)
        result_least_favored_group.append(self.least_favored_group)
        _time_taken_notears += self._time_taken_notears

        # retrieve variables
        self._data = full_data
        self.disparate_impact_ratio = result_DI
        self.statistical_parity_difference = result_SPD
        self.sensitive_group_statistics = result_sensitive_group_statistics
        self.most_favored_group = result_most_favored_group
        self.least_favored_group = result_least_favored_group
        self._time_taken_notears = _time_taken_notears
        self.time_taken = time() - start_time

        return self

    def _get_required_params(self):

        # Some variables which are required later
        self._num_attributes_neg = None
        self._discretized_group_attributes = []
        if(self._model_name == "linear-model"):

            self._original_weights = self.model.coef_[0].copy()

            # choose best discretization param

            # when data contains continuous features, call specilized statistics
            best_accuracy = -1
            best_config = {
                "attributes": None,
                "sensitive_attributes": None,
                "probs": None,
                "weights": None,
                "bias": None,
                "variable_attribute_map": None,
                "column_info": None,
                "converted_df": None,
                "discretized_group_attributes": None
            }

            for num_bins in range(1, self._discretization_bins):
                self._attributes, self._sensitive_attributes, self._probs, self._given_weights, self._variable_attribute_map, self._column_info, converted_df, self._discretized_group_attributes = utils.get_statistics_for_linear_classifier_wrap(
                    self._data.copy(), self._original_weights, self.given_sensitive_attributes, num_bins=num_bins, verbose=self._verbose)

                # consider important features only
                if(self._feature_threshold is not None):
                    feature_importance = abs(self._given_weights)
                    feature_importance = 100.0 * \
                        (feature_importance / feature_importance.max())
                    threshold = np.percentile(
                        np.array(feature_importance), float(self._feature_threshold))
                    self._weights = []
                    for i in range(len(self._given_weights)):
                        if(feature_importance[i] < threshold):
                            self._weights.append(0)
                        else:
                            self._weights.append(self._given_weights[i])
                else:
                    self._weights = self._given_weights

                self._bias = self.model.intercept_[0].copy()

                measured_accuracy = f1_score(self.model.predict(
                    self._data), predict_lin(converted_df, self._weights, self._bias))
                if(measured_accuracy > best_accuracy):
                    best_accuracy = measured_accuracy
                    best_config["attributes"] = self._attributes
                    best_config["sensitive_attributes"] = self._sensitive_attributes
                    best_config["probs"] = self._probs
                    best_config["weights"] = self._weights
                    best_config["bias"] = self._bias
                    best_config["variable_attribute_map"] = self._variable_attribute_map
                    best_config["column_info"] = self._column_info
                    best_config["converted_df"] = converted_df
                    best_config["discretized_group_attributes"] = self._discretized_group_attributes

            for key, val in best_config.items():
                assert val is not None

            self._attributes = best_config["attributes"]
            self._sensitive_attributes = best_config["sensitive_attributes"]
            self._probs = best_config["probs"]
            self._weights = best_config["weights"]
            self._bias = best_config["bias"]
            self._variable_attribute_map = best_config["variable_attribute_map"]
            self._column_info = best_config["column_info"]
            converted_df = best_config["converted_df"]
            self._discretized_group_attributes = best_config["discretized_group_attributes"]

            # get classifier
            lr = linear_classifier_wrap_Wrapper(weights=self._weights, attributes=self._attributes, sensitive_attributes=self._sensitive_attributes, bias=self._bias,
                                                converted_data=converted_df, original_model_prediction=self.model.predict(self._data), convert_to_cnf="Enum" in self.encoding, verbose=self._verbose)
            self._num_attributes = lr.num_attributes
            self._attributes = lr.attributes
            self._classifier = lr.classifier
            self._sensitive_attributes = lr.sensitive_attributes
            self._auxiliary_variables = lr.auxiliary_variables
            self._linear_model = lr

            if(self._verbose):
                # print("Classifier:", self._classifier)
                print("Translation accuracy", accuracy_score(self._linear_model.predict(
                    converted_df, self._linear_model.weights, self._linear_model.bias), self.model.predict(self._data)))
                print("Total number of variables in the formula:",
                      self._num_attributes)
                """
                print("Attribute variables:", self._attributes)
                print("Auxiliary variables:", self._auxiliary_variables)
                print("sensitive feature: ", self._sensitive_attributes)
                if(self.encoding != "Enum-correlation"):
                    print("\nprobabilities:", self._probs)
                """

            # For SSAT-based approach, we require to negate the classifier to learn the least favored group.
            # For subset-sum based encoding, it is not required
            # if("efficient" in self.encoding):
            #     lr_neg = linear_classifier_wrap_Wrapper(weights=self._weights, attributes = self._attributes, sensitive_attributes = self._sensitive_attributes, bias=self._bias, negate=True, verbose=False)
            #     self._num_attributes_neg = lr_neg.num_attributes
            #     self._attributes_neg = lr_neg.attributes
            #     self._classifier_neg = lr_neg.classifier
            #     self._sensitive_attributes_neg = lr_neg.sensitive_attributes
            #     self._auxiliary_variables_neg = lr_neg.auxiliary_variables

        elif(self._model_name == "decision-tree"):
            _sensitive_attributes = utils.get_sensitive_attibutes(
                self.given_sensitive_attributes, self._data.columns.to_list())
            dt_pos = dtWrapper(self.model, self._data, self._data.columns.tolist(
            ), _sensitive_attributes, verbose=self._verbose)
            self._classifier = dt_pos.classifier
            self._num_attributes = dt_pos.num_attributes
            self._attributes = dt_pos.attributes
            self._sensitive_attributes = dt_pos.sensitive_attributes
            self._auxiliary_variables = dt_pos.auxiliary_variables
            self._variable_attribute_map = dt_pos.variable_attribute_map
            self._probs = dt_pos.compute_probability(
                self._data, verbose=self._verbose)
            self._saved_dt_model = dt_pos

            if("efficient" in self.encoding):
                dt_neg = dtWrapper(self.model, self._data, self._data.columns.tolist(
                ), _sensitive_attributes, negate=True, verbose=False)
                self._classifier_neg = dt_neg.classifier
                self._attributes_neg = dt_neg.attributes
                self._num_attributes_neg = dt_neg.num_attributes
                self._sensitive_attributes_neg = dt_neg.sensitive_attributes
                self._auxiliary_variables_neg = dt_neg.auxiliary_variables

        elif(self._model_name == "CNF"):
            self._attributes, self._sensitive_attributes, self._probs, self._variable_attribute_map, self._column_info = utils.get_statistics_from_df(
                self._data, self.given_sensitive_attributes, verbose=self._verbose)
            __classifier = self.model.get_selected_column_index()
            self._classifier = [[] for _ in __classifier]
            for idx in range(len(__classifier)):
                for (_var, _phase) in __classifier[idx]:
                    if(_phase >= 0):
                        self._classifier[idx].append(_var + 1)
                    else:
                        self._classifier[idx].append(-1 * (_var + 1))
            self._auxiliary_variables = []
            self._num_attributes = len(
                self._attributes + list(chain.from_iterable(self._sensitive_attributes)))

            assert self._num_attributes == len(self._data.columns)

        else:
            print(self._model_name, "is not defined")
            raise ValueError

    def _compute(self):

        self._meta_sensitive_groups = []
        for _group in self._sensitive_attributes:
            if(len(_group) == 1):
                _group = [_group[0], -1*_group[0]]
            self._meta_sensitive_groups.append(len(_group))

        min_value = None
        max_value = None
        self._execution_error = False
        self.most_favored_group = None
        self.least_favored_group = None

        if("dependency" in self.encoding or self.encoding in ['Learn-correlation', 'Learn-efficient-correlation']):
            _sensitive_attributes = []
            for _group in self._sensitive_attributes:
                if(len(_group) == 1):
                    _group = [_group[0], -1*_group[0]]
                _sensitive_attributes.append(_group)

            # flatten to 1d list
            all_sensitive_attributes = [
                abs(_var) for _group in self._sensitive_attributes for _var in _group]

            notears_time_start = time()

            # construct conditional probability table based on dependency graph
            edges = None
            if(self.encoding in ['Learn-correlation', 'Learn-efficient-correlation']):
                edges, self._graph_edges = dependency_utils.compound_group_correlation(
                    self._sensitive_attributes, self._attributes, self._discretized_group_attributes)
            else:
                """
                Alternate approach: 1. probabilistic graphical model 2. Notears
                """

                if(self._given_dependency_graph is None):
                    if(True):
                        """
                            Learn DAG
                        """

                        if(False):
                            edges, self._graph_edges, flag = dependency_utils._call_notears(self._transform(
                                self._data.copy()), self._sensitive_attributes, verbose=self._verbose, filename=self._filename)
                            if(not flag):
                                print(
                                    "Notears cannot learn dependency within timeout!!")
                                self._execution_error = True
                        else:
                            edges, self._graph_edges, flag = dependency_utils._call_pgmpy(self._transform(self._data.copy(
                            )), self._sensitive_attributes, verbose=self._verbose, threshold=self._dag_threshold)
                            if(not flag):
                                print(
                                    "pgmpy cannot learn dependency within timeout!!")
                                self._execution_error = True

                            # print(edges)

                        # edges, self._graph_edges = dependency_utils.default_dependency(all_sensitive_attributes, self._attributes)
                    else:
                        """
                            Learn DAG on continuous feature-space
                        """
                        self._given_dependency_graph = utils.call_notears(
                            self._data.copy(), regularizer=self._notears_regularizer)
                        edges, self._graph_edges = self._construct_edges_from_given_dependency()

                    # edges, self._graph_edges = dependency_utils.default_dependency(all_sensitive_attributes, self._attributes)

                else:
                    # Use provided dependency graph
                    edges, self._graph_edges = self._construct_edges_from_given_dependency()

            # refine dependency constraints
            # edges = dependency_utils.refine_dependency_constraints(self._sensitive_attributes, edges)

            # parameter estimate
            if(False):
                edge_weights = self._parameter_estimation_frequency(edges)
            else:
                edge_weights = dependency_utils.Bayesian_estimate(
                    self._transform(self._data.copy()), edges, self._graph_edges)

            self._time_taken_notears = time() - notears_time_start

            if((self._model_name == "linear-model" and "Learn" in self.encoding)):
                dependency = dependency_utils.Dependency(
                    [], {}, {}, self._num_attributes)
            else:

                if(self._verbose):
                    print("\n\nBefore encoding dependency")
                    print("Attribute variables:", self._attributes)
                    print("Auxiliary variables:", self._auxiliary_variables)
                    print("sensitive feature: ", self._sensitive_attributes)
                    print("Classifier:", self._classifier)

                    print("\n")
                    print("Neg Attribute variables:", self._attributes)
                    print("Neg Auxiliary variables:",
                          self._auxiliary_variables)
                    print("sensitive feature: ", self._sensitive_attributes)

                    # print("Neg classifier:", self._classifier_neg)
                    if(self.encoding != "Enum-correlation"):
                        print("probabilities:", self._probs)
                    print("Dependency edges:", edges)
                    print("Original edges", self._graph_edges)
                    print("\n")

                dependency = dependency_utils.Dependency(edges, edge_weights, self._probs, variable_header=self._num_attributes if self._num_attributes_neg is None else max(
                    self._num_attributes, self._num_attributes_neg))

                # redefine
                # sensitive attribute can appear in auxuliary variable when it is a dependent node in the causal graph
                # we avoid this case
                self._attributes = list(set(
                    [var for var in self._attributes if var not in dependency.auxiliary_variables] + dependency.attributes))
                self._probs = dependency.probs
                self._auxiliary_variables += [
                    var for var in dependency.auxiliary_variables if var not in all_sensitive_attributes]

                if(self._verbose):
                    """
                    print(dependency)
                    print("Attribute variables:", self._attributes)
                    print("Auxiliary variables:", self._auxiliary_variables)
                    print("sensitive feature: ", self._sensitive_attributes)
                    if(self.encoding != "Enum-correlation"):
                        print("\nprobabilities:", self._probs)
                    """
                    print("\nMapping", dependency.introduced_variables_map)
                    print("\n\n\n")

            if(self.encoding == "Enum-dependency"):
                self._encode_path_specific_causal_fairness(
                    variable_map=dependency.introduced_variables_map)
                max_value, min_value = self._run_Enum(
                    dependency_constraints=dependency.CNF)

            elif(self.encoding == "Learn-efficient-dependency" or self.encoding == "Learn-efficient-correlation"):

                if(not ((self._model_name == "linear-model" and "Learn" in self.encoding))):
                    self._attributes_neg = list(set(
                        [var for var in self._attributes_neg if var not in dependency.auxiliary_variables] + dependency.attributes))
                    self._auxiliary_variables_neg += [
                        var for var in dependency.auxiliary_variables if var not in all_sensitive_attributes]

                self._encode_path_specific_causal_fairness(
                    variable_map=dependency.introduced_variables_map)
                max_value, min_value = self._run_Learn_efficient(
                    dependency_constraints=dependency.CNF, edge_weights=edge_weights)

            elif(self.encoding == "Learn-dependency" or self.encoding == "Learn-correlation"):
                self._encode_path_specific_causal_fairness(
                    variable_map=dependency.introduced_variables_map)
                max_value, min_value = self._run_Learn(
                    dependency_constraints=dependency.CNF)

            else:
                raise ValueError(self.encoding)

        elif(self.encoding == "Enum-correlation"):

            max_value, min_value = self._run_Enum_correlation()

        # Base encoding
        elif(self.encoding == "Enum"):
            self._encode_path_specific_causal_fairness()
            max_value, min_value = self._run_Enum()

        elif(self.encoding == "Learn-efficient"):
            self._encode_path_specific_causal_fairness()
            max_value, min_value = self._run_Learn_efficient()

        elif(self.encoding == "Learn"):
            self._encode_path_specific_causal_fairness()
            max_value, min_value = self._run_Learn()
        else:
            raise ValueError

        if(self._execution_error == True):
            # print("Never reaches here")
            if(self._verbose):
                print("Execution error occured")
            self.statistical_parity_difference = None
            self.disparate_impact_ratio = None

            return

        self.statistical_parity_difference = max_value - min_value
        if(max_value == min_value):
            self.disparate_impact_ratio = 1
        elif(max_value == 0):
            self.disparate_impact_ratio = float("inf")
        else:
            self.disparate_impact_ratio = float(min_value/max_value)

    def _run_Enum(self, dependency_constraints=[]):

        if(self._verbose > 1):
            print("\nProbabilities:")
            print(self._probs)

        # prepare for combination
        _sensitive_attributes = []
        for _group in self._sensitive_attributes:
            if(len(_group) == 1):
                _group = [_group[0], -1*_group[0]]
            _sensitive_attributes.append(_group)

        # For each combination, update answer
        min_value = 1
        max_value = 0
        _combinations = list(itertools.product(*_sensitive_attributes))
        for configuration in _combinations:
            if(self._verbose):
                print("configuration: ", configuration, ":",
                      self._get_group_from_configuration(configuration))

            fv = Fairness_verifier(timeout=float(
                self._timeout/len(_combinations)))
            fv.encoding_Enum_SSAT(self._classifier, self._attributes, self._sensitive_attributes, self._auxiliary_variables, self._probs, self._filename,
                                  dependency_constraints=dependency_constraints, sensitive_attributes_assignment=list(configuration), verbose=self._verbose)
            flag = fv.invoke_SSAT_solver(self._filename, verbose=self._verbose)

            # print(fv.formula)

            if(flag == True):
                self._execution_error = True
                break

            if(min_value > fv.sol_prob):
                min_value = fv.sol_prob
                self.least_favored_group = self._get_group_from_configuration(
                    configuration)
            if(max_value < fv.sol_prob):
                max_value = fv.sol_prob
                self.most_favored_group = self._get_group_from_configuration(
                    configuration)
            self.sensitive_group_statistics.append((list(self._get_group_from_configuration(
                configuration).items()), fv.sol_prob, fv.num_variables, fv.num_clauses))

        return max_value, min_value

    def _get_group_from_configuration(self, configuration):
        result = {}
        for var in configuration:
            if var > 0:
                if(self._model_name in ["CNF"]):
                    feature, threshold = self._variable_attribute_map[var]
                    result[feature] = ("==", threshold)
                elif(self._model_name == "decision-tree"):
                    (feature, threshold) = self._variable_attribute_map[var]
                    result[feature] = ("<", threshold)
                elif(self._model_name == "linear-model"):
                    feature, _, threshold, _, _ = self._variable_attribute_map[var]
                    result[feature] = ("==", threshold)

                else:
                    raise ValueError
            else:
                if(self._model_name in ["CNF"]):
                    feature, threshold = self._variable_attribute_map[-1 * var]
                    result[feature] = ("!=", threshold)
                elif(self._model_name == "decision-tree"):
                    (feature,
                     threshold) = self._variable_attribute_map[-1 * var]
                    result[feature] = (">=", threshold)
                elif(self._model_name == "linear-model"):
                    feature, _, threshold, _, _ = self._variable_attribute_map[-1 * var]
                    result[feature] = ("!=", threshold)

                else:
                    raise ValueError

        return result

    def _run_Learn_efficient(self, dependency_constraints=[], edge_weights={}):

        max_value = None
        min_value = None

        if(self._model_name == "linear-model"):

            sensitive_flatten = [
                abs(var) for group in self._sensitive_attributes for var in group]
            converted_weight = {}
            for idx, weight in enumerate(self._linear_model.weights):
                if(idx + 1 not in sensitive_flatten):
                    converted_weight[idx + 1] = weight
            for group in self._sensitive_attributes:
                if(len(group) == 1):
                    converted_weight[group[0]
                                     ] = self._linear_model.weights[group[0] - 1]
                else:
                    converted_weight[tuple(group)] = [
                        self._linear_model.weights[var - 1] for var in group]

            # Maximum probability
            fv = SubSetSumCount(weights=converted_weight, probs=self._probs, sum=-1 * self._linear_model.bias,
                                choice_variables=self._sensitive_attributes, conditional_probs=edge_weights, verbose=self._verbose)
            fv.compute()

            if(fv.execution_error == True):
                self._execution_error = True
                return None, None
            else:

                #
                fv.sol_prob = fv.sol_prob_max
                fv.assignment_to_exists_variables = fv.assignment_to_exists_variables_max

                max_value = fv.sol_prob
                self.most_favored_group = self._get_group(fv)
                if(self._verbose):
                    print("Most favored group:", self.most_favored_group)
                self.sensitive_group_statistics.append((list(self.most_favored_group.items(
                )), fv.sol_prob, len(self._linear_model.weights), len(edge_weights)))

            # Minimum probability
            # fv = SubSetSumCount(weights = converted_weight, probs = self._probs, sum = -1 * self._linear_model.bias, choice_variables=self._sensitive_attributes, find_maximum=False, conditional_probs=edge_weights, verbose=False)

                fv.sol_prob = fv.sol_prob_min
                fv.assignment_to_exists_variables = fv.assignment_to_exists_variables_min

                min_value = fv.sol_prob
                self.least_favored_group = self._get_group(fv)
                if(self._verbose):
                    print("Least favored group:", self.least_favored_group)

                self.sensitive_group_statistics.append((list(self.least_favored_group.items(
                )), fv.sol_prob, len(self._linear_model.weights), len(edge_weights)))

            return max_value, min_value

        # maximum probability
        fv = Fairness_verifier(timeout=float(self._timeout/2))
        fv.encoding_Learn_SSAT(self._classifier, self._attributes, self._sensitive_attributes, self._auxiliary_variables, self._probs,
                               self._filename, dependency_constraints=dependency_constraints, find_maximization=True,  verbose=self._verbose)
        flag = fv.invoke_SSAT_solver(
            self._filename, find_maximization=True, verbose=self._verbose)

        # print(fv.formula)

        if(flag == True):
            self._execution_error = True
            return None, None
        else:
            max_value = fv.sol_prob
            self.most_favored_group = self._get_group(fv)
            if(self._verbose):
                print("Most favored group:", self.most_favored_group)
            self.sensitive_group_statistics.append(
                (list(self.most_favored_group.items()), fv.sol_prob, fv.num_variables, fv.num_clauses))

        # minimum probability
        fv = Fairness_verifier(timeout=float(self._timeout/2))
        fv.encoding_Learn_SSAT(self._classifier_neg, self._attributes_neg, self._sensitive_attributes_neg, self._auxiliary_variables_neg,
                               self._probs, self._filename, dependency_constraints=dependency_constraints, find_maximization=True,  verbose=self._verbose)
        flag = fv.invoke_SSAT_solver(
            self._filename, find_maximization=False, verbose=self._verbose)

        min_value = fv.sol_prob
        self.least_favored_group = self._get_group(fv)
        if(self._verbose):
            print("Least favored group:", self.least_favored_group)

        self.sensitive_group_statistics.append(
            (list(self.least_favored_group.items()), fv.sol_prob, fv.num_variables, fv.num_clauses))

        return max_value, min_value

    def _run_Learn(self, dependency_constraints=[]):

        max_value = None
        min_value = None

        # maximum probability
        fv = Fairness_verifier(timeout=float(self._timeout/2))
        fv.encoding_Learn_SSAT(self._classifier, self._attributes, self._sensitive_attributes, self._auxiliary_variables, self._probs,
                               self._filename, dependency_constraints=dependency_constraints, find_maximization=True, verbose=self._verbose)
        flag = fv.invoke_SSAT_solver(
            self._filename, find_maximization=True, verbose=self._verbose)
        if(flag == True):
            self._execution_error = True
            return None, None
        else:
            max_value = fv.sol_prob
            self.most_favored_group = self._get_group(fv)
            if(self._verbose):
                print("Most favored group:", self.most_favored_group)
            self.sensitive_group_statistics.append(
                (list(self.most_favored_group.items()), fv.sol_prob, fv.num_variables, fv.num_clauses))

        # maximum probability by complementing the classifier
        fv = Fairness_verifier(timeout=float(self._timeout/2))
        fv.encoding_Learn_SSAT(self._classifier, self._attributes, self._sensitive_attributes, self._auxiliary_variables, self._probs,
                               self._filename,  dependency_constraints=dependency_constraints, find_maximization=False, verbose=self._verbose)
        flag = fv.invoke_SSAT_solver(
            self._filename, find_maximization=False, verbose=self._verbose)
        if(flag == True):
            self._execution_error = True
        else:
            min_value = fv.sol_prob
            self.least_favored_group = self._get_group(fv)
            if(self._verbose):
                print("Least favored group:", self.least_favored_group)
            self.sensitive_group_statistics.append(
                (list(self.least_favored_group.items()), fv.sol_prob, fv.num_variables, fv.num_clauses))

        return max_value, min_value

    def _get_group(self, fairness_verifier):

        if(fairness_verifier.assignment_to_exists_variables is None):
            return {}
        configuration = []
        for group in self._sensitive_attributes:
            if(len(group) == 1):
                if(group[0] in fairness_verifier.assignment_to_exists_variables):
                    configuration.append(group[0])
                else:
                    configuration.append(-1 * group[0])
            elif(len(group) > 1):
                reach = 0
                for var in group:
                    if(var in fairness_verifier.assignment_to_exists_variables):
                        configuration.append(var)
                        reach += 1
                assert reach == 1
            else:
                raise ValueError
        return self._get_group_from_configuration(configuration)

    def _encode_path_specific_causal_fairness(self, variable_map={}):
        # We are given mediator attributes as parameter in Justicia, which is stored in self.given_mediator_attributes.
        # Our goal is to learn corresponding mediator attributes once all parameters are computed and adjust probabilities
        # accordingly.
        self._mediator_attributes = []
        if(self.given_major_group == None):
            self._execution_error = True
            return

        if(len(self.given_mediator_attributes) == 0 or len(self.given_major_group) == 0):
            return

        for var in self._variable_attribute_map:
            if(type(self._variable_attribute_map[var]) == tuple):
                if(self._model_name == "decision-tree"):
                    (_feature, _threshold) = self._variable_attribute_map[var]
                    if(_feature in self.given_mediator_attributes):
                        self._mediator_attributes.append(var)
                elif(self._model_name == "linear-model"):
                    _feature, _, _threshold_1, _comparator_2, _threshold_2 = self._variable_attribute_map[
                        var]
                    if(_feature in self.given_mediator_attributes):
                        self._mediator_attributes.append(var)
                elif(self._model_name in ["CNF"]):
                    _feature, _threshold = self._variable_attribute_map[var]
                    if(_feature in self.given_mediator_attributes):
                        self._mediator_attributes.append(var)
                else:
                    raise ValueError(self._model_name)
            else:
                raise ValueError(self._model_name)
                if(self._variable_attribute_map[var] in self.given_mediator_attributes):
                    self._mediator_attributes.append(var)

        if(self._verbose):
            print("\n\n\nInfo on mediator attributes")
            print(self.given_mediator_attributes)
            print("Attributes in Justicia:", self._mediator_attributes)
            print("Major group:", self.given_major_group)
            print("\n\n\n")

        # construct a mask for the given major group
        mask = (True)
        for attribute in self.given_major_group:
            comparator, threshold = self.given_major_group[attribute]
            if(comparator == "=="):
                mask = mask & (self._data[attribute] == threshold)
            elif(comparator == "!="):
                mask = mask & (self._data[attribute] != threshold)
            elif(comparator == ">"):
                mask = mask & (self._data[attribute] > threshold)
            elif(comparator == ">="):
                mask = mask & (self._data[attribute] >= threshold)
            elif(comparator == "<"):
                mask = mask & (self._data[attribute] < threshold)
            elif(comparator == "<="):
                mask = mask & (self._data[attribute] <= threshold)
            else:
                raise ValueError(comparator)

        # compute probabilities of mediator variables for the major group
        mediator_probs = None
        if(self._model_name == "decision-tree"):
            mediator_probs = self._saved_dt_model.compute_probability(
                self._data[mask], verbose=False)
        elif(self._model_name in ["linear-model", "CNF"]):
            mediator_probs = utils.calculate_probs_linear_classifier_wrap(
                self._data[mask], self._column_info)
        else:
            raise ValueError

        if(self._verbose):
            print("Following intervention on probabilities")
        # In self._probs, update probabilities of mediator variables
        for mediator_attribute in self._mediator_attributes:
            if(mediator_attribute in variable_map):
                for each_map in variable_map[mediator_attribute]:
                    if(self._verbose):
                        print(mediator_attribute, "(" + str(each_map) + ")", ":",
                              self._probs[each_map], "->", mediator_probs[mediator_attribute])
                    self._probs[each_map] = mediator_probs[mediator_attribute]
            else:
                if(self._verbose):
                    print(mediator_attribute, ":",
                          self._probs[mediator_attribute], "->", mediator_probs[mediator_attribute])
                self._probs[mediator_attribute] = mediator_probs[mediator_attribute]

        if(self._verbose):
            print("\n\n\n")

    def _transform(self, orig_df):

        if(self._model_name == "CNF"):
            return orig_df

        df = pd.DataFrame()
        for variable in self._variable_attribute_map:
            if(type(self._variable_attribute_map[variable]) == tuple):
                if(self._model_name == "decision-tree"):
                    (_feature,
                     _threshold) = self._variable_attribute_map[variable]
                    df[variable] = (orig_df[_feature] <=
                                    _threshold).astype(int)
                elif(self._model_name == "linear-model"):
                    _feature, _, _threshold_1, _comparator_2, _threshold_2 = self._variable_attribute_map[
                        variable]
                    if(_comparator_2 == "<"):
                        df[variable] = ((orig_df[_feature] >= _threshold_1) & (
                            orig_df[_feature] < _threshold_2)).astype(int)
                    elif(_comparator_2 == "<="):
                        df[variable] = ((orig_df[_feature] >= _threshold_1) & (
                            orig_df[_feature] <= _threshold_2)).astype(int)
                    elif(_comparator_2 == "=="):
                        df[variable] = (orig_df[_feature] ==
                                        _threshold_1).astype(int)

                    else:
                        raise ValueError(_comparator_2)
                elif(self._model_name == "CNF"):
                    (_feature,
                     _threshold) = self._variable_attribute_map[variable]
                    df[variable] = (orig_df[_feature] ==
                                    _threshold).astype(int)
                else:
                    raise ValueError(self._model_name)
            else:
                raise ValueError(self._model_name)
                df[variable] = (
                    self._data[self._variable_attribute_map[variable]] == 1).astype(int)
        # reorder columns
        df = df[[i + 1 for i in range(len(self._variable_attribute_map))]]

        return df

    def _get_df_mask(self, dominating_var):
        # Each dominating var put constraints on the dataframe, which is captured by the mask.
        # When dominating var is negative, we simply calculate for positive var and finally complement it.

        mask = (True)
        negate_mask = False
        if(dominating_var < 0):
            # we need to negate the mask
            negate_mask = True
            dominating_var = -1 * dominating_var

        if(type(self._variable_attribute_map[dominating_var]) == tuple):
            if(self._model_name == "decision-tree"):
                (_feature,
                 _threshold) = self._variable_attribute_map[dominating_var]
                mask = mask & (self._data[_feature] <= _threshold)
            elif(self._model_name == "linear-model"):
                _feature, _, _threshold_1, _comparator_2, _threshold_2 = self._variable_attribute_map[
                    dominating_var]
                if(_comparator_2 == "<"):
                    mask = mask & (self._data[_feature] >= _threshold_1) & (
                        self._data[_feature] < _threshold_2)
                elif(_comparator_2 == "<="):
                    mask = mask & (self._data[_feature] >= _threshold_1) & (
                        self._data[_feature] <= _threshold_2)
                elif(_comparator_2 == "=="):
                    mask = mask & (self._data[_feature] == _threshold_1)
                else:
                    raise ValueError(_comparator_2)
            elif(self._model_name == "CNF"):
                (_feature,
                 _threshold) = self._variable_attribute_map[dominating_var]
                mask = mask & (self._data[_feature] == _threshold)
            else:
                raise ValueError(self._model_name)
        else:
            raise ValueError(self._model_name)

        if(negate_mask):
            mask = ~mask

        return mask

    def _parameter_estimation_frequency(self, edges):

        # key is parent, value is set of childs
        edge_weights = {}
        dic = {}
        for (a, b) in edges:
            if(a in dic):
                dic[a].append(b)
            else:
                dic[a] = [b]

        for key in dic:
            mask = (True)

            if(isinstance(key, tuple)):
                # multuple parents
                for each_var in key:
                    mask = mask & self._get_df_mask(each_var)
                pass

            else:
                mask = mask & self._get_df_mask(key)

            if(self._model_name == "decision-tree"):
                marginal_probs = self._saved_dt_model.compute_probability(
                    self._data[mask], verbose=False, selected_columns=dic[key])
            elif(self._model_name in ["linear-model", "CNF"]):
                marginal_probs = utils.calculate_probs_linear_classifier_wrap(
                    self._data[mask], self._column_info, selected_columns=dic[key])
            else:
                raise ValueError

            for var in marginal_probs:
                edge_weights[(key, var)] = marginal_probs[var]

        assert len(edges) == len(edge_weights)

        """
        print(edge_weights)
        print("\n\n")


        print(dic)
        print("\n\n")

        quit()

        # Dominating vars: list of nodes where a directed edge starts
        dominating_vars = list(set([a for (a,b) in edges]))            
        # dominating_vars should have all combinations
        edge_weights = {}
        print(dominating_vars)
        print("\n\n")
        print(edges)
        print("\n\n")


        quit()

        for dominating_var in dominating_vars:

            mask = (True)

            if(isinstance(dominating_var, tuple)):
                # multuple parents
                for each_var in dominating_var:
                    mask = self._get_df_mask(each_var, mask)                        
                pass

            
            else:
                mask = self._get_df_mask(dominating_var, mask)
            
                  
            
            if(self._model_name == "decision-tree"):
                marginal_probs = self._saved_dt_model.compute_probability(self._data[mask], verbose=False)
            elif(self._model_name in ["linear-model", "CNF"]):
                marginal_probs = utils.calculate_probs_linear_classifier_wrap(self._data[mask], self._column_info)
            else:
                raise ValueError

            # if(self._data[mask].empty):
            #     print("Empty dataframe") 
            #     print(dominating_var)
            #     print(marginal_probs)
            #     print()
            
            for var in marginal_probs:
                edge_weights[(dominating_var,var)] = marginal_probs[var]
        
        print(edge_weights)
        print("\n\n")
        

        # pruning edge_weights
        revised_edge_weights = {}
        for edge in edges:
            revised_edge_weights[edge] = edge_weights[edge]
         
        edge_weights = revised_edge_weights

        print(edge_weights)
        print("\n\n")
        """
        # quit()

        return edge_weights

    def _run_Enum_correlation(self):
        _sensitive_attributes = []
        for _group in self._sensitive_attributes:
            if(len(_group) == 1):
                _group = [_group[0], -1*_group[0]]
            _sensitive_attributes.append(_group)

        min_value = 1
        max_value = 0
        _combinations = list(itertools.product(*_sensitive_attributes))
        for configuration in _combinations:
            if(self._verbose):
                print("configuration: ", configuration, ":",
                      self._get_group_from_configuration(configuration))
                # ":", [self._variable_attribute_map[var] if var > 0 else "not " + self._variable_attribute_map[-1 * var] for var in configuration]

            # recalculate probs according to conditional probabiities of sensitve attributes
            mask = (True)
            for _attribute in configuration:
                mask = mask & self._get_df_mask(_attribute)

            if(self._model_name == "decision-tree"):
                self._probs = self._saved_dt_model.compute_probability(
                    self._data[mask], verbose=False)
            elif(self._model_name in ["linear-model", "CNF"]):
                self._probs = utils.calculate_probs_linear_classifier_wrap(
                    self._data[mask], self._column_info)
            else:
                raise ValueError

            if(self._verbose):
                print(self._probs)

            self._encode_path_specific_causal_fairness()

            fv = Fairness_verifier(timeout=float(
                self._timeout/len(_combinations)))
            fv.encoding_Enum_SSAT(self._classifier, self._attributes, self._sensitive_attributes, self._auxiliary_variables,
                                  self._probs, self._filename, sensitive_attributes_assignment=list(configuration), verbose=self._verbose)
            flag = fv.invoke_SSAT_solver(self._filename, verbose=self._verbose)

            if(flag == True):
                self._execution_error = True
                max_value == min_value
                break

            if(min_value > fv.sol_prob):
                min_value = fv.sol_prob
                self.least_favored_group = self._get_group_from_configuration(
                    configuration)
            if(max_value < fv.sol_prob):
                max_value = fv.sol_prob
                self.most_favored_group = self._get_group_from_configuration(
                    configuration)
            self.sensitive_group_statistics.append((list(self._get_group_from_configuration(
                configuration).items()), fv.sol_prob, fv.num_variables, fv.num_clauses))

        return max_value, min_value

    def _retrieve_model_name(self):
        if(isinstance(self.model, imli)):
            return "CNF"
        if(isinstance(self.model, DecisionTreeClassifier)):
            return "decision-tree"
        if(isinstance(self.model, SVC)):
            return "linear-model"
        if(isinstance(self.model, LogisticRegression)):
            return "linear-model"
        if(isinstance(self.model, Poison_Model)):
            return "linear-model"
        raise ValueError(str(self.model) + " not supported in Justicia")

    def _construct_edges_from_given_dependency(self):
        """
            Given edge information for original attributes. We convert them to SSAT variables. 
            For a real-valued attribute, its discretized variables are used. Hence, each dependency tuple is multiplied for 
            all combinations of discretized attributes.
        """

        # complement variable_attribute_map
        attribute_variable_map = {}
        for key in self._variable_attribute_map:
            # Each variable in SSAT should map unique attribute in input space
            assert self._variable_attribute_map[key] not in attribute_variable_map
            attribute_variable_map[self._variable_attribute_map[key]] = key

        """
            Refine user provided dependency graph for the following reasons. 
                1. If there is an incomding edge to a sensitive variable, we reverse the edge direction. Because, in our encoding all
                    sensitive variables are existentially quantified and thus their fixed (also permutation) assignment are considered. 
        """
        # TODO bad implementation. Iterates all sensitive attributes for each edge in given_dependency_graph

        new_edges = []
        deleted_edges = []
        original_edges = self._given_dependency_graph.edges()

        # reverse edge on sensitive attribute
        for a, b in original_edges:
            if b in self.given_sensitive_attributes:
                # reverse edge
                new_edges.append((b, a))
                deleted_edges.append((a, b))
                if(self._verbose):
                    print("Reversed edge: ", b, "->", a)
                continue
            for sensitive_attribute in self.given_sensitive_attributes:
                if b.startswith(sensitive_attribute) and "_" in b:
                    new_edges.append((b, a))
                    deleted_edges.append((a, b))
                    if(self._verbose):
                        print("Reversed edge: ", b, "->", a)

            # assert reach_count < 2, str(self.given_sensitive_attributes) + " has more than one matching for " + str(b)

        # refinement
        if(len(new_edges) > 0):
            self._given_dependency_graph.remove_edges_from(deleted_edges)
            self._given_dependency_graph.add_edges_from(new_edges)

        if(self._verbose):
            print("\nRefining given dependency graph")
            print(self._given_dependency_graph.edges())

        # Map original attribute to discretized variables
        attribute_dic = {}
        for attribute in attribute_variable_map:
            _feature = None
            if(type(attribute) == tuple):
                if(self._model_name in ["decision-tree", "CNF"]):
                    (_feature, _threshold) = attribute

                elif(self._model_name == "linear-model"):
                    _feature, _, _threshold_1, _comparator_2, _threshold_2 = attribute
                else:
                    raise ValueError(self._model_name)
            else:
                _feature = attribute

            for node in self._given_dependency_graph.nodes():
                if(_feature.startswith(node)):
                    if(node in attribute_dic):
                        attribute_dic[node].append(
                            attribute_variable_map[attribute])
                    else:
                        attribute_dic[node] = [
                            attribute_variable_map[attribute]]

        # print("\nattribute variable map:", attribute_variable_map)
        # print("\nattribute to var map", attribute_dic)

        # Consider all combinations of discretized variables
        edges = []
        for a, b in self._given_dependency_graph.edges():
            if(a in attribute_dic and b in attribute_dic):
                for var_a in attribute_dic[a]:
                    for var_b in attribute_dic[b]:
                        edges.append((var_a, var_b))
            else:
                # attributes not participating in classifier, hence not considered
                pass
        # print()
        # print(edges)

        edges = dependency_utils.refine_dependency_constraints(
            self._sensitive_attributes, edges)
        # print()
        return dependency_utils.do_combinations(edges), edges
