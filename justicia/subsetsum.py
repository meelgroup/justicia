import numpy as np
import networkx as nx
from time import time


class SubSetSumCount():
    """
        Limitation : 1. Does not consider conditional probabilities for choice variables
    """

    def __init__(self,  weights={},
                 probs={},
                 sum=100,
                 choice_variables=[],
                 conditional_probs={},
                 sort_heuristic=True,
                 verbose=True,
                 timeout=900):
        self.verbose = verbose
        self.sort_heuristic = sort_heuristic
        self.weights = weights
        self.varCount = 0
        for weight in self.weights:
            if(isinstance(weight, tuple)):
                self.varCount += len(weight)
            else:
                self.varCount += 1

        self.timeout = timeout
        self._start_time = time()
        self.probs = probs
        self.conditional_probs = conditional_probs
        self.bias = sum
        self.lookup = {}
        self.choice = choice_variables  # index of existential/universal variables
        self.choice_assignment = {}  # assignment of choice variables

    def compute(self):
        self.preprocess()
        self.cnt = 0
        self.sol_prob = None
        self.execution_error = False

        if(len(self.conditional_probs) == 0):

            n_count = len(self.resolve_order)
            choice_count = len(self.choice)

            self.choice_assignment_max = {}
            self.choice_assignment_min = {}

            self.find_max = True
            try:
                self.sol_prob_max = self.subsetSum(
                    n_count, self.bias, choice_phase={})
            except Exception as e:
                if(self.verbose):
                    print(e)
                return
            # print(self.lookup)

            for n, sum in list(self.lookup.keys()):
                if n > n_count - choice_count:
                    del self.lookup[(n, sum)]

            self.find_max = False
            try:
                self.sol_prob_min = self.subsetSum(
                    n_count, self.bias, choice_phase={})
            except Exception as e:
                if(self.verbose):
                    print(e)
                return
            # print(self.lookup)

            if(not self.execution_error):
                self.assignment_to_exists_variables_max = self.optimal_assignment(
                    self.choice_assignment_max, self.sol_prob_max)
                self.assignment_to_exists_variables_min = self.optimal_assignment(
                    self.choice_assignment_min, self.sol_prob_min)
                if(self.verbose):
                    print(self.sol_prob_max, self.sol_prob_min)
                    print("Lookup table len", len(self.lookup))
                    print(self.assignment_to_exists_variables_max,
                          self.assignment_to_exists_variables_min)

        else:
            self.choice_assignment_max = {}
            self.choice_assignment_min = {}
            # self.sol_prob_max, self.sol_prob_min = self.subsetSum_conditionals(len(self.resolve_order), self.bias, choice_phase = {}, all_variables_assignment = {})
            try:
                self.sol_prob_max, self.sol_prob_min = self.subsetSum_conditionals(
                    len(self.resolve_order), self.bias, choice_phase={}, all_variables_assignment={})
            except Exception as e:
                if(self.verbose):
                    print(e)
                return

            # print(self.lookup)

            if(not self.execution_error):
                self.assignment_to_exists_variables_max = self.optimal_assignment(
                    self.choice_assignment_max, self.sol_prob_max)
                self.assignment_to_exists_variables_min = self.optimal_assignment(
                    self.choice_assignment_min, self.sol_prob_min)

                if(self.verbose):
                    print(self.sol_prob_max, self.sol_prob_min)
                    print("Lookup table len", len(self.lookup))
                    print(self.assignment_to_exists_variables_max,
                          self.assignment_to_exists_variables_min)

    def preprocess(self):

        self._choice_flatten = []

        # a mapping of variable to choice variable index
        choice_var_to_group_map = {}
        for idx, group in enumerate(self.choice):
            for var in group:
                var = abs(var)
                self._choice_flatten.append(var)
                choice_var_to_group_map[var] = idx

        self.choiceVarCount = len(self._choice_flatten)
        # index of random variables
        self.chance = [i for i in range(
            1, self.varCount + 1) if i not in self._choice_flatten]
        self.inner_choice = []
        self.converted_conditional_probs = {}
        self.conditional_relations = {}

        # compute sum of negative weights
        self._sum_negative_weight = 0
        self._sum_positive_weight = 0
        # _choice_var_max_weight = {}

        for key in self.weights:

            if(isinstance(key, tuple)):
                for weight in self.weights[key]:
                    if(weight < 0):
                        self._sum_negative_weight += weight
                    else:
                        self._sum_positive_weight += weight
                # _choice_var_max_weight[key] = np.array(self.weights[key]).max()
            else:
                weight = self.weights[key]
                if(weight < 0):
                    self._sum_negative_weight += weight
                else:
                    self._sum_positive_weight += weight
                # if(key in self._choice_flatten):
                    # _choice_var_max_weight[(key,)] = weight

        # sort choice variables based on descending maximum weights
        # self.choice = [tuple(group) for group in self.choice]
        # self.choice.sort(key=_choice_var_max_weight.get, reverse=True)
        # self.choice = [list(group) for group in self.choice]
        # print(self.choice)

        # parent_list = []
        graph = nx.DiGraph()

        for key in self.conditional_probs:
            parent, child = key
            if(not isinstance(parent, tuple)):
                parent = (parent,)
            self.converted_conditional_probs[parent +
                                             (child,)] = self.conditional_probs[key]
            self.converted_conditional_probs[parent +
                                             (-1 * child,)] = 1 - self.conditional_probs[key]
            relation = tuple([abs(var) for var in parent] + [abs(child)])

            if(relation[-1] not in self.conditional_relations):
                self.conditional_relations[relation[-1]] = relation
                # self.conditional_relations.append(relation)

            if(child in self.chance):
                self.chance.remove(child)
                # to compute conditional probability
                self.inner_choice.append(child)
            elif(child in self._choice_flatten):
                raise ValueError(
                    "A choice variable cannot have indegree edge in current formulation")

            # graph construction
            for var in parent:
                var = abs(var)
                graph.add_edge(var, abs(child))

            self.probs.pop(child, None)
            assert child not in self.probs, str(
                child) + " has conditional probabilitiy but it exists in independent probability table"

        # print("edges", graph.edges, "nodes", graph.nodes)
        # Ordering of nodes in the graph
        self.topological_sort = list(nx.topological_sort(graph))

        parent_list = []  # this node does not have any parent itself
        revised_choice_variable_order = []  # to track nodes within a choice group

        revised_inner_choice = []
        for var in self.topological_sort:
            if(var in self.inner_choice):
                revised_inner_choice.append(var)
                continue
            if(var in self._choice_flatten):
                if(choice_var_to_group_map[var] not in revised_choice_variable_order):
                    revised_choice_variable_order.append(
                        choice_var_to_group_map[var])
                continue
            parent_list.append(var)

        self.inner_choice = revised_inner_choice

        choice_order_first = []  # placed at first
        choice_order_last = []  # placed at last
        self._effective_choice_var = []  # no conditional probabilities envolved
        for idx, group in enumerate(self.choice):
            if(idx in revised_choice_variable_order):
                choice_order_first.append(group)
            else:
                choice_order_last.append(group)
                self._effective_choice_var.append(tuple(group))

        # no conditional probabilities envolved
        self._effective_chance_var = [
            var for var in self.chance if var not in parent_list]

        reverse = True  # descending, False = ascending
        if(self.sort_heuristic):
            # sort all_variables
            if(0.5*(self._sum_positive_weight - self._sum_negative_weight) < self.bias):
                reverse = False
            self._effective_chance_var.sort(
                key=self.weights.get, reverse=reverse)
            parent_list.sort(key=self.weights.get, reverse=reverse)
            # self.inner_choice.sort(key=self.weights.get, reverse=reverse)

        self.choice = choice_order_first + choice_order_last
        # self.choice = choice_order_last + choice_order_first
        self.resolve_order = self.choice + parent_list + \
            self.inner_choice + self._effective_chance_var

        if(self.verbose):
            print("\n\n============================== SubSet-Sum =============\n")
            # print("Converted conditional probs", self.converted_conditional_probs)
            print("Conditional relations", self.conditional_relations)
            if(self.verbose > 1):
                print("Conditional probabilities", self.conditional_probs)
            print("Chance variable", self.chance)
            print("Choice variable", self.choice)
            print("Inner choice variable", self.inner_choice)
            print("BN parents", parent_list)
            # print("Variable to choice map", self.var_to_choice_map)
            print("Weights", self.weights)
            if(self.verbose > 1):
                print("Marginal probabilities", self.probs)
            print("Threshold", self.bias)
            print("Resolve order", self.resolve_order)
            print("Var count", self.varCount)
            print("Effective choice variables", self._effective_choice_var)
            print("Effective chance variables", self._effective_chance_var)
            print("Probs", self.probs)
            print("Topological sort", self.topological_sort)
            print("Max negative weight", self._sum_negative_weight)
            print("Max positive weight", self._sum_positive_weight)
            print("Ordering descending?", not False)

            # print("Converted index", self.convertedIndex)
            print()

        assert len(self.resolve_order) == len(self.weights)
        assert tuple(self.choice) == tuple(
            self.resolve_order[:len(self.choice)])
        for i in self.chance:
            assert i in self.probs, str(i) + " is not assigned probability"

    def subsetSum(self, n, sum, choice_phase):
        # print(len(self.resolve_order) - n + 1, sum)
        if(time() - self._start_time > self.timeout):
            self.execution_error = True
            raise RuntimeError("Time limit exceeds")
        # self.cnt += 1

        # Early termination
        if(sum <= self._sum_negative_weight):
            # print("Advanced termination", len(self.resolve_order) - n + 1, sum)

            return 1

        if(sum > self._sum_positive_weight):
            return 0

        # When all weights are exhausted
        if n == 0:
            if sum <= 0:
                return 1
            else:
                return 0

        key = (n, sum)
        # if the subproblem is seen for the first time, solve it and
        # store its result in a dictionary
        if key not in self.lookup:

            # A is resolved from left side
            item_index = self.resolve_order[-1 * n]

            if(isinstance(item_index, list)):
                return_values = []
                choice_phase_dummies = []
                if(len(item_index) == 1):

                    if(self.weights[item_index[0]] > 0):
                        if(self.find_max):
                            choice_phase_dummies.append(choice_phase.copy())
                            choice_phase_dummies[-1][item_index[0]] = 1

                            # Case 1. Include current item
                            return_values.append(self.subsetSum(
                                n - 1, sum - self.weights[item_index[0]], choice_phase_dummies[-1]))

                        else:
                            choice_phase_dummies.append(choice_phase.copy())
                            choice_phase_dummies[-1][item_index[0]] = 0

                            # Case 2. Exclude the current item
                            # the remaining items `n-1`
                            return_values.append(self.subsetSum(
                                n - 1, sum, choice_phase_dummies[-1]))
                    else:
                        if(self.find_max):
                            choice_phase_dummies.append(choice_phase.copy())
                            choice_phase_dummies[-1][item_index[0]] = 0

                            # Case 1. Include current item
                            return_values.append(self.subsetSum(
                                n - 1, sum, choice_phase_dummies[-1]))

                        else:
                            choice_phase_dummies.append(choice_phase.copy())
                            choice_phase_dummies[-1][item_index[0]] = 1

                            # Case 2. Exclude the current item
                            # the remaining items `n-1`
                            return_values.append(self.subsetSum(
                                n - 1, sum - self.weights[item_index[0]], choice_phase_dummies[-1]))

                else:
                    template_choice_phase = choice_phase.copy()

                    max_index = 0
                    min_index = 0
                    max_weight = self.weights[tuple(item_index)][0]
                    min_weight = self.weights[tuple(item_index)][0]

                    for var_idx, _ in enumerate(item_index):
                        template_choice_phase[item_index[var_idx]] = 0

                        if(max_weight < self.weights[tuple(item_index)][var_idx]):
                            max_index = var_idx
                            max_weight = self.weights[tuple(
                                item_index)][var_idx]

                        if(min_weight > self.weights[tuple(item_index)][var_idx]):
                            min_index = var_idx
                            min_weight = self.weights[tuple(
                                item_index)][var_idx]

                    # print(max_index, max_weight, min_index, min_weight)

                    if(self.find_max):
                        choice_phase_dummies.append(
                            template_choice_phase.copy())
                        choice_phase_dummies[-1][item_index[max_index]] = 1
                        return_values.append(self.subsetSum(
                            n - 1, sum - max_weight, choice_phase_dummies[-1]))

                    else:
                        choice_phase_dummies.append(
                            template_choice_phase.copy())
                        choice_phase_dummies[-1][item_index[min_index]] = 1
                        return_values.append(self.subsetSum(
                            n - 1, sum - min_weight, choice_phase_dummies[-1]))

                assert len(return_values) <= 2

                if(self.find_max):
                    self.lookup[key] = -100
                else:
                    self.lookup[key] = 100

                if(self.find_max):
                    for i in range(len(return_values)):
                        if(self.lookup[key] < return_values[i]):
                            self.lookup[key] = return_values[i]
                            if(len(choice_phase_dummies[i]) == self.choiceVarCount):
                                self.choice_assignment_max[tuple(
                                    choice_phase_dummies[i].items())] = self.lookup[key]

                else:
                    for i in range(len(return_values)):
                        if(self.lookup[key] > return_values[i]):
                            self.lookup[key] = return_values[i]
                            if(len(choice_phase_dummies[i]) == self.choiceVarCount):
                                self.choice_assignment_min[tuple(
                                    choice_phase_dummies[i].items())] = self.lookup[key]

            else:

                # Case 1. Include current item
                include = self.subsetSum(
                    n - 1, sum - self.weights[item_index], choice_phase)

                # Case 2. Exclude the current item
                exclude = self.subsetSum(n - 1, sum, choice_phase)

                result = None
                result = None
                if(item_index in self.inner_choice):

                    raise RuntimeError()

                    # val_include = 1
                    # val_exclude = 1

                    # for relation, flag in self.conditional_relations:
                    #     if(relation[-1] == item_index):
                    #         val_include *= self.converted_conditional_probs.get(tuple([index if all_variables_assignment_include[index] == 1 else -1 * index for index in relation]), 1)
                    #         val_exclude *= self.converted_conditional_probs.get(tuple([index if all_variables_assignment_exclude[index] == 1 else -1 * index for index in relation]), 1)

                    # result = include * val_include + exclude * val_exclude

                elif(item_index in self.chance):
                    result = (self.probs[item_index] * include) + \
                        (1 - self.probs[item_index]) * exclude
                else:
                    raise ValueError
                self.lookup[key] = result
        # else:

            # print("collision", len(self.resolve_order) - n + 1, sum)
        # return solution to the current subproblem
        return self.lookup[key]

    def subsetSum_conditionals(self, n, sum, choice_phase, all_variables_assignment):
        # print(len(self.resolve_order) - n + 1, sum)
        if(time() - self._start_time > self.timeout):
            self.execution_error = True
            raise RuntimeError("Time limit exceeds")

        """
            Returns max_probability, min_probability
        """
        # print(all_variables_assignment)

        # Early termination
        if(sum <= self._sum_negative_weight):
            return 1, 1

        if(sum > self._sum_positive_weight):
            return 0, 0

        # When all weights are exhausted
        if n == 0:
            if sum <= 0:
                return 1, 1
            else:
                return 0, 0

        result_max = None
        result_min = None

        item_index = self.resolve_order[-1 * n]  # A is resolved from left side

        # Choice variable
        if(isinstance(item_index, list)):

            # Lookup table for dynamic approach, applied on (effective) choice and chance variables.
            key = None
            selective_run = False
            if(tuple(item_index) in self._effective_choice_var):
                selective_run = True
                key = (n, sum)
                if(key in self.lookup):
                    # print("Hit", key)
                    # print("collision", len(self.resolve_order) - n + 1, sum)
                    return self.lookup[key]

            return_values = []
            choice_phase_dummies = []

            # Binary choice variable
            if(len(item_index) == 1):

                # Necessary for joint progability computation
                all_variables_assignment_include = all_variables_assignment.copy()
                all_variables_assignment_exclude = all_variables_assignment.copy()

                if(item_index[0] in self.topological_sort):
                    all_variables_assignment_include[item_index[0]] = 1
                    all_variables_assignment_exclude[item_index[0]] = 0

                choice_phase_dummies.append(choice_phase.copy())
                choice_phase_dummies[-1][item_index[0]] = 1

                # Case 1. Include current item
                return_values.append(self.subsetSum_conditionals(
                    n - 1, sum - self.weights[item_index[0]], choice_phase_dummies[-1], all_variables_assignment_include))

                choice_phase_dummies.append(choice_phase.copy())
                choice_phase_dummies[-1][item_index[0]] = 0

                # Case 2. Exclude the current item
                # the remaining items `n-1`
                return_values.append(self.subsetSum_conditionals(
                    n - 1, sum, choice_phase_dummies[-1], all_variables_assignment_exclude))

            # Categorical choice variable
            else:
                template_choice_phase = choice_phase.copy()
                template_all_variables_assignment = all_variables_assignment.copy()
                for var_idx, _ in enumerate(item_index):
                    template_choice_phase[item_index[var_idx]] = 0
                    if(item_index[var_idx] in self.topological_sort):
                        template_all_variables_assignment[item_index[var_idx]] = 0

                if(selective_run):
                    # contains max and min weight
                    considered_weights = [np.array(self.weights[tuple(item_index)]).max(
                    ), np.array(self.weights[tuple(item_index)]).min()]
                    # print(len(considered_weights))

                for var_idx, _ in enumerate(item_index):
                    choice_phase_dummies.append(template_choice_phase.copy())
                    choice_phase_dummies[-1][item_index[var_idx]] = 1
                    temp_assignment = template_all_variables_assignment.copy()
                    if(item_index[var_idx] in self.topological_sort):
                        temp_assignment[item_index[var_idx]] = 1
                    if(selective_run):
                        if(self.weights[tuple(item_index)][var_idx] in considered_weights):
                            considered_weights.remove(
                                self.weights[tuple(item_index)][var_idx])
                            return_values.append(self.subsetSum_conditionals(
                                n-1, sum - self.weights[tuple(item_index)][var_idx], choice_phase_dummies[-1], temp_assignment))
                    else:
                        return_values.append(self.subsetSum_conditionals(
                            n-1, sum - self.weights[tuple(item_index)][var_idx], choice_phase_dummies[-1], temp_assignment))

                if(selective_run):
                    assert len(return_values) <= 2

            # Compute max
            result_max = -100
            for i in range(len(return_values)):
                if(result_max < return_values[i][0]):
                    result_max = return_values[i][0]
                    if(len(choice_phase_dummies[i]) == self.choiceVarCount):
                        self.choice_assignment_max[tuple(
                            choice_phase_dummies[i].items())] = result_max
            # Compute min
            result_min = 100
            for i in range(len(return_values)):
                if(result_min > return_values[i][1]):
                    result_min = return_values[i][1]
                    if(len(choice_phase_dummies[i]) == self.choiceVarCount):
                        self.choice_assignment_min[tuple(
                            choice_phase_dummies[i].items())] = result_min

            # Store in memory
            if(tuple(item_index) in self._effective_choice_var):
                # print("Storing", key, "in lookup table")
                self.lookup[key] = (result_max, result_min)
                pass
        else:

            all_variables_assignment_include = all_variables_assignment.copy()
            all_variables_assignment_exclude = all_variables_assignment.copy()
            if(item_index in self.topological_sort):
                all_variables_assignment_include[item_index] = 1
                all_variables_assignment_exclude[item_index] = 0

            if(item_index in self._effective_chance_var):

                key = (n, sum)
                if(key in self.lookup):
                    # print("Hit", key)
                    # print("collision", len(self.resolve_order) - n + 1, sum)
                    return self.lookup[key]

                # When Bayesian network is already resolved, there is no point in enumeration. We can directly use dynamic programming
                # By this time, all network variables are called
                """
                    Base method subsetsum is called
                """
                include = self.subsetSum(
                    n - 1, sum - self.weights[item_index], choice_phase)
                exclude = self.subsetSum(n - 1, sum, choice_phase)
                include_max = include
                exclude_max = exclude
                include_min = include
                exclude_min = exclude
            else:
                include_max, include_min = self.subsetSum_conditionals(
                    n - 1, sum - self.weights[item_index], choice_phase, all_variables_assignment_include)
                exclude_max, exclude_min = self.subsetSum_conditionals(
                    n - 1, sum, choice_phase, all_variables_assignment_exclude)

            # Compute conditionals probabilities
            if(item_index in self.inner_choice):

                val_include = 1
                val_exclude = 1

                # consider conditional relation where child is current variable

                # if(item_index in self.conditional_probs):

                relation = self.conditional_relations[item_index]
                # print(item_index, all_variables_assignment_include)
                val_include *= self.converted_conditional_probs.get(tuple(
                    [index if all_variables_assignment_include[index] == 1 else -1 * index for index in relation]), 1)
                val_exclude *= self.converted_conditional_probs.get(tuple(
                    [index if all_variables_assignment_exclude[index] == 1 else -1 * index for index in relation]), 1)

                result_max = include_max * val_include + exclude_max * val_exclude
                # print(val_include, val_exclude, result_max, tuple([index if all_variables_assignment_include[index] == 1 else -1 * index for index in relation]))
                result_min = include_min * val_include + exclude_min * val_exclude

            # Compute independent probabilities
            elif(item_index in self.chance):
                result_max = (self.probs[item_index] * include_max) + \
                    (1 - self.probs[item_index]) * exclude_max
                result_min = (self.probs[item_index] * include_min) + \
                    (1 - self.probs[item_index]) * exclude_min

                if(item_index in self._effective_chance_var):
                    self.lookup[key] = (result_max, result_min)

            else:
                raise ValueError(item_index)

        assert result_max is not None
        assert result_min is not None

        return result_max, result_min

    def optimal_assignment(self, choice_assignment, optimal_val):
        # print("Hi")
        # print(self.choice_assignment)
        for key in choice_assignment:
            if(choice_assignment[key] == optimal_val):
                result = []
                for k, v in key:
                    if(v == 1):
                        result.append(k)
                    else:
                        result.append(-k)
                return result

                # return [k if (not self.convertedIndex[k - 1]) else -1 * k for (k,v) in key]


# subsetsumcount = SubSetSumCount(weights = {1: 1, 2: 1, 3: 1, 4:-1},
#                                 probs = {2:0.4,  3: 0.5, 4:0.3},
#                                 sum = 2,
#                                 choice_variables=[[1]],
#                                 # conditional_probs={
#                                 #         ((1), 2) : 0.6,
#                                 #         ((-1), 2) : 0.3,
#                                 #                     },
#                                 verbose=True)
# subsetsumcount.compute()
"""
subsetsumcount = SubSetSumCount(weights = {1: 1, 2: 1, 3: 1, 4:1},
                                probs = {2:0.4,  3: 0.5, 4:0.3}, 
                                sum = 1, 
                                choice_variables=[[1]], 
                                # conditional_probs={
                                #         ((1), 2) : 0.6,
                                #         ((-1), 2) : 0.3,
                                #                     }, 
                                verbose=True)
subsetsumcount.compute()
"""
