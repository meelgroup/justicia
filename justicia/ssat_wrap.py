
import os
import subprocess
from pysat.formula import CNF
import numpy as np
from pysat.pb import *
import warnings


class Fairness_verifier():

    def __init__(self, timeout=400):
        self.num_variables = 0
        self.num_clauses = 0
        self.timeout = max(int(timeout), 10)
        self.execution_error = False
        pass

    def __str__(self):
        s = "\nFairness verifier ->\n-number of variables: %s\n-number is clauses %s" % (
            self.num_variables, self.num_clauses)
        return s

    def _apply_random_quantification(self, var, prob):
        return "r " + str(prob)+" "+str(var)+" 0\n"

    def _apply_exist_quantification(self, var):
        return "e " + str(abs(var))+" 0\n"

    def invoke_SSAT_solver(self, filename, find_maximization=True, verbose=True):

        # print("\n\n")
        # print(self.formula)
        # print("\n\n")

        dir_path = os.path.dirname(os.path.realpath(filename))

        # Execute and read output
        cmd = "timeout " + str(self.timeout) + " stdbuf -oL " + " abc -c \"ssat -v " + str(dir_path) + "/" + str(filename) + \
            "\" 1>" + str(dir_path) + "/" + str(filename) + "_out.log" + \
            " 2>" + str(dir_path) + "/" + str(filename) + "_err.log"
        os.system(cmd)

        if(verbose):
            f = open(str(dir_path) + "/" + str(filename) + "_err.log", "r")
            lines = f.readlines()
            if(len(lines) > 0):
                print("Error print of SSAT (if any)")
            print(("").join(lines))
            f.close()

        f = open(str(dir_path) + "/" + str(filename) + "_out.log", "r")
        lines = f.readlines()
        f.close()

        # os.system("rm " + str(dir_path) + "/" +str(filename) + "_out")

        # process output
        upper_bound = None
        lower_bound = None
        read_optimal_assignment = False
        self.sol_prob = None
        self.assignment_to_exists_variables = None

        for line in lines:
            if(read_optimal_assignment):
                try:
                    self.assignment_to_exists_variables = list(
                        map(int, line[:-2].strip().split(" ")))
                except:
                    warnings.warn(
                        "Assignment extraction failure: existential variable is probably not in CNF")
                    self.execution_error = True
                if(verbose):
                    print("Learned assignment:",
                          self.assignment_to_exists_variables)
                read_optimal_assignment = False
            if(line.startswith("  > Satisfying probability:")):
                self.sol_prob = float(line.split(" ")[-1])
            if(line.startswith("  > Best upper bound:")):
                upper_bound = float(line.split(" ")[-1])
            if(line.startswith("  > Best lower bound:")):
                lower_bound = float(line.split(" ")[-1])
            if(line.startswith("  > Found an optimizing assignment to exist vars:")):
                read_optimal_assignment = True
                # read optimal assignment in the next iteration

        # When conclusive solution is not found
        if(self.sol_prob is None):
            if(find_maximization):
                try:
                    assert upper_bound is not None
                    self.sol_prob = upper_bound
                except:
                    self.execution_error = True
            else:
                try:
                    assert lower_bound is not None
                    self.sol_prob = lower_bound
                except:
                    self.execution_error = True

        if(not find_maximization):
            self.sol_prob = 1 - self.sol_prob
        if(verbose):
            print("Probability:", self.sol_prob)
            print("\n===================================\n")

        # remove formula file
        # os.system("rm " + str(dir_path) + "/" +str(filename))

        return self.execution_error

    def _construct_clause(self, vars):
        self.num_clauses += 1
        clause = ""
        for var in vars:
            clause += str(var) + " "

        clause += "0\n"
        return clause

    def _construct_header(self):
        return "p cnf " + str(self.num_variables) + " " + str(self.num_clauses) + "\n"

    def encoding_Enum_SSAT(self, classifier, attributes, sensitive_attributes, auxiliary_variables, probs, filename, sensitive_attributes_assignment, dependency_constraints=[], verbose=True):
        # classifier is assumed to be a boolean formula. It is a 2D list
        # attributes is a list of variables. The last attribute is the sensitive attribute
        # probs is a dictionary containing the i.i.d. probabilities of attributes
        # sensitive attribute is assumed to have the next index after attributes

        self.instance = "Enum"

        self.num_variables = np.array(
            attributes + [abs(_var) for _group in sensitive_attributes for _var in _group] + auxiliary_variables).max()
        self.formula = ""

        # random quantification over non-sensitive attributes
        for attribute in attributes:
            self.formula += self._apply_random_quantification(
                attribute, probs[attribute])

        # the sensitive attribute is exist quantified
        for group in sensitive_attributes:
            for var in group:
                self.formula += self._apply_exist_quantification(var)

        # for each sensitive-attribute (one hot vector), at least one group must be True
        self._formula_for_equal_one_constraints = ""
        for _group in sensitive_attributes:
            if(len(_group) > 1):  # when categorical attribute is not Boolean
                equal_one_constraint = PBEnc.equals(
                    lits=_group, weights=[1 for _ in _group],  bound=1, top_id=self.num_variables)
                for clause in equal_one_constraint.clauses:
                    self._formula_for_equal_one_constraints += self._construct_clause(
                        clause)
                auxiliary_variables += [i for i in range(
                    self.num_variables + 1, equal_one_constraint.nv + 1)]
                self.num_variables = max(
                    equal_one_constraint.nv, self.num_variables)

        # other variables (auxiliary) are exist quantified
        for var in auxiliary_variables:
            self.formula += self._apply_exist_quantification(var)

        # clauses for the classifier
        for clause in classifier:
            self.formula += self._construct_clause(clause)

        # clauses for the dependency constraints
        for clause in dependency_constraints:
            self.formula += self._construct_clause(clause)

        # append previous constraints. TODO this is crucial as auxiliary variables are added after deriving the constraint.
        self.formula += self._formula_for_equal_one_constraints

        # specify group to measure fairness
        for _var in sensitive_attributes_assignment:
            self.formula += self._construct_clause([_var])

        # store in a file
        self.formula = self._construct_header() + self.formula[:-1]
        file = open(filename, "w")
        file.write(self.formula)
        file.close()

        # if(verbose):
        #     print("SSAT instance ->")
        #     print(self.formula)

    def encoding_Learn_SSAT(self, classifier, attributes, sensitive_attributes, auxiliary_variables, probs, filename, dependency_constraints=[], ignore_sensitive_attribute=None, verbose=True, find_maximization=True, negate_dependency_CNF=False):
        """  
        Maximization-minimization algorithm
        """

        self.instance = "Learn"

        # TODO here we call twice: 1) get the sensitive feature with maximum favor, 2) get the sensitive feature with minimum favor
        # classifier is assumed to be a boolean formula. It is a 2D list
        # attributes, sensitive_attributes and auxiliary_variables is a list of variables
        # probs is the list of i.i.d. probabilities of attributes that are not sensitive

        self.num_variables = np.array(
            attributes + [abs(_var) for _group in sensitive_attributes for _var in _group] + auxiliary_variables).max()
        self.formula = ""

        # the sensitive attribute is exist quantified
        for group in sensitive_attributes:
            if(ignore_sensitive_attribute == None or group != ignore_sensitive_attribute):
                for var in group:
                    self.formula += self._apply_exist_quantification(var)

        # random quantification over non-sensitive attributes
        if(len(attributes) > 0):
            for attribute in attributes:
                self.formula += self._apply_random_quantification(
                    attribute, probs[attribute])
        else:
            # dummy random variables
            self.formula += self._apply_random_quantification(
                self.num_variables, 0.5)
            self.num_variables += 1

        if(ignore_sensitive_attribute != None):
            for var in ignore_sensitive_attribute:
                self.formula += self._apply_exist_quantification(var)

        # Negate the classifier
        if(not find_maximization):
            _classifier = CNF(from_clauses=classifier)
            _negated_classifier = _classifier.negate(topv=self.num_variables)
            classifier = _negated_classifier.clauses
            # print(auxiliary_variables, self.num_variables, _negated_classifier.auxvars)
            auxiliary_variables += [i for i in range(
                self.num_variables + 1, _negated_classifier.nv + 1)]
            # print(auxiliary_variables)
            self.num_variables = max(
                _negated_classifier.nv, self.num_variables)

        """
        The following should not work as negation of dependency constraints does not make sense
        """
        # # When dependency CNF is provided, we also need to negate it to learn the least favored group
        # if(negate_dependency_CNF and len(dependency_constraints) > 0):
        #     """
        #     (not a) And b is provided where a = classifier and b = dependency constraints.
        #     Additionally, (not a) is in CNF.
        #     Our goal is to construct (not (a And b)) <-> (not a) Or (not b) <-> (x Or y) And (x -> not a) And (y -> not b).
        #     In this encoding, x and y are two introduced variables.
        #     """

        #     _dependency_CNF = CNF(from_clauses=dependency_constraints)
        #     _negated_dependency_CNF = _dependency_CNF.negate(topv=self.num_variables)

        #     # redefine
        #     auxiliary_variables += [i for i in range(self.num_variables + 1, _negated_dependency_CNF.nv + 3)]
        #     self.num_variables = max(_negated_dependency_CNF.nv, self.num_variables)
        #     dependency_constraints = [clause + [-1 * (self.num_variables + 1)] for clause in _negated_dependency_CNF.clauses]
        #     dependency_constraints += [[self.num_variables + 1, self.num_variables + 2]]
        #     classifier = [clause + [-1 * (self.num_variables + 2)] for clause in classifier]
        #     self.num_variables += 2 # two additional variables are introduced

        # for each sensitive-attribute (one hot vector), at least one group must be True
        self._formula_for_equal_one_constraints = ""
        for _group in sensitive_attributes:
            if(len(_group) > 1):  # when categorical attribute is not Boolean
                equal_one_constraint = PBEnc.equals(
                    lits=_group, weights=[1 for _ in _group],  bound=1, top_id=self.num_variables)
                for clause in equal_one_constraint.clauses:
                    self._formula_for_equal_one_constraints += self._construct_clause(
                        clause)
                auxiliary_variables += [i for i in range(
                    self.num_variables + 1, equal_one_constraint.nv + 1)]
                self.num_variables = max(
                    equal_one_constraint.nv, self.num_variables)

        # other variables (auxiliary) are exist quantified
        for var in auxiliary_variables:
            self.formula += self._apply_exist_quantification(var)

        # clauses for the classifier
        for clause in classifier:
            self.formula += self._construct_clause(clause)

        # clauses for the dependency constraints
        for clause in dependency_constraints:
            self.formula += self._construct_clause(clause)

        self.formula += self._formula_for_equal_one_constraints

        # store in a file
        self.formula = self._construct_header() + self.formula[:-1]
        file = open(filename, "w")
        file.write(self.formula)
        file.close()

        # if(verbose):
        #     print("SSAT instance ->")
        #     print(self.formula)
