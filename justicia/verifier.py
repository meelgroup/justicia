
import os
import subprocess
from pysat.formula import CNF
import numpy as np
from pysat.pb import *


class Fairness_verifier():

    def __init__(self, timeout = 400):
        self.num_variables = 0
        self.num_clauses = 0
        self.timeout = timeout
        self.execution_error = False
        pass

    def __str__(self):
        s = "\nFairness verifier ->\n-number of variables: %s\n-number is clauses %s" % (self.num_variables, self.num_clauses)
        return s

    def _apply_random_quantification(self, var, prob):
        return "r " + str(prob)+" "+str(var)+" 0\n"

    def _apply_exist_quantification(self, var):
        return "e " + str(var)+" 0\n"


    def invoke_SSAT_solver(self, filename, find_maximization = True, verbose = True):
        
        dir_path = os.path.dirname(os.path.realpath(filename))
        cmd = "timeout " + str(self.timeout) + " abc -c \"ssat "+ str(dir_path) + "/" +str(filename)+"\""
        try:
            cmd_output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            if(verbose):
                print(e.output)
            cmd_output = e.output
            self.execution_error = True
            
        lines = cmd_output.decode('utf-8').split("\n")

        # process output
        tight_upper_bound = False
        upper_bound = 0
        lower_bound = 0
        read_optimal_assignment = False
        if(verbose):
            print("\n")
        
        for line in lines:
            if(read_optimal_assignment):
                self.assignment_to_exists_variables = list(map(int, line[:-2].strip().split(" ")))
                if(verbose):
                    print("Learned assignment:",self.assignment_to_exists_variables)
                read_optimal_assignment = False
            if(line == "[INFO] Exactly solve the instance: upper bound is tight"):
                tight_upper_bound = True
            if(line.startswith("  > Upper bound =")):
                upper_bound = float(line.split(" ")[-1])
            if(line.startswith("  > Lower bound =")):
                lower_bound = float(line.split(" ")[-1])
            if(line.startswith("  > optimizing assignment to exist vars")):
                read_optimal_assignment = True 
                # read optimal assignment in the next iteration

        if(find_maximization):
            # print("Is upper bound tight? ", tight_upper_bound)
            if(verbose):
                if(self.instance == "Enum"):
                    print("upper bound", upper_bound)
                if(not tight_upper_bound):
                    print("lower bound", lower_bound)
                else:
                    print("upper bound is tight")
            self.upper_bound = upper_bound
            self.lower_bound = lower_bound
        else:
            if(verbose):
                print("upper bound", 1 - lower_bound)
                if(self.instance == "Enum"):
                    print("lower bound", 1 - upper_bound)
            self.upper_bound = 1 - lower_bound
            self.lower_bound = 1 - upper_bound

        if(verbose):
            print("\n===================================\n")

        os.system("rm " + str(dir_path) + "/" +str(filename))
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

    def encoding_Enum_SSAT(self, classifier, attributes, sensitive_attributes, auxiliary_variables, probs, filename, sensitive_attributes_assignment, dependency_constraints = [], verbose=True):
        # classifier is assumed to be a boolean formula. It is a 2D list
        # attributes is a list of variables. The last attribute is the sensitive attribute
        # probs is a dictionary containing the i.i.d. probabilities of attributes
        # sensitive attribute is assumed to have the next index after attributes

        self.instance = "Enum"

        self.num_variables = len(attributes) + len([_var for _group in sensitive_attributes for _var in _group]) + len(auxiliary_variables)
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
            if(len(_group)>1): # when categorical attribute is not Boolean
                equal_one_constraint = PBEnc.equals(lits=_group, weights=[1 for _ in _group],  bound=1, top_id=self.num_variables)
                for clause in equal_one_constraint.clauses:
                    self._formula_for_equal_one_constraints += self._construct_clause(clause)
                auxiliary_variables += [i for i in range(self.num_variables + 1, equal_one_constraint.nv + 1)]
                self.num_variables = max(equal_one_constraint.nv, self.num_variables)

        
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
    
    def encoding_Learn_SSAT(self, classifier, attributes, sensitive_attributes, auxiliary_variables, probs, filename, dependency_constraints = [], ignore_sensitive_attribute = None, verbose=True, find_maximization = True):
        """  
        Maximization-minimization algorithm
        """

        self.instance = "Learn"

        # TODO here we call twice: 1) get the sensitive feature with maximum favor, 2) get the sensitive feature with minimum favor
        # classifier is assumed to be a boolean formula. It is a 2D list
        # attributes, sensitive_attributes and auxiliary_variables is a list of variables
        # probs is the list of i.i.d. probabilities of attributes that are not sensitive

        self.num_variables = len(attributes) + len([_var for _group in sensitive_attributes for _var in _group]) + len(auxiliary_variables)
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

        if(not find_maximization):
            _classifier = CNF(from_clauses=classifier)
            _negated_classifier = _classifier.negate(topv=self.num_variables)
            classifier = _negated_classifier.clauses
            # print(auxiliary_variables, self.num_variables, _negated_classifier.auxvars)
            auxiliary_variables += [i for i in range(self.num_variables + 1, _negated_classifier.nv + 1)]
            # print(auxiliary_variables)
            self.num_variables = max(_negated_classifier.nv, self.num_variables)

        
        # for each sensitive-attribute (one hot vector), at least one group must be True
        self._formula_for_equal_one_constraints = ""
        for _group in sensitive_attributes:
            if(len(_group)>1): # when categorical attribute is not Boolean
                equal_one_constraint = PBEnc.equals(lits=_group, weights=[1 for _ in _group],  bound=1, top_id=self.num_variables)
                for clause in equal_one_constraint.clauses:
                    self._formula_for_equal_one_constraints += self._construct_clause(clause)
                auxiliary_variables += [i for i in range(self.num_variables + 1, equal_one_constraint.nv + 1)]
                self.num_variables = max(equal_one_constraint.nv, self.num_variables)

        

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

        


    
