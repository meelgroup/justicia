from justicia import utils
from justicia.linear_classifier_wrap import linear_classifier_wrap_Wrapper
import itertools
from justicia.verifier import Fairness_verifier
from justicia.decision_tree_wrap import dtWrapper
import numpy as np
import time 
from itertools import chain

class Metric():

    def __init__(self, model, model_name, data, sensitive_attributes, neg_model=None, encoding="Enum", filename="sample.sdimacs", verbose=True, feature_threshold = 0, linear_continuous_data =  True, correlation = False, timeout=900):
        self.encoding = encoding
        self.model = model
        self.model_name = model_name
        self.data = data
        self.given_sensitive_attributes = sensitive_attributes
        self._filename = filename
        self._neg_model = neg_model
        self.verbose = verbose
        self.meta_sensitive_groups = None
        self._encode_correlation = correlation
        self._correlation_constraints = []
        self.timeout = timeout            

        
        # preprocessing
        start_time = time.time()
        self._get_required_params(feature_threshold, linear_continuous_data)
        self._compute()
        self.time_taken = time.time() - start_time
        
        
    
    def _get_required_params(self, feature_threshold, linear_continuous_data):
        
        
        if(self.model_name == "lr" or self.model_name == "svm-linear"):

            self._given_weights = self.model.coef_[0]
            
            # when data contains continuous features, call specilized statistics
            if(linear_continuous_data):
                self._attributes, self._sensitive_attributes, self._probs, self._given_weights, self._variable_attribute_map, self.column_info = utils.get_statistics_for_linear_classifier_wrap(self.data, self._given_weights, self.given_sensitive_attributes, verbose=self.verbose)
            else:
                self._attributes, self._sensitive_attributes, self._probs, self._variable_attribute_map, self.column_info = utils.get_statistics_from_df(self.data, self.given_sensitive_attributes)
            
            # consider important features only
            if(feature_threshold is not None):
                feature_importance = abs(self._given_weights)
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                threshold = np.percentile(np.array(feature_importance), float(feature_threshold))
                self._weights = []
                for i in range(len(self._given_weights)):
                    if(feature_importance[i] < threshold):
                        self._weights.append(0)
                    else:
                        self._weights.append(self._given_weights[i])
            else:
                self._weights =  self._given_weights  

            self._bias = self.model.intercept_[0]
            

            
                

            
            # get classifier
            lr = linear_classifier_wrap_Wrapper(weights=self._weights, attributes = self._attributes, sensitive_attributes = self._sensitive_attributes, bias=self._bias, verbose=self.verbose)
            self._num_attributes = lr.num_attributes
            self._attributes = lr.attributes
            self._classifier = lr.classifier
            self._sensitive_attributes = lr.sensitive_attributes
            self._auxiliary_variables = lr.auxiliary_variables

            
            if(self.verbose):
                print("Total number of variables in the formula:", self._num_attributes)
                print("Attribute variables:", self._attributes)
                print("Auxiliary variables:", self._auxiliary_variables)
                print("sensitive feature: ", self._sensitive_attributes)
                if(self.encoding != "Enum-correlation"):
                    print("\nprobabilities:", self._probs)
                
            if("efficient" in self.encoding):
                lr_neg = linear_classifier_wrap_Wrapper(weights=self._weights, attributes = self._attributes, sensitive_attributes = self._sensitive_attributes, bias=self._bias, negate=True, verbose=self.verbose)
                self._num_attributes_neg = lr_neg.num_attributes
                self._attributes_neg = lr_neg.attributes
                self._classifier_neg = lr_neg.classifier
                self._sensitive_attributes_neg = lr_neg.sensitive_attributes
                self._auxiliary_variables_neg = lr_neg.auxiliary_variables


        elif(self.model_name == "dt"):

            _sensitive_attributes = utils.get_sensitive_attibutes(self.given_sensitive_attributes, self.data.columns.to_list())
            dt_pos = dtWrapper(self.model,self.data.columns.tolist(), _sensitive_attributes, verbose=self.verbose)
            self._classifier = dt_pos.classifier
            self._num_attributes = dt_pos.num_attributes
            self._attributes = dt_pos.attributes
            self._sensitive_attributes = dt_pos.sensitive_attributes
            self._auxiliary_variables = dt_pos.auxiliary_variables
            self._variable_attribute_map = dt_pos.variable_attribute_map
            self._probs = dt_pos.compute_probability(self.data, verbose=self.verbose)

            if("correlation" in self.encoding):
                self._temp_model = dt_pos
                        
            
            if("efficient" in self.encoding):
                dt_neg = dtWrapper(self.model,self.data.columns.tolist(), _sensitive_attributes, negate=True, verbose=self.verbose)
                self._classifier_neg = dt_neg.classifier
                self._attributes_neg = dt_neg.attributes
                self._num_attributes_neg = dt_neg.num_attributes
                self._sensitive_attributes_neg = dt_neg.sensitive_attributes
                self._auxiliary_variables_neg = dt_neg.auxiliary_variables
            


        elif(self.model_name == "CNF"):
            self._attributes, self._sensitive_attributes, self._probs, self._variable_attribute_map, self.column_info = utils.get_statistics_from_df(self.data, self.given_sensitive_attributes, verbose=self.verbose)
            __classifier = self.model.get_selected_column_index()
            self._classifier = [[] for _ in __classifier]
            for idx in range(len(__classifier)):
                for (_var, _phase) in __classifier[idx]:
                    if(_phase >= 0):
                        self._classifier[idx].append(_var + 1)
                    else:
                        self._classifier[idx].append(-1 * (_var + 1))
            self._auxiliary_variables = []
            self._num_attributes = len(self._attributes + list(chain.from_iterable(self._sensitive_attributes)))

            assert self._num_attributes == len(self.data.columns)


            if(self.encoding == "Learn-efficient"):
                self.encoding = "Learn"

                
                
        else:
            print(self.model_name, "is not defined")
            raise ValueError
        
            

        



    

    

    def _compute(self):

        
        

        self.meta_sensitive_groups = []
        for _group in self._sensitive_attributes:
            if(len(_group)==1):
                _group = [_group[0], -1*_group[0]]
            self.meta_sensitive_groups.append(len(_group))

        min_value = None
        max_value = None
        self.execution_error = False
        self.most_favored_group = None
        self.least_favored_group = None
       
       
        
        
            
        if(self.encoding == "Enum-correlation"):
            
            max_value, min_value = self._run_Enum_correlation()

                 

        # Base encoding   
        elif(self.encoding == "Enum"):
            
            max_value, min_value = self._run_Enum()
        
            
        elif(self.encoding == "Learn-efficient"):
            
            max_value, min_value = self._run_Learn_efficient()
            

        elif(self.encoding == "Learn"):
            
            max_value, min_value = self._run_Learn()
        
        else:
            # print(self.encoding, "is not defined. Try RE-SSAT, ER-SSAT or ER-SSAT-efficient")
            # raise ValueError  
            pass

        if(self.execution_error == True):
            if(self.verbose):
                print("Execution error occured")
            self.statistical_parity_difference = None
            self.disparate_impact_ratio = None

            return

            

        self.statistical_parity_difference = max_value - min_value
        if(max_value ==  min_value):
            self.disparate_impact_ratio = 1
        elif(max_value == 0):
            self.disparate_impact_ratio = float("inf")
        else:
            self.disparate_impact_ratio = float(min_value/max_value)

    def _run_Enum(self, dependency_constraints=[]):
        # prepare for combination
        _sensitive_attributes = []
        for _group in self._sensitive_attributes:
            if(len(_group)==1):
                _group = [_group[0], -1*_group[0]]
            _sensitive_attributes.append(_group)
        
        # For each combination, update answer
        min_value = 1
        max_value = 0
        _combinations = list(itertools.product(*_sensitive_attributes))
        for configuration in _combinations:
            if(self.verbose):
                print("configuration: ", configuration, ":", self._get_group_from_configuration(configuration))
            fv = Fairness_verifier(timeout=float(self.timeout/len(_combinations)))
            fv.encoding_Enum_SSAT(self._classifier,self._attributes,self._sensitive_attributes,self._auxiliary_variables,self._probs, self._filename, dependency_constraints=dependency_constraints, sensitive_attributes_assignment=list(configuration), verbose=self.verbose)
            flag = fv.invoke_SSAT_solver(self._filename, verbose = self.verbose)
            
            
            if(flag == True):
                self.execution_error = True
                break

            if(min_value > fv.upper_bound):
                min_value = fv.upper_bound
                self.least_favored_group = self._get_group_from_configuration(configuration)
            if(max_value < fv.upper_bound):
                max_value = fv.upper_bound
                self.most_favored_group = self._get_group_from_configuration(configuration)

        return max_value, min_value

    def _get_group_from_configuration(self, configuration):
        result = []
        for var in configuration:
            if var > 0:
                if(self.model_name in ["lr", "CNF", "svm-linear"]):
                    result.append(self._variable_attribute_map[var])
                elif(self.model_name == "dt"):
                    (_feature, _threshold) = self._variable_attribute_map[var]
                    if(_threshold == 0.5):
                        result.append("not " + _feature)
                    else:
                        result.append(_feature + " <= " + str(_threshold))
                else:
                    raise ValueError
            else:
                if(self.model_name in ["lr", "CNF", "svm-linear"]):
                    result.append("not " + self._variable_attribute_map[-1 * var])
                elif(self.model_name == "dt"):
                    (_feature, _threshold) = self._variable_attribute_map[-1 * var]
                    if(_threshold == 0.5):
                        result.append(_feature)
                    else:
                        result.append(_feature + " > " + str(_threshold))
                else:
                    raise ValueError    

        return result

    def _run_Learn_efficient(self, dependency_constraints=[]):
        # maximum probability
        fv = Fairness_verifier(timeout=float(self.timeout/2))
        fv.encoding_Learn_SSAT(self._classifier,self._attributes,self._sensitive_attributes,self._auxiliary_variables,self._probs,self._filename, dependency_constraints=dependency_constraints, find_maximization = True, verbose = self.verbose)
        flag = fv.invoke_SSAT_solver(self._filename, find_maximization = True, verbose=self.verbose)
        max_value = fv.lower_bound
        self.most_favored_group = self._get_group(fv)

        if(flag == True):
            self.execution_error = True


        # minimum probability
        fv = Fairness_verifier(timeout=float(self.timeout/2))
        fv.encoding_Learn_SSAT(self._classifier_neg,self._attributes_neg,self._sensitive_attributes_neg,self._auxiliary_variables_neg,self._probs,self._filename, dependency_constraints=dependency_constraints, find_maximization = True, verbose = self.verbose)
        flag = fv.invoke_SSAT_solver(self._filename, find_maximization = False, verbose = self.verbose)
        if(flag == True):
            self.execution_error = True
                
        min_value = fv.upper_bound
        self.least_favored_group = self._get_group(fv)

        return max_value, min_value
    
    def _run_Learn(self, dependency_constraints=[]):
        # maximum probability
        fv = Fairness_verifier(timeout=float(self.timeout/2))
        fv.encoding_Learn_SSAT(self._classifier,self._attributes,self._sensitive_attributes,self._auxiliary_variables,self._probs,self._filename, dependency_constraints=dependency_constraints, find_maximization = True, verbose = self.verbose)
        flag = fv.invoke_SSAT_solver(self._filename, find_maximization = True, verbose=self.verbose)
        if(flag == True):
            self.execution_error = True

        max_value = fv.lower_bound
        self.most_favored_group = self._get_group(fv)


        # maximum probability by complementing the classifier
        fv = Fairness_verifier(timeout=float(self.timeout/2))
        fv.encoding_Learn_SSAT(self._classifier,self._attributes,self._sensitive_attributes,self._auxiliary_variables,self._probs,self._filename,  dependency_constraints=dependency_constraints, find_maximization = False, verbose = self.verbose)
        flag = fv.invoke_SSAT_solver(self._filename, find_maximization = False, verbose=self.verbose)
        if(flag == True):
            self.execution_error = True

        min_value = fv.upper_bound
        self.least_favored_group = self._get_group(fv)

        return max_value, min_value
        
    def _get_group(self, fairness_verifier):
        
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
        

    def _get_df_mask(self, dominating_var, mask):
        # Each dominating var put constraints on the dataframe, which is captured by the mask.
        # When dominating var is negative, we simply calculate for positive var and finally complement it.
        
        negate_mask = False
        if(dominating_var < 0):
            # we need to negate the mask
            negate_mask = True
            dominating_var = -1 * dominating_var

        if(type(self._variable_attribute_map[dominating_var]) == tuple):
            if(self.model_name == "dt"):
                (_feature, _threshold) = self._variable_attribute_map[dominating_var]
                mask = mask & (self.data[_feature] <= _threshold)
            elif(self.model_name == "lr" or self.model_name == "svm-linear"):
                _feature, _, _threshold_1, _comparator_2, _threshold_2 = self._variable_attribute_map[dominating_var]
                if(_comparator_2 == "<"):
                    mask = mask & (self.data[_feature] >= _threshold_1) & (self.data[_feature] < _threshold_2)
                elif(_comparator_2 == "<="):
                    mask = mask & (self.data[_feature] >= _threshold_1) & (self.data[_feature] <= _threshold_2)
                else:
                    raise ValueError(_comparator_2)
            else:
                raise ValueError(self.model_name)
        else:
            mask = mask & (self.data[self._variable_attribute_map[dominating_var]] == 1)       

        if(negate_mask):
            mask = ~mask

        return mask   


    def _parameter_estimation_frequency(self, edges):
        # Dominating vars: list of nodes where a directed edge starts
        dominating_vars = list(set([a for (a,b) in edges]))            
        # dominating_vars should have all combinations
        edge_weights = {}
        for dominating_var in dominating_vars:

            mask = (True)

            if(isinstance(dominating_var, tuple)):
                # multuple parents
                for each_var in dominating_var:
                    mask = self._get_df_mask(each_var, mask)                        
                pass

            
            else:
                mask = self._get_df_mask(dominating_var, mask)
            
                    
            
            if(self.model_name == "dt"):
                marginal_probs = self._temp_model.compute_probability(self.data[mask], verbose=self.verbose)
            elif(self.model_name in ["lr", "CNF", "svm-linear"]):
                marginal_probs = utils.calculate_probs_linear_classifier_wrap(self.data[mask], self.column_info)
            else:
                raise ValueError
            
            for var in marginal_probs:
                edge_weights[(dominating_var,var)] = marginal_probs[var]
        
        return edge_weights
                

    def _run_Enum_correlation(self):
        _sensitive_attributes = []
        for _group in self._sensitive_attributes:
            if(len(_group)==1):
                _group = [_group[0], -1*_group[0]]
            _sensitive_attributes.append(_group)

        
        min_value = 1
        max_value = 0
        _combinations = list(itertools.product(*_sensitive_attributes))
        for configuration in _combinations:
            if(self.verbose):
                print("configuration: ", configuration, ":", self._get_group_from_configuration(configuration))
                # ":", [self._variable_attribute_map[var] if var > 0 else "not " + self._variable_attribute_map[-1 * var] for var in configuration]
            
            # recalculate probs according to conditional probabiities of sensitve attributes
            mask = (True)
            for _attribute in configuration:
                # if(_attribute in self._variable_attribute_map or -1 * _attribute in self._variable_attribute_map):

                #     if(self.model_name == "dt"):
                #         if(_attribute > 0):
                #             (_feature, _threshold) = self._variable_attribute_map[_attribute]
                #         else:
                #             (_feature, _threshold) = self._variable_attribute_map[-1 * _attribute]
                #         mask = mask & (self.data[_feature] <= _threshold)
                #     elif(self.model_name in ["lr", "svm-linear", "CNF"]):
                #         if(_attribute > 0):
                #             mask = mask & (self.data[self._variable_attribute_map[_attribute]] == 1)
                #         else:
                #             mask = mask & (self.data[self._variable_attribute_map[-1 * _attribute]] == 0)                                
                            
                #     else:
                #             raise ValueError()
                # else:
                #     raise ValueError(_attribute)

                mask = self._get_df_mask(_attribute, mask) 
            
                

            if(self.model_name == "dt"):
                self._probs = self._temp_model.compute_probability(self.data[mask], verbose=self.verbose)
            elif(self.model_name in ["lr", "CNF", "svm-linear"]):
                self._probs = utils.calculate_probs_linear_classifier_wrap(self.data[mask], self.column_info)
            else:
                raise ValueError
                
            if(self.verbose):
                print(self._probs)
            

            fv = Fairness_verifier(timeout=float(self.timeout/len(_combinations)))
            fv.encoding_Enum_SSAT(self._classifier,self._attributes,self._sensitive_attributes,self._auxiliary_variables,self._probs, self._filename, sensitive_attributes_assignment=list(configuration), verbose=self.verbose)
            flag = fv.invoke_SSAT_solver(self._filename, verbose = self.verbose)
            
            
            if(flag == True):
                self.execution_error = True
                max_value ==  min_value
                break

            if(min_value > fv.upper_bound):
                min_value = fv.upper_bound
                self.least_favored_group = self._get_group_from_configuration(configuration)
            if(max_value < fv.upper_bound):
                max_value = fv.upper_bound
                self.most_favored_group = self._get_group_from_configuration(configuration)

        return max_value, min_value


