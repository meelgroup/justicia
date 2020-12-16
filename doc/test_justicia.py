import sys
filename = "sample.sdimacs"
from data.objects.ricci import Ricci
from data.objects.titanic import Titanic
from data.objects.adult import Adult
from data.objects.german import German
from data.objects.bank import Bank
from data.objects.compas import Compas
from justicia import decision_tree_wrap 
from justicia import linear_classifier_wrap 
from justicia import mlic_wrap
from justicia.metrics import Metric

verbose = False
datasetObj = Titanic(verbose=verbose, config=1)
neg_model = None

for model_name in ['dt']:
    if(model_name == 'lr'):
        model, data_train, data_test,sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(datasetObj, classifier=model_name, repaired=False, verbose=verbose, compute_equalized_odds=True)

    if(model_name == 'svm-linear'):
        model, data_train, data_test,sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(datasetObj, classifier=model_name, repaired=False, verbose=verbose, compute_equalized_odds=True)


    if(model_name == 'dt'):
        model, data_train, data_test,sensitive_attributes, y_train, y_test = decision_tree_wrap.init(datasetObj, repaired=False, verbose=verbose, compute_equalized_odds=True)

    if(model_name == "CNF"):    
        model, neg_model, data_train, data_test,sensitive_attributes, y_train, y_test = mlic_wrap.init(datasetObj, repaired=False, verbose=verbose, negated=False, compute_equalized_odds=True)

    
    print("Sensitive attributes:", sensitive_attributes)  
    for i in range(1):   
        for encoding in ['Enum', 'Enum-correlation', 'Learn-efficient']:
            metric = Metric(model[i], model_name, data_test[i],sensitive_attributes, neg_model =  neg_model, verbose=False, encoding=encoding)
            print(encoding, metric.disparate_impact_ratio, metric.statistical_parity_difference)
            print(metric.most_favored_group, metric.least_favored_group)
            