
import sys
sys.path.append("../")


from justicia import decision_tree_wrap 
from justicia import linear_classifier_wrap 
# from justicia import mlic_wrap
from justicia.metrics import Metric

import os


import argparse
from data.objects.ricci import Ricci
from data.objects.titanic import Titanic
from data.objects.adult import Adult
from data.objects.german import German
from data.objects.bank import Bank
from data.objects.compas import Compas
from data.objects.communities import Communities_and_Crimes


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument("--dataset", type=str, default="titanic", choices=['titanic', 'adult', 'ricci', 'german','bank', 'compas', 'communities'])
parser.add_argument("--config", type=int, default=0)
parser.add_argument("--model", default="dt", type=str, choices=['dt'])
parser.add_argument("--encoding", nargs='+', default=['Learn'])
args = parser.parse_args()



verbose = args.verbose

if(args.dataset == "titanic"):
    datasetObj = Titanic(verbose=verbose,config=args.config)
elif(args.dataset == "compas"):
    datasetObj = Compas(verbose=verbose,config=args.config)
elif(args.dataset == "ricci"):
    datasetObj = Ricci(verbose=verbose,config=args.config)
elif(args.dataset == "adult"):
    datasetObj = Adult(verbose=verbose,config=args.config)
elif(args.dataset == "german"):
    datasetObj = German(verbose=verbose,config=args.config)
elif(args.dataset == "bank"):
    datasetObj = Bank(verbose=verbose,config=args.config)
elif(args.dataset == "communities"):
    datasetObj = Communities_and_Crimes(verbose=verbose,config=args.config)
    
else:
    raise ValueError(args.dataset + " is not a defined dataset")

neg_model = None


# structure of encoding
"""
without dependency

    -- Enum (Naive enumeration on all compound groups)
    -- Learn (Learning most and least favored compound group) [Discarded]
    -- Learn-efficient (Similar to Learn, but directly use \neg \phi_Y instead of complementing the CNF)

correlation between non-sensitive and compound sensitive groups

    -- Enum-correlation
    -- Learn-efficient-correlation

with dependency

    -- Enum-dependency
    -- Learn-dependency
    -- Learn-efficient-dependency

"""


model_name = args.model

filename = args.dataset + "_" + model_name +  "_sample.sdimacs"

if(model_name == 'lr'):
    model, data_train, data_test,sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(datasetObj, classifier=model_name, repaired=False, verbose=False, compute_equalized_odds=True)

if(model_name == 'svm-linear'):
    model, data_train, data_test,sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(datasetObj, classifier=model_name, repaired=False, verbose=False, compute_equalized_odds=True)

if(model_name == 'dt'):
    model, data_train, data_test,sensitive_attributes, y_train, y_test = decision_tree_wrap.init(datasetObj, repaired=False, verbose=False, compute_equalized_odds=True, depth=4)

# if(model_name == "CNF"):    
#     model, data_train, data_test,sensitive_attributes, y_train, y_test = mlic_wrap.init(datasetObj, repaired=False, verbose=False, compute_equalized_odds=True)


print("Sensitive attributes:", sensitive_attributes)  
for i in range(1):   
    

    
    for encoding in args.encoding:
        metric = Metric(model[i], data_test[i], sensitive_attributes, mediator_attributes=[], major_group={}, verbose=args.verbose, encoding=encoding, filename=filename).compute()
        print(metric)
        print()


os.system("rm " + filename + "*")

        