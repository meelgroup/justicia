import sys
from justicia import linear_classifier_wrap
from justicia import decision_tree_wrap
from justicia.metrics import Metric
from justicia import mlic_wrap



sys.path.append("../")
from data.objects.compas import Compas
from data.objects.bank import Bank
from data.objects.german import German
from data.objects.adult import Adult
from data.objects.titanic import Titanic
from data.objects.ricci import Ricci
import argparse



filename = "sample.sdimacs"

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument("--model", nargs='+', default=["lr"], type=str)
parser.add_argument("--encoding", nargs='+', default=['best'])
args = parser.parse_args()


datasetObj = Titanic(verbose=args.verbose, config=0)
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


for model_name in args.model:
    if(model_name == 'lr'):
        model, data_train, data_test, sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(
            datasetObj, classifier=model_name, repaired=False, verbose=False, compute_equalized_odds=True)

    if(model_name == 'svm-linear'):
        model, data_train, data_test, sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(
            datasetObj, classifier=model_name, repaired=False, verbose=False, compute_equalized_odds=True)

    if(model_name == 'dt'):
        model, data_train, data_test, sensitive_attributes, y_train, y_test = decision_tree_wrap.init(
            datasetObj, repaired=False, verbose=False, compute_equalized_odds=True, depth=4)

    if(model_name == "CNF"):
        model, data_train, data_test, sensitive_attributes, y_train, y_test = mlic_wrap.init(
            datasetObj, repaired=False, verbose=False, compute_equalized_odds=True)

    print("Sensitive attributes:", sensitive_attributes)
    for i in range(1):

        for encoding in args.encoding:
            metric = Metric(model[i], data_test[i], sensitive_attributes, mediator_attributes=[
            ], major_group={}, verbose=args.verbose, encoding=encoding).compute()
            print(metric)
            print()

            # try:
            #     justicia.utils.draw_dependency(metric).show()
            # except Exception as e:
            #     print(e)

            # print()

            # metric = Metric(model[i], model_name, data_test[i], sensitive_attributes,  mediator_attributes=datasetObj.mediator_attributes, major_group=metric.most_favored_group, neg_model =  neg_model, verbose=args.verbose, encoding=encoding).compute()
            # print("Path-specific causal fairness:")
            # print(encoding, metric.disparate_impact_ratio, metric.statistical_parity_difference, metric.time_taken)
            # print("Most favored: ", metric.most_favored_group)
            # print("Least favored:", metric.least_favored_group)
