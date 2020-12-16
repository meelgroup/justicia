import sys
sys.path.append("..")
from data.objects.communities import Communities_and_Crimes

verbose = True
datasetObj = Communities_and_Crimes(verbose=verbose, config=1)

print(datasetObj.get_df())



