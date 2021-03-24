import pandas as pd


class Poison_Model():
    def __init__(self, clf):
        """
        clf is a CClassifierLogistic instance
        """
        self.coef_ = [clf.w.append(0).get_data()] # including 0 weight for sensitive variable
        self.intercept_ = [clf.b.get_data()[0]]


def get_dataframe(data):
    """
    data is an np array where the rightmost column denotes sensitive features
    """
    df = pd.DataFrame.from_records(data)
    df.columns = ["feature_" + str(i) for i in range(data.shape[1]-1)] + ['sens']

    return df
        
