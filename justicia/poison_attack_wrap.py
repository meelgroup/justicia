import pandas as pd
import numpy as np


class Poison_Model():
    def __init__(self, clf):
        """
        clf is a CClassifierLogistic instance
        """
        self.coef_ = [clf.w.append(0).get_data(
        )]  # including 0 weight for sensitive variable
        self.intercept_ = [clf.b.get_data()[0]]

    def predict(self, X):
        X = np.array(X)
        dot_result = np.dot(X, np.array(self.coef_[0]))
        return (dot_result >= -1 * self.intercept_[0]).astype(int)


def get_dataframe(data):
    """
    data is an np array where the rightmost column denotes sensitive features
    """
    df = pd.DataFrame.from_records(data)
    df.columns = ["feature_" + str(i)
                  for i in range(data.shape[1]-1)] + ['sens']

    return df
