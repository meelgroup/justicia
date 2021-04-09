import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from justicia import utils
import os

class German():

    def __init__(self, verbose = True, config = 0):
        self.name = "german"
        self.filename = os.path.dirname(os.path.realpath(__file__)) + "/../raw/german_credit_data.csv"
        
        if(config == 0):
            self.known_sensitive_attributes = ['Age', 'Sex']
        elif(config == 1):
            self.known_sensitive_attributes = ['Age']
        elif(config == 2):
            self.known_sensitive_attributes = ['Sex']
            
        else:
            raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")
        self.config = config
        
        # only a limited number of columns are considered
        self.categorical_attributes = [ 'Sex', 'Housing', 'Checking account', 'Saving accounts', 'Purpose', 'target']
        self.continuous_attributes = ['Age', 'Job', 'Credit amount', 'Duration']
        self.verbose = verbose

        self.mediator_attributes = ['Credit amount']

    def get_df(self, repaired = False):

        df = pd.read_csv(self.filename)

        assert len(self.categorical_attributes) + len(self.continuous_attributes) == len(df.columns), "Error in classifying columns"
        self.keep_columns = list(df.columns)    
        for known_sensitive_attribute in self.known_sensitive_attributes:
            if(known_sensitive_attribute in self.continuous_attributes):
                df = utils.get_discretized_df(df, columns_to_discretize=[known_sensitive_attribute])
                df = utils.get_one_hot_encoded_df(df, [known_sensitive_attribute])
                self.continuous_attributes.remove(known_sensitive_attribute)

        # scale 
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])

        df['target'] = df['target'].map({'good': 1, 'bad': 0})
        
        

        # df.to_csv("data/raw/reduced_german.csv", index=False)

        # if(repaired):
        #     df = pd.read_csv("data/raw/repaired_german.csv")

        
        
        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))
            
        return df
