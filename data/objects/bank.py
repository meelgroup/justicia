import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from justicia import utils
import os

class Bank():

    def __init__(self, verbose = True, config = 0):
        self.name = "bank"
        self.filename = os.path.dirname(os.path.realpath(__file__)) + "/../raw/bank-additional-full.csv"
        if(config == 0):
            self.known_sensitive_attributes = ['age', 'marital']
        elif(config == 1):
            self.known_sensitive_attributes = ['age']
        elif(config == 2):
            self.known_sensitive_attributes = ['marital']
        else:
            raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")
        self.config = config
        
        
        self.categorical_attributes = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
        self.continuous_attributes = ['age','duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        self.verbose = verbose

    def get_df(self, repaired = False):

        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/../raw/bank-additional-full.csv", sep=";")
        df.rename(columns={'y':'target'}, inplace=True)

        assert len(self.categorical_attributes) + len(self.continuous_attributes) == len(df.columns), "Error in classifying columns:" + str(len(self.categorical_attributes) + len(self.continuous_attributes)) + " " + str(len(df.columns))


        # scale 
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])
        self.keep_columns = list(df.columns)    
        

        for known_sensitive_attribute in self.known_sensitive_attributes:
            if(known_sensitive_attribute in self.continuous_attributes):
                df = utils.get_discretized_df(df, columns_to_discretize=[known_sensitive_attribute])
                df = utils.get_one_hot_encoded_df(df, [known_sensitive_attribute])
                self.continuous_attributes.remove(known_sensitive_attribute)

        
        
        
        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))
        return df
