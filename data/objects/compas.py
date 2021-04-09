import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from justicia import utils

class Compas():

    def __init__(self, verbose = True, config = 0):
        self.name = "compas"
        self.filename = os.path.dirname(os.path.realpath(__file__)) + "/../raw/compas-scores-two-years.csv"
        if(config == 0):
            self.known_sensitive_attributes = ['age']
        elif(config == 1):
            self.known_sensitive_attributes = ['race']
        elif(config == 2):
            self.known_sensitive_attributes = ['sex']
        elif(config == 3):
            self.known_sensitive_attributes = ['age', 'race']
        elif(config == 4):
            self.known_sensitive_attributes = ['age', 'sex']
        elif(config == 5):
            self.known_sensitive_attributes = ['sex', 'race']
        elif(config == 6):
            self.known_sensitive_attributes = ['race', 'age', 'sex']    
        else:
            raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")
        self.config = config
        
        self.categorical_attributes = [ 'sex', 'race', 'c_charge_degree', 'two_year_recid']
        self.continuous_attributes = ['juv_fel_count', 'age', 'juv_misd_count', 'juv_other_count', 'priors_count']
        self.keep_columns = ['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'two_year_recid']
        self.verbose = verbose
        self.mediator_attributes = ['priors_count']
        

    def get_df(self, repaired = False):

        df = pd.read_csv(self.filename)
        df = df[self.keep_columns]
        
        
        
        

        assert len(self.categorical_attributes) + len(self.continuous_attributes) == len(df.columns), "Error in classifying columns:" + str(len(self.categorical_attributes) + len(self.continuous_attributes)) + " " + str(len(df.columns))

        for known_sensitive_attribute in self.known_sensitive_attributes:
            if(known_sensitive_attribute in self.continuous_attributes):
                df = utils.get_discretized_df(df, columns_to_discretize=[known_sensitive_attribute])
                df = utils.get_one_hot_encoded_df(df, [known_sensitive_attribute])
                self.continuous_attributes.remove(known_sensitive_attribute)


        # scale 
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])

        df.rename(columns={'two_year_recid': 'target'}, inplace=True)
        self.keep_columns.remove('two_year_recid')
        self.keep_columns.append("target")

        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))
            
        return df
