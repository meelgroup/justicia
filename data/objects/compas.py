import aif360.datasets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Compas():

    def __init__(self, verbose = True, config = 0):
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
            self.known_sensitive_attributes = ['age', 'age', 'sex']    
        else:
            raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")
        
        self.categorical_attributes = [ 'sex', 'race', 'age_25_to_45', 'age_Greater_than_45', 'age_Less_than_25', 'c_charge_degree_F', 'c_charge_degree_M', 'target']
        self.continuous_attributes = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        self.verbose = verbose
        

    def get_df(self, repaired = False):

        dataset = aif360.datasets.CompasDataset()
        df = pd.DataFrame(data=dataset.features, columns = dataset.feature_names)
        df['target'] = pd.Series(dataset.labels.flatten(), index=df.index) 
        
        # remove columns: c_charge_degree and age
        columns = [column for column in df.columns if not (column == 'age' or column.startswith('c_charge_desc'))]
        df = df[columns]
        
        # rename columns 
        columns = [column.replace(" ", "_").replace("=", "_").replace("-", "_to_").replace("_cat", "") for column in columns]
        df.columns = columns
        
        
        # for column in df.columns:
        #     if(len(df[column].unique()) == 2):
        #         print(column, df[column].unique())        


        assert len(self.categorical_attributes) + len(self.continuous_attributes) == len(df.columns), "Error in classifying columns:" + str(len(self.categorical_attributes) + len(self.continuous_attributes)) + " " + str(len(df.columns))

        # scale 
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])

    
        
        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))
            
        return df
