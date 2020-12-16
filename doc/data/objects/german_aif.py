import aif360.datasets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class German():

    def __init__(self, verbose = True, config = 0):
        if(config == 0):
            self.known_sensitive_attributes = ['age', 'sex']
        elif(config == 1):
            self.known_sensitive_attributes = ['age']
        elif(config == 2):
            self.known_sensitive_attributes = ['sex']
            
        else:
            raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")
        
        # only a limited number of columns are considered
        self.categorical_attributes = [ 'age',  'people_liable_for', 'sex', 'status=A11', 'status=A12', 'status=A13', 
                            'status=A14', 'credit_history=A30', 'credit_history=A31', 'credit_history=A32', 'credit_history=A33', 
                            'credit_history=A34', 'purpose=A40', 'purpose=A41', 'purpose=A410', 'purpose=A42', 'purpose=A43', 
                            'purpose=A44', 'purpose=A45', 'purpose=A46', 'purpose=A48', 'purpose=A49', 'savings=A61', 
                            'savings=A62', 'savings=A63', 'savings=A64', 'savings=A65', 'employment=A71', 
                            'employment=A72', 'employment=A73', 'employment=A74', 'employment=A75', 
                            'other_debtors=A101', 'other_debtors=A102', 'other_debtors=A103', 'property=A121', 
                            'property=A122', 'property=A123', 'property=A124', 'installment_plans=A141', 
                            'installment_plans=A142', 'installment_plans=A143', 'housing=A151', 'housing=A152', 
                            'housing=A153', 'skill_level=A171', 'skill_level=A172', 'skill_level=A173', 'skill_level=A174', 
                            'telephone=A191', 'telephone=A192', 'foreign_worker=A201', 'foreign_worker=A202', 'target' ]
        self.continuous_attributes = ['month', 'credit_amount', 
                                      'investment_as_income_percentage',
                                      'residence_since',
                                      'number_of_credits']
        self.verbose = verbose

    def get_df(self, repaired = False):

        dataset = aif360.datasets.GermanDataset()
        df = pd.DataFrame(data=dataset.features, columns = dataset.feature_names)
        df['target'] = pd.Series(dataset.labels.flatten(), index=df.index) 
        
        

        assert len(self.categorical_attributes) + len(self.continuous_attributes) == len(df.columns), "Error in classifying columns:" + str(len(self.categorical_attributes) + len(self.continuous_attributes)) + " " + str(len(df.columns))


        # scale 
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])

        df['target'] = df['target'].map({1: 1, 2: 0})
        
        

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
