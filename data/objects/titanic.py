import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from justicia import utils
import os

class Titanic:

    def __init__(self, verbose = True, config = 0):
        self.name = "titanic"
        self.filename = os.path.dirname(os.path.realpath(__file__)) + "/../raw/titanic.csv"
        # original columns = D#passenger_class,i#name,D#sex,C#age,D#siblings_or_spouces_aboard,D#parents_or_childred_aboard,i#ticket,C#fare,i#cabin,D#embarked,i#boat,i#body,i#home.dest,cD#survived
        # print("Titanic dataset")
        if(config == 0):
            self.known_sensitive_attributes = ['sex', 'passenger class']
        elif(config == 1):
            self.known_sensitive_attributes = ['passenger class']
        elif(config == 2):
            self.known_sensitive_attributes = ['sex']
        elif(config == 3):
            self.known_sensitive_attributes = ['age']
        elif(config == 4):
            self.known_sensitive_attributes = ['sex', 'age']
        elif(config == 5):
            self.known_sensitive_attributes = ['passenger class', 'age']
        elif(config == 6):
            self.known_sensitive_attributes = ['sex', 'passenger class', 'age']
        else:
            raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")
        self.config = config

        self.mediator_attributes = ['fare']
        
        self.ignore_columns = ['name', 'ticket', 'cabin', 'boat', 'body', 'home destination'] 
        self.categorical_attributes = ['passenger class', 'sex',  'embarked', 'target']
        self.continuous_attributes = ['age', 'fare', 'siblings or spouce aboard', 
                                       'parents or childred aboard',]
        self.verbose = verbose

    def get_df(self, repaired = False):

        
        df = pd.read_csv(self.filename)
        df.columns = ['passenger class', 'name', 'sex',
                    'age', 'siblings or spouce aboard', 
                    'parents or childred aboard', 
                    'ticket', 'fare', 'cabin', 'embarked',
                    'boat', 'body', 'home destination', 'target']

        df = df.drop(self.ignore_columns, axis=1)
        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))
        
        assert len(self.categorical_attributes) + len(self.continuous_attributes) == len(df.columns), str(len(self.categorical_attributes)) + " "  + str(len(self.continuous_attributes))  + " " + str(len(df.columns)) 
        self.keep_columns = list(df.columns)    
        
        
        # scale 
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])

        for known_sensitive_attribute in self.known_sensitive_attributes:
            if(known_sensitive_attribute in self.continuous_attributes):
                df = utils.get_discretized_df(df, columns_to_discretize=[known_sensitive_attribute])
                df = utils.get_one_hot_encoded_df(df, [known_sensitive_attribute])
                self.continuous_attributes.remove(known_sensitive_attribute)
        
        # df['sex'] = df['sex'].map({'female': 0, 'male': 1})
        
        

        df.to_csv(os.path.dirname(os.path.realpath(__file__)) + "/../raw/reduced_titanic.csv", index=False)

        if(repaired):
            df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/../raw/repaired_titanic.csv")

    
        
            
        return df
