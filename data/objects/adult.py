import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from justicia import utils

class Adult:

    def __init__(self, verbose = True, config = 0):
        self.filename = "data/raw/adult.csv"
        # print("Adult dataset")
        if(config == 0):
            self.known_sensitive_attributes = ['race', 'sex']
        elif(config == 1):
            self.known_sensitive_attributes = ['race']
        elif(config == 2):
            self.known_sensitive_attributes = ['sex']
        elif(config == 3):
            self.known_sensitive_attributes = ['age']
        elif(config == 4):
            self.known_sensitive_attributes = ['race', 'sex', 'age']
        elif(config == 5):
            self.known_sensitive_attributes = ['race', 'age']
        elif(config == 6):
            self.known_sensitive_attributes = ['sex', 'age']
        else:
            raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")
        
        # only a limited number of columns are considered
        self.keep_columns = ['race', 'sex', 'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week','income-per-year'] 
        self.categorical_attributes = [ 'race', 'sex', 'workclass', 'education', 'marital-status', 'occupation', 
                                      'relationship', 'native-country', 'target' ]
        self.continuous_attributes = ['age','capital-loss', 'education-num' ,'capital-gain','hours-per-week' ]
        self.verbose = verbose

        if(verbose):
            print("Sensitive attributes:", self.known_sensitive_attributes)

    def get_df(self, repaired = False):

        df = pd.read_csv(self.filename)
        
        

        # scale 
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(df[self.continuous_attributes])

        
        
        df = df[self.keep_columns]

        for known_sensitive_attribute in self.known_sensitive_attributes:
            if(known_sensitive_attribute in self.continuous_attributes):
                df = utils.get_discretized_df(df, columns_to_discretize=[known_sensitive_attribute])
                df = utils.get_one_hot_encoded_df(df, [known_sensitive_attribute])
                self.continuous_attributes.remove(known_sensitive_attribute)
        
        df['income-per-year'] = df['income-per-year'].map({'<=50K': 0, '>50K': 1})
        df.rename(columns={'income-per-year':'target'}, inplace=True)


        
        
        # df['race'] = df['race'].map({'White' : 'White', 'Black' : 'Others', 'Asian-Pac-Islander' : 'Others', 'Amer-Indian-Eskimo' : 'Others', 'Other' : 'Others'})
        

        df.to_csv("data/raw/reduced_adult.csv", index=False)

        if(repaired):
            df = pd.read_csv("data/raw/repaired_adult.csv")

        
        
        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))
            
        return df