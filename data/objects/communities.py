import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from justicia import utils
import numpy as np


class Communities_and_Crimes():

    def __init__(self, verbose=True, config=0):
        self.filename = "data/raw/communities.data"
        # if(config == 0):
        #     self.known_sensitive_attributes = ['race', 'sex']
        # elif(config == 1):
        #     self.known_sensitive_attributes = ['race']
        # elif(config == 2):
        #     self.known_sensitive_attributes = ['sex']
        # elif(config == 3):
        #     self.known_sensitive_attributes = ['age']
        # elif(config == 4):
        #     self.known_sensitive_attributes = ['race', 'sex', 'age']
        # elif(config == 5):
        #     self.known_sensitive_attributes = ['race', 'age']
        # elif(config == 6):
        #     self.known_sensitive_attributes = ['sex', 'age']
        # else:
        #     raise ValueError(str(config)+ " is not a valid configuration for sensitive groups")

        
        self._columns = ["state",
                         "county",
                         "community",
                         "communityname",
                         "fold",
                         "population",
                         "householdsize",
                         "racepctblack",
                         "racePctWhite",
                         "racePctAsian",
                         "racePctHisp",
                         "agePct12t21",
                         "agePct12t29",
                         "agePct16t24",
                         "agePct65up",
                         "numbUrban",
                         "pctUrban",
                         "medIncome",
                         "pctWWage",
                         "pctWFarmSelf",
                         "pctWInvInc",
                         "pctWSocSec",
                         "pctWPubAsst",
                         "pctWRetire",
                         "medFamInc",
                         "perCapInc",
                         "whitePerCap",
                         "blackPerCap",
                         "indianPerCap",
                         "AsianPerCap",
                         "OtherPerCap",
                         "HispPerCap",
                         "NumUnderPov",
                         "PctPopUnderPov",
                         "PctLess9thGrade",
                         "PctNotHSGrad",
                         "PctBSorMore",
                         "PctUnemployed",
                         "PctEmploy",
                         "PctEmplManu",
                         "PctEmplProfServ",
                         "PctOccupManu",
                         "PctOccupMgmtProf",
                         "MalePctDivorce",
                         "MalePctNevMarr",
                         "FemalePctDiv",
                         "TotalPctDiv",
                         "PersPerFam",
                         "PctFam2Par",
                         "PctKids2Par",
                         "PctYoungKids2Par",
                         "PctTeen2Par",
                         "PctWorkMomYoungKids",
                         "PctWorkMom",
                         "NumIlleg",
                         "PctIlleg",
                         "NumImmig",
                         "PctImmigRecent",
                         "PctImmigRec5",
                         "PctImmigRec8",
                         "PctImmigRec10",
                         "PctRecentImmig",
                         "PctRecImmig5",
                         "PctRecImmig8",
                         "PctRecImmig10",
                         "PctSpeakEnglOnly",
                         "PctNotSpeakEnglWell",
                         "PctLargHouseFam",
                         "PctLargHouseOccup",
                         "PersPerOccupHous",
                         "PersPerOwnOccHous",
                         "PersPerRentOccHous",
                         "PctPersOwnOccup",
                         "PctPersDenseHous",
                         "PctHousLess3BR",
                         "MedNumBR",
                         "HousVacant",
                         "PctHousOccup",
                         "PctHousOwnOcc",
                         "PctVacantBoarded",
                         "PctVacMore6Mos",
                         "MedYrHousBuilt",
                         "PctHousNoPhone",
                         "PctWOFullPlumb",
                         "OwnOccLowQuart",
                         "OwnOccMedVal",
                         "OwnOccHiQuart",
                         "RentLowQ",
                         "RentMedian",
                         "RentHighQ",
                         "MedRent",
                         "MedRentPctHousInc",
                         "MedOwnCostPctInc",
                         "MedOwnCostPctIncNoMtg",
                         "NumInShelters",
                         "NumStreet",
                         "PctForeignBorn",
                         "PctBornSameState",
                         "PctSameHouse85",
                         "PctSameCity85",
                         "PctSameState85",
                         "LemasSwornFT",
                         "LemasSwFTPerPop",
                         "LemasSwFTFieldOps",
                         "LemasSwFTFieldPerPop",
                         "LemasTotalReq",
                         "LemasTotReqPerPop",
                         "PolicReqPerOffic",
                         "PolicPerPop",
                         "RacialMatchCommPol",
                         "PctPolicWhite",
                         "PctPolicBlack",
                         "PctPolicHisp",
                         "PctPolicAsian",
                         "PctPolicMinor",
                         "OfficAssgnDrugUnits",
                         "NumKindsDrugsSeiz",
                         "PolicAveOTWorked",
                         "LandArea",
                         "PopDens",
                         "PctUsePubTrans",
                         "PolicCars",
                         "PolicOperBudg",
                         "LemasPctPolicOnPatr",
                         "LemasGangUnitDeploy",
                         "LemasPctOfficDrugUn",
                         "PolicBudgPerPop",
                         "ViolentCrimesPerPop"
                         ]

        self.ignore_columns = ['state', 'county', 'community', 'communityname', 'fold']
        self.target = 'ViolentCrimesPerPop'

        # sensitive information
        self._race_columns = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctAsian'] 
        self._age_columns = ["agePct12t21", "agePct12t29", "agePct16t24", "agePct65up"]
        self._employment_columns = ["pctWWage", "pctWFarmSelf",
                                    "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire"]
        self._marital_columns = ['MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv']
        self._immegration_columns = ['PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10']
        self._language_columns = ['PctSpeakEnglOnly', 'PctNotSpeakEnglWell']
        self._police_race_columns = ["PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor"] 
        

        self.categorical_attributes = [self.target] + \
                                        self._race_columns + \
                                        self._age_columns + \
                                        self._employment_columns + \
                                        self._marital_columns + \
                                        self._immegration_columns + \
                                        self._language_columns + \
                                        self._police_race_columns

        self.continuous_attributes = [column for column in self._columns if column not in self.categorical_attributes and column not in self.ignore_columns]
        self.verbose = verbose

    def get_df(self, repaired=False):

        df = pd.read_csv(self.filename, header=None)
        df.columns = self._columns
        df.drop(self.ignore_columns, axis=1, inplace=True)

        # Sensitive varaibles are continuous, but categorical values are expected
        # We set the dominating variable to 1 and rest of the variables to 0 for
        # related senstive variables set.

        columns_to_del = []
        # race
        df['max_race'] = df[self._race_columns].max(axis=1)
        columns_to_del.append('max_race')
        for column in self._race_columns:
            df[column] = df.apply(lambda x: 1 if x[column] == x['max_race'] else 0, axis=1)

        # age
        df['max_age'] = df[self._age_columns].max(axis=1)
        columns_to_del.append('max_age')
        for column in self._age_columns:
            df[column] = df.apply(lambda x: 1 if x[column] == x['max_age'] else 0, axis=1)
        
        # employment
        df['max_employment'] = df[self._employment_columns].max(axis=1)
        columns_to_del.append('max_employment')
        for column in self._employment_columns:
            df[column] = df.apply(lambda x: 1 if x[column] == x['max_employment'] else 0, axis=1)
        
        # marital
        df['max_marital'] = df[self._marital_columns].max(axis=1)
        columns_to_del.append('max_marital')
        for column in self._marital_columns:
            df[column] = df.apply(lambda x: 1 if x[column] == x['max_marital'] else 0, axis=1)
        

        # immegration (consider within immegration population, may be controversal)
        df['max_immegration'] = df[self._immegration_columns].max(axis=1)
        columns_to_del.append('max_immegration')
        for column in self._immegration_columns:
            df[column] = df.apply(lambda x: 1 if x[column] == x['max_immegration'] else 0, axis=1)


        # language
        df['max_language'] = df[self._language_columns].max(axis=1)
        columns_to_del.append('max_language')
        for column in self._language_columns:
            df[column] = df.apply(lambda x: 1 if x[column] == x['max_language'] else 0, axis=1)

        # Police race
        df['max_police_race'] = df[self._police_race_columns].max(axis=1)
        columns_to_del.append('max_police_race')
        for column in self._police_race_columns:
            df[column] = df.apply(lambda x: 1 if x[column] == x['max_police_race'] else 0, axis=1)

        df.drop(columns_to_del, axis=1, inplace=True)

        # scale (not necessary as mentioned in UCI, data already scaled)
        # scaler = MinMaxScaler()
        # df[self.continuous_attributes] = scaler.fit_transform(
        #     df[self.continuous_attributes])

        
        target_threshold = 0.25
        df[self.target] = df.apply(lambda x: 1 if x[self.target] >= target_threshold else 0, axis=1)
        df.rename(columns={self.target: 'target'}, inplace=True)
        self.target = 'target'


        df.to_csv("data/raw/processed_communities.csv", index=False)

        # if(repaired):
        #     df = pd.read_csv("data/raw/repaired_adult.csv")

        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        
        # drop rows with null values
        # df = df.replace({'?', np.nan}, inplace=True)
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))

        return df
