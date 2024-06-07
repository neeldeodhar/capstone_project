import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def read_dataset(csv_file_path: str) -> pd.DataFrame:
    dataset = pd.read_csv(filepath_or_buffer = csv_file_path)
    
    return dataset

def describe(dataset: pd.DataFrame) ->str:
    print(dataset.describe())
    return dataset.describe()
if __name__ =="__main__":
    absolute_path = "C:/Users/ideod/OneDrive/Documents/new folder zip data/diamonds.csv"
    dfdf = read_dataset(csv_file_path=absolute_path)
    file_description = describe(dataset =dfdf)
    print(describe)
class DataManipulation:
    id = Column(String)
def drop_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()
def drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns = [''], axis = 1)
labelencoder = LabelEncoder()
def encoding(self, data: pd.DataFrame)-> pd.DataFrame:
    return data.labelencoder()


