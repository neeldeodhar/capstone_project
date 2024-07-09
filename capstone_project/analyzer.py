#IMPORTING LIBRARIES
import os
import sys
import pandas as pd
import seaborn as sns
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def read_dataset(csv_file_path: str) -> pd.DataFrame:
    dataset = pd.read_csv(filepath_or_buffer = csv_file_path)
    
    return dataset

def describe(dataset: pd.DataFrame) ->str:
    print(dataset.describe())
    return dataset.describe()
#CREATION of Data manipulation class (pre-processing)
class DataManipulation():
    def __init__(self, df:pd.DataFrame):
   
       self.df = df

    def drop_column(self, column_name: str):
        data_dfdf = self.df.copy()
        
        self.df = data_dfdf.drop(columns=column_name)
        return self.df

    def encode_features(self, column_names):
        data_dfdf = self.df.copy()
        label_encoder = LabelEncoder()
       
        for col in column_names:
          data_dfdf[col] = label_encoder.fit_transform(data_dfdf[[col]])
        self.df= data_dfdf
        return self.df

    def encode_label(self, column_name: str):
        data_dfdf = self.df.copy()
       
        dummies = pd.get_dummies(data_dfdf[column_name])
        dumcat = pd.concat([data_dfdf, dummies], axis =1)
       
        self.df= dumcat
        return self.df
    def shuffle(self): 
        data_dfdf = shuffle(self.df)
        self.df = data_dfdf
       
        return self.df
    def min_max_Scaler(self, column_names):
        data_dfdf = self.df.copy()
        selected_features = self.df[column_names]
        scaler = MinMaxScaler()
        scaler.fit(selected_features)
        scaler.transform(selected_features)
        self.df= data_dfdf
        return self.df
    def z_Score(self, column_names):
        data_dfdf = self.df.copy()
        selected_features = self.df[column_names]
        scaler = StandardScaler()
        scaler.fit(selected_features)
        scaler.transform(selected_features)
        
        mean = np.mean(selected_features)
        std_dev = np.std(selected_features)
        z_scores = (selected_features - mean) / std_dev
        self.df= data_dfdf
        return self.df
    def sample(self , df:pd.DataFrame , factor)-> pd.DataFrame:
        data_dfdf = self.df.copy()
        df = df.sample(frac=factor, replace= True, random_state=1)
        self.df= data_dfdf
        return self.df
 #creating a visualization class 
class visualization():
    def __init__(self, df:pd.DataFrame):
        self.df = df
    def plot_pairPlot(self, column_names):
       
        selected_features = self.df[column_names]
        plt.figure(figsize = (12,5))
        sns.pairplot(selected_features)
        plt.show()
    def plot_correlationMatrix(self, column_names):
        selected_features = self.df[column_names]
       
        plt.figure(figsize =(7,7))
        sns.heatmap(selected_features.corr(), annot = True)
       
        plt.show()
#CODE for boxplots
    def plot_boxPlot(self, column_names):
        selected_features = self.df[column_names]
        plt.figure(figsize = (12,5))
        sns.boxplot(data = selected_features)
        plt.show()
#CODE for numeric histogram
    def plot_histograms_numeric(self,column_names):
        data_dfdf = self.df.copy()
              
        
        selected_features = self.df[column_names]
       
        data_dfdf['carat'] = pd.cut(data_dfdf['carat'], bins= [50 ,100, 150, 200, 250], labels = ['low', 'medium','high','very high'])
       
        data_dfdf['color'] = pd.cut(data_dfdf['color'], bins= [1 ,2, 3, 4, 5], labels = ['low', 'medium','high','very high'])
        data_dfdf['clarity'] = pd.cut(data_dfdf['clarity'], bins= [1 ,2, 3, 4, 5], labels = ['low', 'medium','high','very high'])
        data_dfdf['depth'] = pd.cut(data_dfdf['depth'], bins= [50 ,55, 60, 65, 70], labels = ['low', 'medium','high','very high'])
        
        
        data_dfdf['table'] = pd.cut(data_dfdf['table'], bins= [50 ,55, 60, 65, 70], labels = ['low', 'medium','high','very high'])
        labels = ['low', 'medium','high','very high']
        plt.hist(selected_features)
        plt.xlabel([labels])
        plt.ylabel("count")
        plt.title("Numeric Histogram")
        plt.show()
#code for categorical histograms
    def plot_histograms_categorical(self,column_names):
        selected_features = self.df[column_names]
       
        plt.hist(selected_features)
        plt.xlabel([column_names])
        plt.ylabel("count")
        plt.title("categorical Histogram")
        plt.show()

if __name__ =="__main__":
    
    absolute_path = "C:/Users/ideod/OneDrive/Documents/new folder zip data/diamonds.csv"
    dfdf = read_dataset(csv_file_path=absolute_path)
#CREATING original model for pre-processing
    original = DataManipulation(dfdf)
  
    dfdf = original.drop_column("Unnamed: 0")
    dfdf = original.sample(dfdf, 0.5)
    dfdf = original.shuffle()
    dfdf = original.encode_features(['carat','color','clarity'])
    
    dfdf = original.encode_label("cut")
    dfdf = original.min_max_Scaler(['carat','color','clarity'])
    dfdf = original.z_Score(['carat','color','clarity'])
# CREATING visual model for visualization

    visual = visualization(dfdf)
    print(describe)
    print(dfdf)
    visual.plot_pairPlot(['color','clarity', 'cut', 'depth','table'])
  
    visual.plot_correlationMatrix(['carat','depth','table','price','x','y','z'])
    visual.plot_boxPlot(['carat','depth','table'])
    visual.plot_histograms_numeric(['carat'])
    visual.plot_histograms_categorical(['cut'])


    