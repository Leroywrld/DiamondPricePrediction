import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import pickle

def build_model():
    ## read the dataset
    def read_df(file_dir, file_name):
        path = os.path.join(file_dir, file_name)
        df = pd.read_csv(path)
        return df

    file_dir = 'C:\\Users\\User\\Desktop\\Data Projects\\Diamonds'
    file_loc = 'diamonds.csv'
    df = read_df(file_dir=file_dir, file_name=file_loc)
    ## drop unwanted column
    df = df[['x', 'carat', 'color', 'clarity', 'price']]
    ## change column types to save on memory
    def tweak_dtypes(df:pd.DataFrame):
        for column in df:
            if df[column].dtypes == 'float64':
                df[column] = df[column].astype('float16')
            elif df[column].dtypes == 'int64':
                df[column] = df[column].astype('int32')
            else:
                df[column] = df[column].astype('string')
        return df
    df = tweak_dtypes(df=df)
    
    ## rename columns where necessary
    def rename_cols(df:pd.DataFrame, name_dict:dict):
        return df.rename(columns=name_dict)
    
    name_dict = {'x':'length'}
    df = rename_cols(df=df, name_dict=name_dict)
    
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    ## order the categories since they're ordinal in nature
    def order_categories(df:pd.DataFrame, category_dict:dict):
        for key, value in category_dict.items():
            df[key] = df[key].astype('category')
            df[key] = df[key].cat.set_categories(value, ordered=True)
        return df
    
    
    cat_dict = {
        
    'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
    'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            
            }
    
    X_train = order_categories(df=X_train, category_dict=cat_dict)
    ## numerical columns
    num_vars = ['length', 'carat']
    ## categorical columns
    cat_vars = ['color', 'clarity']
    
    
    
    full_pipeline = ColumnTransformer([
    ('r_scaler', RobustScaler(), num_vars),
    ('ord_encoder', OrdinalEncoder(), cat_vars)
    ])
    
    ## final transformation on the independent features
    X_train = full_pipeline.fit_transform(X_train)
    ## algorithm of choice
    gb_regressor = ensemble.GradientBoostingRegressor()
    gb_regressor.fit(X=X_train, y=y_train)
    
    filename = 'model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(gb_regressor, file)
    
    print(f"model DONE!!")

build_model()
    

    
    

    