import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from category_encoders import CountEncoder

def process(train_data, test_data, standardization, logarithmic, count_encoding):
    # columns with NaN-values in train and test set
    train_nan = {col : train_data[col].isna().sum() for col in train_data.columns if train_data[col].isna().sum() > 0}
    # test_nan = {col : test_data[col].isna().sum() for col in test_data.columns if train_data[col].isna().sum() > 0}

    # drop_cols - columns with more than 25 % of data missing
    drop_cols = [col for col in train_nan.keys() if train_nan[col] >= 0.25*len(train_data)]

    train_data = train_data.drop(columns=drop_cols, inplace=False, axis=1)
    test_data = test_data.drop(columns=drop_cols, inplace=False, axis=1)

    y = train_data.pop('SalePrice')
    X = train_data.copy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)
    X_test = test_data.copy()

    total_cols = [col for col in X_train.keys()]
    categorical_cols = [col for col in X_train.keys() if X_train[col].dtype == 'object']
    numerical_cols = list(set(total_cols) - set(categorical_cols))

    # identify categorical colummns with a high number of categories // cause problems for OneHotEncoding
    bad_cat_cols = {col : X_train[col].nunique() for col in categorical_cols if X_train[col].nunique() >= 20} 
    good_cat_cols = list(set(categorical_cols)-set(bad_cat_cols)) # categories used for OneHotEncoding

    categorical_cols = good_cat_cols

    # Handle NaN's
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[numerical_cols]))
    X_val_num = pd.DataFrame(num_imputer.transform(X_val[numerical_cols]))
    X_test_num = pd.DataFrame(num_imputer.transform(X_test[numerical_cols]))
    
    X_train_num.columns = X_train[numerical_cols].columns
    X_val_num.columns = X_val[numerical_cols].columns
    X_test_num.columns = X_test[numerical_cols].columns

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[categorical_cols]))
    X_val_cat = pd.DataFrame(cat_imputer.transform(X_val[categorical_cols]))
    X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[categorical_cols]))
    
    X_train_cat.columns = X_train[categorical_cols].columns
    X_val_cat.columns = X_val[categorical_cols].columns
    X_test_cat.columns = X_test[categorical_cols].columns

    X_train = pd.concat(objs=[X_train_num, X_train_cat], axis=1)
    X_val = pd.concat(objs=[X_val_num, X_val_cat], axis=1)
    X_test = pd.concat(objs=[X_test_num, X_test_cat], axis=1)

    # Feature Creation 
    X_train['Total_bathrooms'] = (X_train['BsmtFullBath']+ 0.5*X_train['BsmtHalfBath']+ X_train['FullBath'] + 0.5*X_train['HalfBath'])
    X_val['Total_bathrooms'] = (X_val['BsmtFullBath']+ 0.5*X_val['BsmtHalfBath']+ X_val['FullBath'] + 0.5*X_val['HalfBath'])
    X_test['Total_bathrooms'] = (X_test['BsmtFullBath']+ 0.5*X_test['BsmtHalfBath']+ X_test['FullBath'] + 0.5*X_test['HalfBath'])

    X_train['hasPool'] = X_train['PoolArea'].apply(lambda x: 1 if x>0 else 0)
    X_val['hasPool'] = X_val['PoolArea'].apply(lambda x: 1 if x>0 else 0)
    X_test['hasPool'] = X_test['PoolArea'].apply(lambda x: 1 if x>0 else 0)

    X_train['has2ndFloor'] = X_test['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)
    X_val['has2ndFloor'] = X_val['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)
    X_test['has2ndFloor'] = X_test['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)

    X_train['hasGarage'] = X_test['GarageArea'].apply(lambda x: 1 if x>0 else 0)
    X_val['hasGarage'] = X_val['GarageArea'].apply(lambda x: 1 if x>0 else 0)
    X_test['hasGarage'] = X_test['GarageArea'].apply(lambda x: 1 if x>0 else 0)

    X_train['hasFireplace'] = X_test['Fireplaces'].apply(lambda x: 1 if x>0 else 0)
    X_val['hasFireplace'] = X_val['Fireplaces'].apply(lambda x: 1 if x>0 else 0)
    X_test['hasFireplace'] = X_test['Fireplaces'].apply(lambda x: 1 if x>0 else 0)

    X_train['hasBasement'] = X_test['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)
    X_val['hasBasement'] = X_val['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)
    X_test['hasBasement'] = X_test['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)

    X_train['Total_sqr_footage'] = (X_train['BsmtFinSF1'] + X_train['BsmtFinSF2'] + X_train['1stFlrSF'] + X_train['2ndFlrSF'])
    X_val['Total_sqr_footage'] = (X_val['BsmtFinSF1'] + X_val['BsmtFinSF2'] + X_val['1stFlrSF'] + X_val['2ndFlrSF'])
    X_test['Total_sqr_footage'] = (X_test['BsmtFinSF1'] + X_test['BsmtFinSF2'] + X_test['1stFlrSF'] + X_test['2ndFlrSF'])

    total_cols = [col for col in X_train.keys()]
    categorical_cols = [col for col in X_train.keys() if X_train[col].dtype == 'object']
    numerical_cols = list(set(total_cols) - set(categorical_cols))

    # identify categorical colummns with a high number of categories // cause problems for OneHotEncoding
    bad_cat_cols = {col : X_train[col].nunique() for col in categorical_cols if X_train[col].nunique() >= 20} 
    good_cat_cols = list(set(categorical_cols)-set(bad_cat_cols)) # categories used for OneHotEncoding

    categorical_cols = good_cat_cols

    # One Hot Encoding or Count Encoding
    if count_encoding:
        count_encoder = CountEncoder(cols=categorical_cols)
        count_encoder.fit(X_train[categorical_cols])
        
        enc_cols_train = pd.DataFrame(count_encoder.transform(X_train[categorical_cols]))
        enc_cols_val = pd.DataFrame(count_encoder.transform(X_val[categorical_cols]))
        enc_cols_test = pd.DataFrame(count_encoder.transform(X_test[categorical_cols]))

    else:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        enc_cols_train = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
        enc_cols_val = pd.DataFrame(encoder.transform(X_val[categorical_cols]))
        enc_cols_test = pd.DataFrame(encoder.transform(X_test[categorical_cols]))
        
    enc_cols_train.index = X_train.index
    enc_cols_val.index = X_val.index
    enc_cols_test.index = X_test.index

    num_cols_train = X_train.drop(columns=categorical_cols, axis=1, inplace=False)
    num_cols_val = X_val.drop(columns=categorical_cols, axis=1, inplace=False)
    num_cols_test = X_test.drop(columns=categorical_cols, axis=1, inplace=False)

    if logarithmic:
        for col in numerical_cols:
            num_cols_train[col] = np.log(num_cols_train[col]+1)
            num_cols_val[col] = np.log(num_cols_val[col]+1)
            num_cols_test[col] = np.log(num_cols_test[col]+1)
    
    if standardization: 
        for col in numerical_cols:
            num_cols_train[col] = (num_cols_train[col]-num_cols_train[col].mean())/num_cols_train[col].std()
            num_cols_val[col] = (num_cols_val[col]-num_cols_val[col].mean())/num_cols_val[col].std()
            num_cols_test[col] = (num_cols_test[col]-num_cols_test[col].mean())/num_cols_test[col].std()

    X_train = pd.concat(objs=[num_cols_train, enc_cols_train], axis=1)
    X_val = pd.concat(objs=[num_cols_val, enc_cols_val], axis=1)
    X_test = pd.concat(objs=[num_cols_test, enc_cols_test], axis=1)

    return X_train, y_train, X_val, y_val, X_test