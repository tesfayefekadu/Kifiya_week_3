import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import shap

import pandas as pd

# Feature Engineering: Create additional features
def feature_engineering(df):
    df['TransactionYear'] = pd.to_datetime(df['TransactionMonth']).dt.year
    df['TransactionMonthOnly'] = pd.to_datetime(df['TransactionMonth']).dt.month
    return df

# Encode Categorical Variables using One-Hot Encoding
def encode_categorical(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

# Split Data into Train and Test
def train_test_split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# Build and Train Linear Regression Model
def linear_regression_model(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model
# Build and Train Random Forest Model
def random_forest_model(X_train, y_train, n_estimators=100, random_state=42):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

# Build and Train XGBoost Model
def xgboost_model(X_train, y_train, random_state=42):
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    xgboost_model.fit(X_train, y_train)
    return xgboost_model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'R2': r2}


# Analyze Feature Importance using SHAP
def shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")