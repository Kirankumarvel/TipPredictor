from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

#Dataset Analysis
# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

# Display the dataset
print(raw_data.head())

if 'tip_amount' in raw_data.columns:
    correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
else:
    raise KeyError("The column 'tip_amount' is missing from the dataset.")

# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# Ensure only numeric columns are used for normalization
proc_data = proc_data.select_dtypes(include=['number'])

# get the feature matrix used for training
# Normalize the feature matrix without using .values
X = normalize(proc_data, axis=1, norm='l1', copy=True)

# Dataset Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor

# Build a Decision Tree Regressor model with Scikit-Learn
dt_reg = DecisionTreeRegressor(criterion='friedman_mse',
                               max_depth=8,
                               random_state=35,
                               ccp_alpha=0.0)

# Train the model
dt_reg.fit(X_train, y_train)

# Run inference using the sklearn model
y_pred = dt_reg.predict(X_test)

# Evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print(f'MSE score: {mse_score:.3f}')

# Evaluate R-squared score (R²)
r2_score = dt_reg.score(X_test, y_test)
print(f'R² score: {r2_score:.3f}')

# Task 1: Identify the top 3 features with the most effect on the `tip_amount`
# Reuse the previously computed correlation values
top_features = correlation_values.abs().sort_values(ascending=False).head(3)
print("\nTop 3 features with the highest correlation to 'tip_amount':")
print(top_features)

# Task 2: Remove low-correlation features and reprocess the dataset
columns_to_drop = ['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge']
proc_data = raw_data.drop(columns=columns_to_drop, errors='ignore').select_dtypes(include=['number'])

# Normalize the updated feature matrix
X = normalize(proc_data, axis=1, norm='l1', copy=False)

# Confirm remaining columns
print("\nRemaining columns after dropping low-correlation features:")
print(proc_data.columns)
y = raw_data[['tip_amount']].values.astype('float32')

# Re-split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#task 3: Check the effect of decreasing the max_depth parameter to 4 on the and values.
dt_reg_4 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4, random_state=35, ccp_alpha=0.0)

# Train the model
dt_reg_4.fit(X_train, y_train)

# Run inference
# Evaluate R-squared score (R²)
r2_score_4 = dt_reg_4.score(X_test, y_test)
print('R² score with max_depth=4: {0:.3f}'.format(r2_score_4))

# Run inference for max_depth=4
y_pred_4 = dt_reg_4.predict(X_test)

# Evaluate Mean Squared Error (MSE)
mse_score_4 = mean_squared_error(y_test, y_pred_4)
print('MSE score with max_depth=4: {0:.3f}'.format(mse_score_4))

# Evaluate R-squared score (R²)
r2_score_4 = dt_reg_4.score(X_test, y_test)
print('R² score with max_depth=4: {0:.3f}'.format(r2_score_4))

