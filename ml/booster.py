import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

# Display the first few rows of the dataset
print(data.head())

# Drop non-numeric columns such as Date and City (adjust based on your dataset)
data = data.drop(columns=['Date', 'City'], errors='ignore')

# Check if there are any remaining non-numeric columns and handle them accordingly
print(data.dtypes)

# Convert categorical columns to numeric if necessary
data = pd.get_dummies(data)

# Handle missing values (fill with the mean)
data = data.fillna(data.mean())

# Split the data into features and target variable
X = data.drop(columns=['AQI'])
y = data['AQI']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Linear Regression
# ------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set
y_pred_linear = linear_model.predict(X_test)

# Evaluate the Linear Regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print('--- Linear Regression Results ---')
print(f'Mean Squared Error (Linear): {mse_linear}')
print(f'R^2 Score (Linear): {r2_linear}')

# ------------------------------
# Lasso Regression
# ------------------------------
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha value for regularization strength
lasso_model.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set using Lasso
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the Lasso Regression model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print('--- Lasso Regression Results ---')
print(f'Mean Squared Error (Lasso): {mse_lasso}')
print(f'R^2 Score (Lasso): {r2_lasso}')

# ------------------------------
# Decision Tree Regressor
# ------------------------------
tree_model = DecisionTreeRegressor(random_state=42)  # You can adjust other hyperparameters like max_depth
tree_model.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set using Decision Tree
y_pred_tree = tree_model.predict(X_test)

# Evaluate the Decision Tree model
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print('--- Decision Tree Regressor Results ---')
print(f'Mean Squared Error (Decision Tree): {mse_tree}')
print(f'R^2 Score (Decision Tree): {r2_tree}')

# ------------------------------
# KNN Regressor
# ------------------------------
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
knn_model.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set using KNN
y_pred_knn = knn_model.predict(X_test)

# Evaluate the KNN Regressor model
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print('--- KNN Regressor Results ---')
print(f'Mean Squared Error (KNN): {mse_knn}')
print(f'R^2 Score (KNN): {r2_knn}')

# ------------------------------
# Random Forest Regressor
# ------------------------------
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators (number of trees)
forest_model.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set using RandomForestRegressor
y_pred_forest = forest_model.predict(X_test)

# Evaluate the Random Forest model
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print('--- Random Forest Regressor Results ---')
print(f'Mean Squared Error (Random Forest): {mse_forest}')
print(f'R^2 Score (Random Forest): {r2_forest}')

# ------------------------------
# XGBoost Regressor
# ------------------------------
xgboost_model = XGBRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators
xgboost_model.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set using XGBoost
y_pred_xgboost = xgboost_model.predict(X_test)

# Evaluate the XGBoost Regressor model
mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
r2_xgboost = r2_score(y_test, y_pred_xgboost)

print('--- XGBoost Regressor Results ---')
print(f'Mean Squared Error (XGBoost): {mse_xgboost}')
print(f'R^2 Score (XGBoost): {r2_xgboost}')
