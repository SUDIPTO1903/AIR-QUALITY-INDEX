import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
# Hyperparameter Tuning for Random Forest
# ------------------------------
forest_model = RandomForestRegressor(random_state=42)
forest_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
forest_grid_search = GridSearchCV(estimator=forest_model, param_grid=forest_param_grid, 
                                   scoring='neg_mean_squared_error', cv=5, verbose=1)
forest_grid_search.fit(X_train, y_train)

# Best Random Forest parameters
print("Best Random Forest Hyperparameters:", forest_grid_search.best_params_)

# Make predictions with the best Random Forest model
y_pred_forest = forest_grid_search.predict(X_test)

# Evaluate the Random Forest model
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print('--- Random Forest Regressor Results ---')
print(f'Mean Squared Error (Random Forest): {mse_forest}')
print(f'R^2 Score (Random Forest): {r2_forest}')

# ------------------------------
# Hyperparameter Tuning for XGBoost
# ------------------------------
xgboost_model = XGBRegressor(random_state=42)
xgboost_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
xgboost_grid_search = GridSearchCV(estimator=xgboost_model, param_grid=xgboost_param_grid, 
                                    scoring='neg_mean_squared_error', cv=5, verbose=1)
xgboost_grid_search.fit(X_train, y_train)

# Best XGBoost parameters
print("Best XGBoost Hyperparameters:", xgboost_grid_search.best_params_)

# Make predictions with the best XGBoost model
y_pred_xgboost = xgboost_grid_search.predict(X_test)

# Evaluate the XGBoost model
mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
r2_xgboost = r2_score(y_test, y_pred_xgboost)

print('--- XGBoost Regressor Results ---')
print(f'Mean Squared Error (XGBoost): {mse_xgboost}')
print(f'R^2 Score (XGBoost): {r2_xgboost}')
