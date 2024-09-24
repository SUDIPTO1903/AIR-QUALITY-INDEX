import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Convert categorical columns to numeric if necessary
data = pd.get_dummies(data)

# Handle missing values (fill with the mean)
data = data.fillna(data.mean())

# Check for null and infinite values
if data.isnull().sum().any():
    print("Data contains null values.")
if np.isinf(data).sum().any():
    print("Data contains infinite values.")

# Split the data into features and target variable
X = data.drop(columns=['AQI'])
y = data['AQI']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Hyperparameter Tuning for Random Forest
# ------------------------------
forest_model = RandomForestRegressor(random_state=42)
forest_param_distributions = {
    'n_estimators': [50, 100],  # Reasonable number of trees
    'max_depth': [None, 5, 10],  # Ensure valid values
    'min_samples_split': [2, 5]   # Ensure valid values
}
forest_random_search = RandomizedSearchCV(estimator=forest_model, param_distributions=forest_param_distributions, 
                                           n_iter=3, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=2)

try:
    forest_random_search.fit(X_train, y_train)
    print("Best Random Forest Hyperparameters:", forest_random_search.best_params_)

    # Make predictions with the best Random Forest model
    y_pred_forest = forest_random_search.predict(X_test)

    # Evaluate the Random Forest model
    mse_forest = mean_squared_error(y_test, y_pred_forest)
    r2_forest = r2_score(y_test, y_pred_forest)

    print('--- Random Forest Regressor Results ---')
    print(f'Mean Squared Error (Random Forest): {mse_forest}')
    print(f'R^2 Score (Random Forest): {r2_forest}')

except Exception as e:
    print(f"An error occurred during Random Forest fitting: {e}")

# ------------------------------
# Hyperparameter Tuning for XGBoost
# ------------------------------
xgboost_model = XGBRegressor(random_state=42)
xgboost_param_distributions = {
    'n_estimators': [50, 100],    # Reasonable number of trees
    'max_depth': [3, 5],          # Valid depth values
    'learning_rate': [0.01, 0.1]  # Reasonable learning rates
}
xgboost_random_search = RandomizedSearchCV(estimator=xgboost_model, param_distributions=xgboost_param_distributions, 
                                            n_iter=3, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=2)

try:
    xgboost_random_search.fit(X_train, y_train)
    print("Best XGBoost Hyperparameters:", xgboost_random_search.best_params_)

    # Make predictions with the best XGBoost model
    y_pred_xgboost = xgboost_random_search.predict(X_test)

    # Evaluate the XGBoost model
    mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
    r2_xgboost = r2_score(y_test, y_pred_xgboost)

    print('--- XGBoost Regressor Results ---')
    print(f'Mean Squared Error (XGBoost): {mse_xgboost}')
    print(f'R^2 Score (XGBoost): {r2_xgboost}')

except Exception as e:
    print(f"An error occurred during XGBoost fitting: {e}")
