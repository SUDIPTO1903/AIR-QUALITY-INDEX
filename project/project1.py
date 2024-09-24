import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV

# Load the dataset from a CSV file
data = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

# Ensure correct column names and data types
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop rows with missing target 'AQI'
data = data.dropna(subset=['AQI'])

# Drop non-numeric columns before modeling
X = data.drop(['AQI', 'City', 'State'], axis=1, errors='ignore')

# Ensure that only numeric features are included
X = X.select_dtypes(include=[np.number])
y = data['AQI']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
y_pred_linear = linear_reg.predict(X_test_scaled)
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_linear))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))

# Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_reg.predict(X_test_scaled)
print("Lasso Regression R2 Score:", r2_score(y_test, y_pred_lasso))
print("Lasso Regression MSE:", mean_squared_error(y_test, y_pred_lasso))

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
print("Decision Tree R2 Score:", r2_score(y_test, y_pred_dt))
print("Decision Tree MSE:", mean_squared_error(y_test, y_pred_dt))

# KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)
y_pred_knn = knn_reg.predict(X_test_scaled)
print("KNN Regressor R2 Score:", r2_score(y_test, y_pred_knn))
print("KNN Regressor MSE:", mean_squared_error(y_test, y_pred_knn))

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))

# XGBoost Regressor
xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)
print("XGBoost R2 Score:", r2_score(y_test, y_pred_xgb))
print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))

# Hyperparameter tuning with GridSearchCV for RandomForest
rf_reg = RandomForestRegressor(random_state=42)  # Reset model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)
print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# ANN
ann = Sequential()
ann.add(Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]))
ann.add(Dense(units=32, activation='relu'))
ann.add(Dense(units=1))  # Regression has 1 output unit
ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train_scaled, y_train, epochs=100, batch_size=32)
y_pred_ann = ann.predict(X_test_scaled).flatten()
print("ANN R2 Score:", r2_score(y_test, y_pred_ann))
print("ANN MSE:", mean_squared_error(y_test, y_pred_ann))
