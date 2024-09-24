 # Import essential libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # To split data into train and test
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import LinearRegression, Lasso  # ML algorithms for regression
from sklearn.tree import DecisionTreeRegressor  # Decision Tree Regressor
from sklearn.neighbors import KNeighborsRegressor  # KNN Regressor
from sklearn.ensemble import RandomForestRegressor  # Random Forest Regressor
import xgboost as xgb  # XGBoost Regressor
from sklearn.metrics import mean_squared_error, r2_score  # For evaluating model performance
from keras.models import Sequential  # For ANN model creation
from keras.layers import Dense  # For defining layers in ANN
from sklearn.model_selection import GridSearchCV  # For Hyperparameter tuning

# Load the dataset from a CSV file
data = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

# Display the first few rows of the dataset to understand the structure
print(data.head())

# Separate features (X) and target (y)
X = data.drop('AQI', axis=1)  # Features: All columns except AQI
y = data['AQI']  # Target: AQI column

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



    # Initialize the Linear Regression model
linear_reg = LinearRegression()

# Train the model
linear_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_linear = linear_reg.predict(X_test_scaled)

# Evaluate the performance
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_linear))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))

# Initialize the Lasso Regression model
lasso_reg = Lasso(alpha=0.1)

# Train the model
lasso_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lasso = lasso_reg.predict(X_test_scaled)

# Evaluate the performance
print("Lasso Regression R2 Score:", r2_score(y_test, y_pred_lasso))
print("Lasso Regression MSE:", mean_squared_error(y_test, y_pred_lasso))

# Initialize the Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42)

# Train the model
dt_reg.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_reg.predict(X_test)

# Evaluate the performance
print("Decision Tree R2 Score:", r2_score(y_test, y_pred_dt))
print("Decision Tree MSE:", mean_squared_error(y_test, y_pred_dt))

# Initialize the KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)

# Train the model
knn_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_knn = knn_reg.predict(X_test_scaled)

# Evaluate the performance
print("KNN Regressor R2 Score:", r2_score(y_test, y_pred_knn))
print("KNN Regressor MSE:", mean_squared_error(y_test, y_pred_knn))

# Initialize the Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_reg.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_reg.predict(X_test)

# Evaluate the performance
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
# Initialize the XGBoost Regressor
xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100)

# Train the model
xgb_reg.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_reg.predict(X_test)

# Evaluate the performance
print("XGBoost R2 Score:", r2_score(y_test, y_pred_xgb))
print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3, scoring='r2', verbose=1)

# Train the model with GridSearch
grid_search.fit(X_train, y_train)

# Best parameters and model performance
print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Initialize the ANN model
ann = Sequential()

# Add input layer and first hidden layer
ann.add(Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]))

# Add second hidden layer
ann.add(Dense(units=32, activation='relu'))

# Add output layer (since it's regression, we use one unit and no activation function)
ann.add(Dense(units=1))

# Compile the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
ann.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

# Make predictions
y_pred_ann = ann.predict(X_test_scaled)

# Evaluate the performance
print("ANN R2 Score:", r2_score(y_test, y_pred_ann))
print("ANN MSE:", mean_squared_error(y_test, y_pred_ann))

