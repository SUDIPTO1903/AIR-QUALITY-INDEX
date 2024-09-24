import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
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
