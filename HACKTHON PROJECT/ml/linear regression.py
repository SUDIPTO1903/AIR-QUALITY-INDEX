import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

# Display the first few rows of the dataset
print(data.head())

# Drop columns that are non-numeric or irrelevant to the model (like dates or cities)
# You can also use get_dummies to convert categorical data if needed
data = data.drop(columns=['Date', 'City'], errors='ignore')  # Adjust these column names based on your dataset

# Check if there are any remaining non-numeric columns and handle them accordingly
print(data.dtypes)

# If there are categorical columns, convert them to numeric using pd.get_dummies
data = pd.get_dummies(data)

# Handle missing values (if any)
data = data.fillna(data.mean())  # Alternatively, you could drop rows with missing values

# Split the data into features and target variable
X = data.drop(columns=['AQI'])
y = data['AQI']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set
y_pred = linear_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
