import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your actual file)
# Assume the dataset has a column 'AQI' as target and others as features
data = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

# Display the first few arows of the dataset
print(data.head())

# Split the data into features and target variable
X = data.drop(columns=['AQI'])
y = data['AQI']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()  # Model assignment
linear_model.fit(X_train, y_train)  # Calling the fit method
