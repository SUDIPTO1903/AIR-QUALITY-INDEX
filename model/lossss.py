import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

# Load your dataset
data = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

# Drop non-numeric columns such as Date and City (adjust based on your dataset)
data = data.drop(columns=['Date', 'City'], errors='ignore')

# Convert categorical columns to numeric if necessary
data = pd.get_dummies(data)

# Handle missing values (fill with the mean)
data = data.fillna(data.mean())

# Split the data into features and target variable
X = data.drop(columns=['AQI'])
y = data['AQI']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))  # Input layer
model.add(layers.Dense(32, activation='relu'))  # Hidden layer
model.add(layers.Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
from tensorflow.keras.callbacks import EarlyStopping

# Build the ANN model
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))  # Input layer
model.add(layers.Dense(32, activation='relu'))  # Hidden layer
model.add(layers.Dense(1))  # Output layer for regression

# Compile the model with a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Adjust learning rate here
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with validation split
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)


print('--- ANN Results ---')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
