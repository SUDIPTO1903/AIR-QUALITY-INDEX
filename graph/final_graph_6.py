import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import joblib
from tkinter import messagebox
from sklearnex import patch_sklearn
from sklearnex.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Patch sklearn to use Intel's optimizations
patch_sklearn()

    

def process_csv(input_file):
    headers="City Date PM2.5	PM10	NO	NO2	NOx	NH3	CO	SO2	O3	Benzene	Toluene	Xylene	AQI	"
    columns=headers.split()
    
    df = pd.read_csv(input_file)

    df_selected = df[columns]

    # df_cleaned = df_selected.dropna()

    return df_selected
    
def plot_avg_pm25_between_dates():
    # Read the CSV file
    df=process_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

    # Check if required columns exist
    if 'Date' not in df.columns or 'PM2.5' not in df.columns or 'City' not in df.columns:
        raise ValueError("The required columns 'Date', 'PM2.5', and 'City' are not in the CSV file.")

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Filter out rows with invalid dates
    df = df.dropna(subset=['Date'])

    # Replace non-numeric values in 'PM2.5' with NaN
    df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')

    # Ask user for the city name
    city_name = input("Enter the city name: ").strip()

    # Ask user for the start date
    print("Enter the start date:")
    start_day = int(input("Enter the day (1-31): "))
    start_month = int(input("Enter the month (1-12): "))
    start_year = int(input("Enter the year (e.g., 2023): "))

    # Ask user for the end date
    print("\nEnter the end date:")
    end_day = int(input("Enter the day (1-31): "))
    end_month = int(input("Enter the month (1-12): "))
    end_year = int(input("Enter the year (e.g., 2023): "))

    # Create datetime objects from the user input
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)

    # Filter data for the specified city and date range
    result = df[(df['City'].str.lower() == city_name.lower()) & 
                (df['Date'] >= start_date) & 
                (df['Date'] <= end_date)]

    # Check if any data exists for the given date range and city
    if result.empty:
        print(f"No data available for {city_name.title()} between {start_date.date()} and {end_date.date()}.")
        return

    # Group by date and calculate the average PM2.5 for each day within the range
    daily_avg = result.groupby(result['Date'])['PM2.5'].mean().reset_index()
    daily_avg.columns = ['Date', 'Avg_PM2.5']

    # Prepare the data for Linear Regression
    daily_avg['Date_Ordinal'] = daily_avg['Date'].map(pd.Timestamp.toordinal)

    X = daily_avg[['Date_Ordinal']]
    y = daily_avg['Avg_PM2.5']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    joblib.dump(model, "finalgraph.pkl")

    # Predicting PM2.5 values on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Future prediction for the next 30 days
    future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=30)
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    
    # Generate predicted PM2.5 values
    future_pm25_predictions = model.predict(future_dates_ordinal)

    # Add some noise to the predictions to simulate realistic fluctuations
    noise = np.random.normal(0, 5, future_pm25_predictions.shape)  # Add noise with mean 0 and std dev 5
    future_pm25_predictions_with_noise = future_pm25_predictions + noise

    # Prepare DataFrame for plotting
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted PM2.5': future_pm25_predictions_with_noise})

    # Plot the historical daily average PM2.5 values and predictions
    plt.figure(figsize=(12, 6))
    plt.plot(daily_avg['Date'], daily_avg['Avg_PM2.5'], marker='o', label='Average PM2.5', linestyle='-', color='blue')
    plt.plot(future_df['Date'], future_df['Predicted PM2.5'], marker='x', label='Predicted PM2.5', linestyle='--', color='orange')

    plt.xlabel('Date')
    plt.ylabel('PM2.5')
    plt.title(f'Average PM2.5 Levels for {city_name.title()} from {start_date.date()} to {end_date.date()}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

def show_alert():
    # Create a new window using Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Display the alert message in a pop-up window
    message = ('''PM2.5 value reached 300: Immediate actions required (e.g., stop outdoor activities, wear masks, close windows).\n
PM2.5 value reached 250: Caution advised (e.g., reduce outdoor exposure, avoid strenuous activities).\n
PM2.5 value reached 200: Caution advised (e.g., Avoid all outdoor activities. Stay indoors in sealed, air-conditioned environments. Use high-quality air purifiers indoors. Avoid ventilation or opening windows unless using filtered air.).\n
PM2.5 value reached 150: Caution advised (e.g., Avoid prolonged outdoor exertion. Use air purifiers indoors to reduce PM levels. Keep windows and doors closed. Avoid exercising outdoors.).\n
PM2.5 value reached 100: Caution advised (e.g., Consider reducing outdoor activities during the day. Monitor local air quality updates.).\n
PM2.5 value reached 50: Caution advised (e.g., Limit prolonged outdoor exertion. Consider wearing a basic dust mask when outside for long periods. Keep windows closed if indoor air is filtered and cooler.).\n
''') 
    messagebox.showinfo("PM2.5 Alert", message)

    # Destroy the Tkinter window after showing the message
    root.destroy()

# Run the function
plot_avg_pm25_between_dates()
show_alert()
