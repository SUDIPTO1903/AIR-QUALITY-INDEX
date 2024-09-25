import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

def avg_data():
    # Read the CSV file
    df = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

    # Check if required columns exist
    if 'Date' not in df.columns or 'PM2.5' not in df.columns:
        raise ValueError("The required columns 'Date' and 'PM2.5' are not in the CSV file.")

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Filter out rows with invalid dates and invalid PM2.5 values
    df = df.dropna(subset=['Date'])
    
    # Replace non-numeric values in 'PM2.5' with NaN
    df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')

    # Group by date and calculate the daily average of PM2.5
    daily_avg = df.groupby(df['Date'].dt.date)['PM2.5'].mean()

    # Convert the result to a list for plotting
    average = daily_avg.tolist()
    return average

def check_thresholds(avg):
    # Create a new window using Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Set flags for threshold breaches
    message = ""

    if any(value >= 300 for value in avg):
        message += "PM2.5 value reached 300: Immediate actions required (e.g., stop outdoor activities, wear masks, close windows).\n"
    if any(value >= 250 and value < 300 for value in avg):
        message += "\nPM2.5 value reached 250: Caution advised (e.g., reduce outdoor exposure, avoid strenuous activities).\n"
    if any(value >= 200 and value < 250 for value in avg):
        message += "\nPM2.5 value reached 200: Caution advised (e.g., Avoid all outdoor activities. Stay indoors in sealed, air-conditioned environments. Use high-quality air purifiers indoors. Avoid ventilation or opening windows unless using filtered air.).\n"
    if any(value >= 150 and value < 200 for value in avg):
        message += "\nPM2.5 value reached 150: Caution advised (e.g., Avoid prolonged outdoor exertion. Use air purifiers indoors to reduce PM levels. Keep windows and doors closed. Avoid exercising outdoors.).\n"
    if any(value >= 100 and value < 150 for value in avg):
        message += "\nPM2.5 value reached 100: Caution advised (e.g., Consider reducing outdoor activities during the day. Monitor local air quality updates.).\n"
    if any(value >= 50 and value < 100 for value in avg):
        message += "\nPM2.5 value reached 50: Caution advised (e.g., Limit prolonged outdoor exertion. Consider wearing a basic dust mask when outside for long periods. Keep windows closed if indoor air is filtered and cooler.).\n"

    # If thresholds are breached, display the message in a new window
    if message:
        messagebox.showinfo("PM2.5 Alert", message)
    else:
        messagebox.showinfo("PM2.5 Alert", "No critical PM2.5 thresholds breached.")

    root.destroy()  # Close the window after showing the message

if __name__ == "__main__":
    avg = avg_data()  # Call the function

  

    plt.plot(range(len(avg)), avg, label="Average PM2.5 data")
    plt.xlabel('Day')
    plt.ylabel('PM 2.5')
    plt.title('Daily Average PM2.5 Levels')
    plt.legend(loc='upper right')
    plt.show()
      # Plot the average PM2.5 data
    check_thresholds(avg)