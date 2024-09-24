import pandas as pd
import matplotlib.pyplot as plt

def plot_avg_pm25_between_dates():
    # Read the CSV file
    df = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')

    # Check if required columns exist
    if 'Date' not in df.columns or 'PM2.5' not in df.columns:
        raise ValueError("The required columns 'Date' and 'PM2.5' are not in the CSV file.")

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Filter out rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    # Replace non-numeric values in 'PM2.5' with NaN
    df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')

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

    # Filter data for the specified date range
    result = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Check if any data exists for the given date range
    if result.empty:
        print(f"No data available between {start_date.date()} and {end_date.date()}.")
    else:
        # Group by date and calculate the average PM2.5 for each day within the range
        daily_avg = result.groupby(result['Date'].dt.date)['PM2.5'].mean()

        # Print the daily average PM2.5 values
        print(f"\nAverage PM2.5 values between {start_date.date()} and {end_date.date()}:")
        print(daily_avg)

        # Plot the daily average PM2.5 values
        plt.plot(daily_avg.index, daily_avg.values, marker='o', label='Average PM2.5')
        plt.xlabel('Date')
        plt.ylabel('PM 2.5')
        plt.title(f'Average PM2.5 Levels from {start_date.date()} to {end_date.date()}')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

# Run the function
plot_avg_pm25_between_dates()
