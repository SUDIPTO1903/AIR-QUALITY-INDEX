import pandas as pd
import matplotlib.pyplot as plt

def plot_pm25_on_date():
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

    # Ask user for the day, month, and year
    day = int(input("Enter the day (1-31): "))
    month = int(input("Enter the month (1-12): "))
    year = int(input("Enter the year (e.g., 2023): "))

    # Create a datetime object from the user input
    search_date = pd.Timestamp(year=year, month=month, day=day)

    # Filter data for the specified date
    result = df[df['Date'].dt.date == search_date.date()]

    # Check if any data exists for the given date
    if result.empty:
        print(f"No data available for {search_date.date()}.")
    else:
        # Calculate the average PM2.5 for that day
        avg_pm25 = result['PM2.5'].mean()
        print(f"Average PM2.5 value on {search_date.date()} is: {avg_pm25:.2f}")

        # Plot PM2.5 values for the specific day
        plt.plot(result['Date'], result['PM2.5'], marker='o', label='PM2.5 values')
        plt.xlabel('Time')
        plt.ylabel('PM 2.5')
        plt.title(f'PM2.5 Values on {search_date.date()}')
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.grid(True)
        plt.tight_layout()  # Adjust the layout to prevent label overlap
        plt.show()

# Run the function
plot_pm25_on_date()
