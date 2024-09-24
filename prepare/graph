import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    avg = avg_data()  # Call the function
    
    # Plot the average PM2.5 data
    plt.plot(range(len(avg)), avg, label="Average PM2.5 data")
    plt.xlabel('Day')
    plt.ylabel('PM 2.5')
    plt.title('Daily Average PM2.5 Levels')
    plt.legend(loc='upper right')
    plt.show()
