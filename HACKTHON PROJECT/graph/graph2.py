import pandas as pd

def avg_pm25_on_date():
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
        
        # Output the result
        print(f"Average PM2.5 value on {search_date.date()} is: {avg_pm25:.2f}")

# Run the function
avg_pm25_on_date()
