import pandas as pd

# Load the dataset
file_path = 'C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv'
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Function to retrieve air quality data for a specific date
def get_data_by_date(year, month, day):
    # Create a date object
    search_date = pd.Timestamp(year, month, day)
    
    # Filter the DataFrame for the specified date
    result = df[df['Date'].dt.date == search_date.date()]
    
    if result.empty:
        print(f"No data found for {search_date.date()}.")
    else:
        print(f"Air Quality data on {search_date.date()}:")
        # List of relevant columns for air quality metrics (adjust based on the dataset)
        columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3', 'Benzene', 'Toluene', 'Xylene']
        for col in columns:
            if col in result.columns:
                value = result[col].values[0]
                print(f"{col}: {value}")
            else:
                print(f"{col}: Not available in the dataset.")

# Input the day, month, and year to search for
day = int(input("Enter the day (1-31): "))
month = int(input("Enter the month (1-12): "))
year = int(input("Enter the year (e.g., 2023): "))

# Call the function to retrieve air quality data for the specified date
get_data_by_date(year, month, day)
