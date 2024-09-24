import pandas as pd
import matplotlib.pyplot as plt

# Correct path to include the .csv extension
file_path = "C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df)

# If you want to access a specific column, you can do so like this:
# print(df['ColumnName'])  # Replace 'ColumnName' with the actual column name from your CSV

# Replace with the actual path to your CSV file
file_path = 'C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format (if not already in that format)
df['Date'] = pd.to_datetime(df['Date'])

# Function to retrieve data based on day, month, and year
def get_data_by_date(year, month, day):
    # Create a date object
    search_date = pd.Timestamp(day, month, year)
    
    # Filter the DataFrame for the specified date
    result = df[df['Date'] == search_date]
    
    if result.empty:
        print("No data found for the specified date.")
    else:
         print("Pollutant concentrations on", search_date.date())
         print(result[['NO2', 'CO', 'SO2', 'O3', 'NH3', 'Benzene', 'Toluene', 'Xylene']])

# Input the day, month, and year you want to search for
day = int(input("Enter the day (1-31): "))
month = int(input("Enter the month (1-12): "))
year = int(input("Enter the year (e.g., 2023): "))


# Call the function with user input
get_data_by_date(day, month, year)
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def avg_data():
    average = []
    df = pd.read_csv('C:\\Users\\HP\\Desktop\\dataset\\city_day.csv.csv')
                
                     
                
    # Check if required columns exist
    if 'Date' not in df.columns or 'PM2.5' not in df.columns:
        raise ValueError("The required columns 'Date' and 'PM2.5' are not in the CSV file.")

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Filter out rows with invalid dates
    df = df.dropna(subset=['Date'])

    # Group by day and calculate the average PM2.5
    daily_avg = df.groupby(df['Date'].dt.date)['PM2.5'].apply(
        lambda x: x[x.isin(['NoData', 'PwrFail', '---', 'InVld']) == False].astype(float)
                     .mean() if not x.empty else None
    )

    average = daily_avg.tolist()
    return average

if __name__ == "__main__":
    avg = avg_data()  # Call the function
    plt.plot(range(len(avg)), avg, label="Average PM2.5 data")
    plt.xlabel('Day')
    plt.ylabel('PM 2.5')
    plt.legend(loc='upper right')
    plt.show()
   