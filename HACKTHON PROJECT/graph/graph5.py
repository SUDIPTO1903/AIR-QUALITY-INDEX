import pandas as pd
import matplotlib.pyplot as plt

def simulate_reduction(df, max_limit=35):
    """
    Simulate actions to reduce PM2.5 levels to a maximum limit.
    
    Args:
    - df: DataFrame with PM2.5 data.
    - max_limit: The maximum PM2.5 level allowed after interventions.
    
    Returns:
    - modified_df: DataFrame with modified PM2.5 levels after interventions.
    """
    
    # Copy the original dataframe to avoid modifying it directly
    modified_df = df.copy()
    
    # Apply cap on PM2.5 levels to ensure they don't exceed the max_limit
    modified_df['PM2.5'] = modified_df['PM2.5'].apply(lambda x: min(x, max_limit))
    
    return modified_df

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
    daily_avg = df.groupby(df['Date'].dt.date)['PM2.5'].mean().reset_index()
    
    return daily_avg

if __name__ == "__main__":
    # Load and compute daily average PM2.5 data
    avg_df = avg_data()

    # Set the max limit for PM2.5 levels
    pm_max_limit = 35  # µg/m³, cap at 35

    # Simulate PM2.5 reduction to the max limit
    reduced_df = simulate_reduction(avg_df, max_limit=pm_max_limit)
    
    # Plot original and reduced PM2.5 levels
    plt.figure(figsize=(10, 6))
    plt.plot(avg_df['Date'], avg_df['PM2.5'], label="Original PM2.5", color='red', linestyle='--')
    plt.plot(reduced_df['Date'], reduced_df['PM2.5'], label="Reduced PM2.5 (Capped at 35)", color='green')
    plt.xlabel('Date')
    plt.ylabel('PM 2.5 Levels (µg/m³)')
    plt.title('PM2.5 Levels Before and After Capping at 35')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
