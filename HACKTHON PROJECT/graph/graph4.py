import pandas as pd
import matplotlib.pyplot as plt

def simulate_reduction(df, threshold=50, reduction_factor=0.8):
    """
    Simulate actions to reduce PM2.5 levels when they exceed a threshold.
    
    Args:
    - df: DataFrame with PM2.5 data.
    - threshold: The PM2.5 level above which actions will be simulated.
    - reduction_factor: The factor by which PM2.5 levels are reduced after intervention.
    
    Returns:
    - modified_df: DataFrame with modified PM2.5 levels after interventions.
    """
    
    # Copy the original dataframe to avoid modifying it directly
    modified_df = df.copy()
    
    # Apply intervention where PM2.5 exceeds the threshold
    modified_df.loc[modified_df['PM2.5'] > threshold, 'PM2.5'] *= reduction_factor
    
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

    # Set a threshold and reduction factor for PM2.5 levels
    pm_threshold = 50  # µg/m³
    reduction_factor = 0.8  # 20% reduction
    
    # Simulate PM2.5 reduction
    reduced_df = simulate_reduction(avg_df, threshold=pm_threshold, reduction_factor=reduction_factor)
    
    # Plot original and reduced PM2.5 levels
    plt.figure(figsize=(10, 6))
    plt.plot(avg_df['Date'], avg_df['PM2.5'], label="Original PM2.5", color='red', linestyle='--')
    plt.plot(reduced_df['Date'], reduced_df['PM2.5'], label="Reduced PM2.5", color='green')
    plt.xlabel('Date')
    plt.ylabel('PM 2.5 Levels (µg/m³)')
    plt.title('PM2.5 Levels Before and After Reduction Interventions')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
