# analyze_performance_by_time.py
import pandas as pd
import os

def analyze_by_time(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath, parse_dates=['EntryTime'])
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return

    df['EntryHour'] = df['EntryTime'].dt.hour
    
    # Group by the entry hour and calculate key stats
    hourly_performance = df.groupby('EntryHour').agg(
        TotalTrades=('PnL', 'count'),
        TotalPnL=('PnL', 'sum'),
        WinRate=('PnL', lambda x: (x > 0).mean())
    ).round(2)

    hourly_performance['AvgPnL_PerTrade'] = hourly_performance['TotalPnL'] / hourly_performance['TotalTrades']
    
    print("\n" + "="*60)
    print("Strategy Performance by Entry Hour")
    print("="*60)
    print(hourly_performance)
    print("="*60)
    print("\nAnalyze this table to see if specific hours are significantly more or less profitable.")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Update this with the name of your latest backtest results file.
    backtest_filename = 'backtest_results_02e075c7-f827-4d15-a061-ae11c86c6201.csv' # <-- CHANGE THIS
    
    file_path = os.path.join('static', 'downloads', backtest_filename)
    analyze_by_time(file_path)