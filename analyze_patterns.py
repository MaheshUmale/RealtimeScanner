# analyze_patterns.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def plot_analysis(df_analysis, column_name, job_id):
    """Generates and saves a plot visualizing the performance analysis."""
    if df_analysis.empty:
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Bar plot for Total PnL
    sns.barplot(x=df_analysis.index.astype(str), y=df_analysis['TotalPnL'], ax=ax1, alpha=0.6, palette="coolwarm")
    ax1.set_ylabel('Total PnL (₹)', color='black')
    ax1.set_xlabel(f'Buckets for {column_name}')
    ax1.tick_params(axis='x', rotation=45)
    
    # Line plot for Win Rate on a secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(x=df_analysis.index.astype(str), y=df_analysis['WinRate'], ax=ax2, color='green', marker='o', label='Win Rate')
    ax2.set_ylabel('Win Rate (%)', color='green')
    ax2.set_ylim(0, 1) # Win rate is between 0 and 1
    
    # Add trade count as text on the bars
    for i, p in enumerate(ax1.patches):
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width() / 2., max(0, height) + 0.1, f"n={df_analysis['TotalTrades'].iloc[i]}", ha="center")

    plt.title(f'Performance Analysis by {column_name}')
    fig.tight_layout()
    
    # Save the plot
    plot_filename = f'analysis_{column_name}_{job_id}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved analysis plot to '{plot_filename}'")


def analyze_by_bin(df, column_name, num_bins=5):
    """Analyzes performance by binning a continuous variable."""
    print(f"\n--- Analyzing Performance by {column_name} ---")
    
    # Create bins/buckets for the continuous variable
    try:
        df[f'{column_name}_Bin'] = pd.qcut(df[column_name], num_bins, duplicates='drop')
    except ValueError:
        print(f"Could not create bins for {column_name}. Might have too few unique values. Skipping.")
        return pd.DataFrame()

    analysis = df.groupby(f'{column_name}_Bin').agg(
        TotalTrades=('PnL', 'count'),
        TotalPnL=('PnL', 'sum'),
        WinRate=('PnL', lambda x: (x > 0).mean())
    )
    analysis['WinRate'] = (analysis['WinRate'] * 100).round(2)
    analysis['TotalPnL'] = analysis['TotalPnL'].round(2)
    
    print(analysis)
    return analysis


def analyze_by_category(df, column_name):
    """Analyzes performance by a categorical variable like 'Direction'."""
    print(f"\n--- Analyzing Performance by {column_name} ---")
    
    analysis = df.groupby(column_name).agg(
        TotalTrades=('PnL', 'count'),
        TotalPnL=('PnL', 'sum'),
        WinRate=('PnL', lambda x: (x > 0).mean())
    )
    analysis['WinRate'] = (analysis['WinRate'] * 100).round(2)
    analysis['TotalPnL'] = analysis['TotalPnL'].round(2)
    
    print(analysis)
    return analysis


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Find the Job ID (the long string of characters) from your results file name
    # e.g., for 'backtest_results_c3c2a044-....csv', the ID is 'c3c2a044-....'
    #backtest_results_a0b623ca-2639-4dbc-9e72-e4169b218137.csv
    job_id = 'a0b623ca-2639-4dbc-9e72-e4169b218137'  # <-- CHANGE THIS TO YOUR LATEST JOB ID

    results_filename = f'backtest_results_{job_id}.csv'
    config_filename = f'backtest_config_{job_id}.json'
    
    results_filepath = os.path.join('static', 'downloads', results_filename)
    config_filepath = os.path.join('static', 'downloads', config_filename)

    # Load the data
    try:
        df_results = pd.read_csv(results_filepath)
        with open(config_filepath, 'r') as f:
            config_used = json.load(f)
    except FileNotFoundError:
        print(f"Error: Make sure both '{results_filename}' and '{config_filename}' exist in 'static/downloads/'")
        exit()

    print("="*60)
    print(f"Analyzing Backtest Job: {job_id}")
    print("Configuration Used for this Run:")
    print(json.dumps(config_used, indent=2))
    print("="*60)

    # --- Run and Plot Analysis for Different Factors ---
    
    # 1. Analyze by Volume Ratio
    vol_ratio_analysis = analyze_by_bin(df_results, 'VolumeRatio', num_bins=5)
    plot_analysis(vol_ratio_analysis, 'VolumeRatio', job_id)

    # 2. Analyze by Value Traded (in Crores)
    value_traded_analysis = analyze_by_bin(df_results, 'ValueTraded', num_bins=5)
    plot_analysis(value_traded_analysis, 'ValueTraded', job_id)

    # 3. Analyze by Consolidation Ratio
    # Note: This requires the 'ConsolidationRatio' column which was added.
    if 'ConsolidationRatio' in df_results.columns:
        consolidation_analysis = analyze_by_bin(df_results, 'ConsolidationRatio', num_bins=4)
        plot_analysis(consolidation_analysis, 'ConsolidationRatio', job_id)
    else:
        print("\n'ConsolidationRatio' column not found. Please re-run backtest to generate it.")
    
    # 4. Analyze by Direction (Long vs. Short)
    direction_analysis = analyze_by_category(df_results, 'Direction')
    plot_analysis(direction_analysis, 'Direction', job_id)