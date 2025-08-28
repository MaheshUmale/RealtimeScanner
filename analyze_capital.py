# analyze_capital.py
import pandas as pd
import os
from config import MAX_CAPITAL_PER_TRADE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_peak_capital(csv_filepath):
    """
    Analyzes a backtest results CSV to find the maximum number of concurrent trades
    and the peak capital required.
    """
    # 1. Load the backtest results data
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            print("The CSV file is empty. No trades to analyze.")
            return
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        print("Please make sure the file path is correct.")
        return

    print(f"Analyzing {len(df)} trades from '{os.path.basename(csv_filepath)}'...")

    # 2. Convert time columns to proper datetime objects for sorting
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])

    # 3. Create a list of all events (entries and exits)
    # An event is a tuple: (timestamp, event_type)
    # event_type is +1 for an entry (a position opens) and -1 for an exit (a position closes)
    events = []
    for _, row in df.iterrows():
        events.append((row['EntryTime'], 1))
        events.append((row['ExitTime'], -1))

    # 4. Sort all events chronologically by their timestamp
    events.sort(key=lambda x: x[0])

    # 5. Iterate through the timeline of events to find the peak concurrent positions
    max_concurrent_positions = 0
    current_concurrent_positions = 0
    for timestamp, event_type in events:
        current_concurrent_positions += event_type
        if current_concurrent_positions > max_concurrent_positions:
            max_concurrent_positions = current_concurrent_positions
            print(f" current max ={max_concurrent_positions} for time {timestamp}")

    # 6. Calculate peak capital and display the results
    peak_capital_required = max_concurrent_positions * MAX_CAPITAL_PER_TRADE

    print("\n" + "="*50)
    print("Peak Capital and Position Analysis")
    print("="*50)
    print(f"Capital Allocated Per Trade (from config): ₹{MAX_CAPITAL_PER_TRADE:,.2f}")
    print("-" * 50)
    print(f"Maximum Concurrent Positions Open: {max_concurrent_positions}")
    print(f"PEAK CAPITAL REQUIRED FOR STRATEGY: ₹{peak_capital_required:,.2f}")
    print("="*50)
    print("\nThis is the maximum amount of capital that was deployed at a single moment during the backtest.")



class BacktestAnalyzer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.df = None
        
    def load_results(self):
        """Load backtest results"""
        if not os.path.exists(self.results_file):
            print(f"Results file {self.results_file} not found")
            return False
            
        print(self.results_file)
        self.df = pd.read_csv(self.results_file)
        
        # Convert datetime columns
        self.df['EntryTime'] = pd.to_datetime(self.df['EntryTime'])
        self.df['ExitTime'] = pd.to_datetime(self.df['ExitTime'])
        
        # Extract date and time
        self.df['EntryDate'] = self.df['EntryTime'].dt.date
        self.df['EntryHour'] = self.df['EntryTime'].dt.hour
        
        return True
    
    def analyze_performance(self):
        """Analyze overall performance"""
        if self.df is None:
            return
            
        print("Analyzing backtest performance...")
        
        # Basic metrics
        total_trades = len(self.df)
        winning_trades = len(self.df[self.df['PnL'] > 0])
        losing_trades = len(self.df[self.df['PnL'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = self.df['PnL'].sum()
        avg_pnl = self.df['PnL'].mean()
        max_win = self.df['PnL'].max()
        max_loss = self.df['PnL'].min()
        profit_factor = self.df[self.df['PnL'] > 0]['PnL'].sum() / abs(self.df[self.df['PnL'] < 0]['PnL'].sum()) if len(self.df[self.df['PnL'] < 0]) > 0 else float('inf')
        
        # Performance by direction
        long_trades = self.df[self.df['Direction'] == 'Long']
        short_trades = self.df[self.df['Direction'] == 'Short']
        
        long_win_rate = len(long_trades[long_trades['PnL'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['PnL'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
        # Performance by exit reason
        exit_reasons = self.df['ExitReason'].value_counts()
        target_hit_rate = len(self.df[self.df['ExitReason'] == 'TARGET']) / total_trades if total_trades > 0 else 0
        
        # Drawdown analysis
        cumulative_pnl = self.df.sort_values('EntryTime')['PnL'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Create summary
        summary = {
            'Total_Trades': total_trades,
            'Winning_Trades': winning_trades,
            'Losing_Trades': losing_trades,
            'Win_Rate': win_rate,
            'Total_PnL': total_pnl,
            'Avg_PnL': avg_pnl,
            'Max_Win': max_win,
            'Max_Loss': max_loss,
            'Profit_Factor': profit_factor,
            'Max_Drawdown': max_drawdown,
            'Long_Win_Rate': long_win_rate,
            'Short_Win_Rate': short_win_rate,
            'Target_Hit_Rate': target_hit_rate
        }
        
        # Save to CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(ANALYSIS_OUTPUT_DIR, 'performance_summary.csv'), index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("BACKTEST ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total PnL: ₹{total_pnl:,.2f}")
        print(f"Average PnL: ₹{avg_pnl:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: ₹{max_drawdown:,.2f}")
        print(f"Long Win Rate: {long_win_rate:.2%}")
        print(f"Short Win Rate: {short_win_rate:.2%}")
        print(f"Target Hit Rate: {target_hit_rate:.2%}")
        
        return summary
    
    def analyze_by_time(self):
        """Analyze performance by time of day"""
        if self.df is None:
            return
            
        print("Analyzing by time of day...")
        
        # Group by entry hour
        hourly_analysis = self.df.groupby('EntryHour').agg({
            'PnL': ['count', 'sum', 'mean'],
            'Direction': lambda x: (x == 'Long').sum()
        }).round(2)
        
        hourly_analysis.columns = ['_'.join(col).strip() for col in hourly_analysis.columns]
        hourly_analysis = hourly_analysis.reset_index()
        hourly_analysis.columns = ['Hour', 'Trade_Count', 'Total_PnL', 'Avg_PnL', 'Long_Trades']
        hourly_analysis['Win_Rate'] = self.df.groupby('EntryHour')['PnL'].apply(lambda x: (x > 0).mean()).round(2).values
        
        # Save to CSV
        hourly_analysis.to_csv(os.path.join(ANALYSIS_OUTPUT_DIR, 'hourly_analysis.csv'), index=False)
        
        return hourly_analysis
    
    def analyze_by_symbol(self):
        """Analyze performance by symbol"""
        if self.df is None:
            return
            
        print("Analyzing by symbol...")
        
        symbol_analysis = self.df.groupby('Symbol').agg({
            'PnL': ['count', 'sum', 'mean'],
            'Direction': lambda x: (x == 'Long').sum()
        }).round(2)
        
        symbol_analysis.columns = ['_'.join(col).strip() for col in symbol_analysis.columns]
        symbol_analysis = symbol_analysis.reset_index()
        symbol_analysis.columns = ['Symbol', 'Total_Trades', 'Total_PnL', 'Avg_PnL', 'Long_Trades']
        symbol_analysis['Win_Rate'] = self.df.groupby('Symbol')['PnL'].apply(lambda x: (x > 0).mean()).round(2).values
        
        # Sort by total PnL
        symbol_analysis = symbol_analysis.sort_values('Total_PnL', ascending=False)
        
        # Save to CSV
        symbol_analysis.to_csv(os.path.join(ANALYSIS_OUTPUT_DIR, 'symbol_analysis.csv'), index=False)
        
        return symbol_analysis
    
    def create_visualizations(self):
        """Create analysis visualizations"""
        if self.df is None:
            return
            
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Backtest Analysis', fontsize=16, fontweight='bold')
        
        # 1. PnL Distribution
        axes[0, 0].hist(self.df['PnL'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.df['PnL'].mean(), color='red', linestyle='--', label=f'Mean: ₹{self.df["PnL"].mean():,.2f}')
        axes[0, 0].set_xlabel('PnL (₹)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('PnL Distribution')
        axes[0, 0].legend()
        
        # 2. Cumulative PnL
        df_sorted = self.df.sort_values('EntryTime')
        df_sorted['Cumulative_PnL'] = df_sorted['PnL'].cumsum()
        axes[0, 1].plot(df_sorted['EntryTime'], df_sorted['Cumulative_PnL'], color='green')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Cumulative PnL (₹)')
        axes[0, 1].set_title('Cumulative PnL Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Win Rate by Hour
        hourly_analysis = self.analyze_by_time()
        if not hourly_analysis.empty:
            axes[1, 0].bar(hourly_analysis['Hour'], hourly_analysis['Win_Rate'])
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].set_title('Win Rate by Hour')
            axes[1, 0].set_xticks(hourly_analysis['Hour'])
        
        # 4. PnL by Symbol
        symbol_analysis = self.analyze_by_symbol()
        if not symbol_analysis.empty:
            top_symbols = symbol_analysis.head(10)
            colors = ['green' if pnl > 0 else 'red' for pnl in top_symbols['Total_PnL']]
            axes[1, 1].bar(top_symbols['Symbol'], top_symbols['Total_PnL'], color=colors)
            axes[1, 1].set_xlabel('Symbol')
            axes[1, 1].set_ylabel('Total PnL (₹)')
            axes[1, 1].set_title('Total PnL by Symbol (Top 10)')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, 'analysis_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Run the analysis
if __name__ == "__main__":
      # --- IMPORTANT ---
    # Update this variable with the name of your backtest results file.
    # The file is located in the 'static/downloads/' folder.
    backtest_filename = 'backtest_results_1968e0fd-2336-4b0f-82d1-183649425932.csv' # <-- CHANGE THIS
    
    file_path = os.path.join('static', 'downloads', backtest_filename)
    analyze_peak_capital(file_path)
    # ===================== CONFIG ===================== 
    ANALYSIS_OUTPUT_DIR = os.path.join('backtest_analysis')    
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    analyzer = BacktestAnalyzer(file_path)
    if analyzer.load_results():
        analyzer.analyze_performance()
        analyzer.create_visualizations()
     