# app.py - Main Flask Application
import sqlite3
import pandas as pd
from datetime import datetime
import threading
import time
import traceback
import uuid
import os
import io
import base64
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import traceback

from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from tradingview_screener import Query, col
from tvDatafeed import TvDatafeed, Interval
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt

# Import strategy and default config
from strategy_core import HighVolumeBreakoutStrategy
import config as default_config
# app.py - Main Flask Application
import sqlite3
import pandas as pd
from datetime import datetime
import threading
import time
import traceback
import uuid
import os
import io
import base64
import json
import pandas_ta as ta

from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from tradingview_screener import Query, col
from tvDatafeed import TvDatafeed, Interval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from strategy_core import HighVolumeBreakoutStrategy
import config as default_config

# --- App Initialization & Configuration ---
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
APP_CONFIG = {attr: getattr(default_config, attr) for attr in dir(default_config) if not callable(getattr(default_config, attr)) and not attr.startswith("__")}
JOBS_DB_FILE = 'backtest_jobs.json'

# --- Global State ---
tv = None

executor = ThreadPoolExecutor(max_workers=APP_CONFIG['MAX_CONCURRENT_SCANS'])
strategy = HighVolumeBreakoutStrategy(APP_CONFIG)

# ... (scanner-related global variables) ...

# --- Backtest Job Persistence ---
def load_jobs():
    """Loads backtest job history from a JSON file."""
    if not os.path.exists(JOBS_DB_FILE):
        return {}
    with open(JOBS_DB_FILE, 'r') as f:
        return json.load(f)

def save_jobs(jobs):
    """Saves the backtest job history to a JSON file."""
    with open(JOBS_DB_FILE, 'w') as f:
        json.dump(jobs, f, indent=4)

backtest_jobs = load_jobs()



current_opportunities = []

# Scanner state
scanner_thread = None
scanner_stop_event = threading.Event()
SCANNER_STATUS = "Stopped"

# Backtest state
backtest_jobs = {} # Stores status and results of backtests

# --- Utility Functions ---
def get_tv_client():
    """Get or create a TradingView client instance."""
    global tv
    if tv is None:
        tv = TvDatafeed(APP_CONFIG.get('TV_USERNAME'), APP_CONFIG.get('TV_PASSWORD'))
    return tv

def get_latest_opportunities_csv_path():
    """Generates the path for today's opportunities CSV file."""
    current_date = datetime.now().strftime('%d_%m_%y')
    return f'realtime_breakout_opportunities_{current_date}.csv'

def initialize_csv():
    """Initializes the CSV file with headers if it doesn't exist."""
    csv_file_path = get_latest_opportunities_csv_path()
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'name', 'price', 'volume', 'value_traded',
                         'avg_volume_10d', 'entry_price', 'stop_loss', 'target', 'position_size',
                         'risk_per_share', 'potential_pnl', 'confidence_score', 'breakout_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    return csv_file_path


# --- NEW: Integrated Pattern Analyzer ---

# app.py

# ... (other imports and code)

# --- NEW: Integrated Pattern Analyzer (Corrected) ---
class PatternAnalyzer:
    def __init__(self, df_trades, job_id):
        self.df = df_trades
        self.job_id = job_id
        if self.df.empty:
            self.is_empty = True
        else:
            self.is_empty = False

    def _generate_plot(self, df_analysis, column_name):
        # ... (this method is unchanged)
        if df_analysis.empty: return None
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # --- START: THIS LINE IS THE FIX for the Seaborn Warning ---
        # We assign the x-axis variable to 'hue' and disable the legend
        sns.barplot(x=df_analysis.index.astype(str), y=df_analysis['TotalPnL'], ax=ax1, alpha=0.7, palette="viridis", hue=df_analysis.index.astype(str), legend=False)
        # --- END: THIS LINE IS THE FIX ---
        #sns.barplot(x=df_analysis.index.astype(str), y=df_analysis['TotalPnL'], ax=ax1, alpha=0.7, palette="viridis")
        ax1.set_ylabel('Total PnL (₹)')
        ax1.tick_params(axis='x', rotation=45)
        ax2 = ax1.twinx()
        sns.lineplot(x=df_analysis.index.astype(str), y=df_analysis['WinRate'], ax=ax2, color='red', marker='o')
        ax2.set_ylabel('Win Rate', color='red')
        ax2.set_ylim(0, 1)
        for i, p in enumerate(ax1.patches):
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width() / 2., height + 0.1, f"n={df_analysis['TotalTrades'].iloc[i]}", ha="center")
        plt.title(f'Performance by {column_name}')
        fig.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    def _analyze_by_bin(self, column_name, num_bins=5):
        """Analyzes a continuous variable by binning."""
        # --- START: THIS SECTION IS THE FIX ---
        # Check for failure conditions first
        if column_name not in self.df.columns or self.df[column_name].nunique() < num_bins:
            # On failure, return None. This is unambiguous for the 'if' check.
            return None, None
        # --- END: THIS SECTION IS THE FIX ---

        try:
            self.df[f'{column_name}_Bin'] = pd.qcut(self.df[column_name], num_bins, duplicates='drop', labels=False)
            bin_labels = self.df.groupby(f'{column_name}_Bin')[column_name].agg(['min', 'max'])
            analysis = pd.DataFrame(self.df.groupby(f'{column_name}_Bin').agg(
                TotalTrades=('PnL', 'count'), TotalPnL=('PnL', 'sum'), WinRate=('PnL', lambda x: (x > 0).mean())
            ))
            analysis.index = analysis.index.map(lambda x: f'({bin_labels.loc[x, "min"]:.2f} - {bin_labels.loc[x, "max"]:.2f}]')
        except Exception:
            # If binning fails for any other reason, also return None
            return None, None

        plot_b64 = self._generate_plot(analysis, column_name)
        return analysis.to_html(classes='table table-sm table-striped'), plot_b64

    def _analyze_by_category(self, column_name):
        """Analyzes a categorical variable."""
        if column_name not in self.df.columns:
            return None, None
            
        analysis = pd.DataFrame(self.df.groupby(column_name).agg(
            TotalTrades=('PnL', 'count'), TotalPnL=('PnL', 'sum'), WinRate=('PnL', lambda x: (x > 0).mean())
        ))
        plot_b64 = self._generate_plot(analysis, column_name)
        return analysis.to_html(classes='table table-sm table-striped'), plot_b64

    def run_all_analyses(self):
        """Runs all predefined analyses and returns a dictionary of results."""
        if self.is_empty: return {}
        
        results = {}
        results['VolumeRatio'] = self._analyze_by_bin('VolumeRatio')
        results['ValueTraded'] = self._analyze_by_bin('ValueTraded')
        results['ConsolidationRatio'] = self._analyze_by_bin('ConsolidationRatio', num_bins=4)
        results['Direction'] = self._analyze_by_category('Direction')
        
        # This comprehension will now work correctly because `if v[0]` will be `if None` (which is False)
        # in failure cases, avoiding the ValueError.
        return {k: {'table': v[0], 'plot': v[1]} for k, v in results.items() if v[0] is not None}



# --- Backtester Logic ---
class Backtester:
    def __init__(self, config):
        self.config = config
        self.strategy = HighVolumeBreakoutStrategy(self.config)
        self.conn = sqlite3.connect(self.config['DB_FILE'], check_same_thread=False)

    def scan_for_breakouts(self):
        """Main scanning function that runs in a loop."""
        global current_opportunities, SCANNER_STATUS
        SCANNER_STATUS = "Running"
        print("Scanner thread started.")

        while not scanner_stop_event.is_set():
            try:
                query = (Query()
                        .set_markets('india')
                        .select('name', 'close', 'volume|1', 'Value.Traded|1', 'average_volume_10d_calc|1')
                        .where(
                            col('is_primary') == True,
                            col('typespecs').has('common'),
                            col('exchange') == 'NSE',
                            col('volume|1') > 10000,
                            col('close').between(2, 10000),
                            col('Value.Traded|1') > APP_CONFIG['MIN_VALUE_CR'] * 10000000)
                        .order_by('Value.Traded', ascending=False)
                        .limit(50))

                _, df = query.get_scanner_data()

                if df is not None and not df.empty:
                    df = df[df['volume|1'] > APP_CONFIG['VOLUME_MULTIPLIER'] * df['average_volume_10d_calc|1']]
                    df_renamed = df.rename(columns={
                        'ticker': 'symbol', 'close': 'price', 'volume|1': 'volume',
                        'Value.Traded|1': 'value_traded', 'average_volume_10d_calc|1': 'avg_volume_10d'
                    })

                    new_opportunities = []
                    for _, row in df_renamed.iterrows():
                        symbol = row['symbol']
                        if not symbol.startswith('BSE:'):
                            opportunity = self.analyze_breakout_opportunity(symbol, row.to_dict())
                            if opportunity:
                                new_opportunities.append(opportunity)

                    current_opportunities = sorted(new_opportunities, key=lambda x: x['confidence_score'], reverse=True)

                    if current_opportunities:
                        csv_file_path = initialize_csv()
                        with open(csv_file_path, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=current_opportunities[0].keys())
                            for opp in current_opportunities:
                                writer.writerow(opp)
                
                time.sleep(APP_CONFIG['SCAN_INTERVAL'])

            except Exception as e:
                print(f"Error in scan_for_breakouts loop: {e}")
                time.sleep(APP_CONFIG['SCAN_INTERVAL'])
        
        SCANNER_STATUS = "Stopped"
        print("Scanner thread stopped.")
    # --- Real-time Scanner Logic ---
    def analyze_breakout_opportunity(self,symbol, stock_data):
        """Analyze a potential breakout opportunity using the unified strategy logic."""
        try:
            tv_client = get_tv_client()
            exchange = 'NSE'
            data = tv_client.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=5000)

            if data is None or data.empty:
                return None

            df = data.reset_index()
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
            
            # ADDED: Calculate Price SMA for trend analysis

            df['price_sma_200'] = df['close'].rolling(200).mean()
            df[f'price_ma_{self.config["MA_FAST_PERIOD"]}'] = df['close'].rolling(self.config["MA_FAST_PERIOD"]).mean()
            df[f'price_ma_{self.config["MA_SLOW_PERIOD"]}'] = df['close'].rolling(self.config["MA_SLOW_PERIOD"]).mean()
            df['value_cr'] = (df['close'] * df['volume']) / 1e7
            df['vol_sma_200'] = df['volume'].rolling(200).mean()

            setup = strategy.analyze_setup(df, len(df)-1)
            if setup is None:
                return None

            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'name': stock_data.get('name', 'Unknown'),
                'price': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
                'value_traded': stock_data.get('value_traded', 0),
                'avg_volume_10d': stock_data.get('avg_volume_10d', 0),
                'entry_price': setup['entry_price'],
                'stop_loss': setup['stop_loss'],
                'target': setup['target'],
                'position_size': setup['position_size'],
                'risk_per_share': setup['entry_price'] - setup['stop_loss'] if setup['direction'] == "Long" else setup['stop_loss'] - setup['entry_price'],
                'potential_pnl': APP_CONFIG['TARGET_R_MULTIPLIER'] * APP_CONFIG['RISK_PER_TRADE'],
                'confidence_score': 80,
                'breakout_type': f"{setup['direction']} Breakout"
            }

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None


    def load_symbol_data(self, symbol, start_date, end_date):
        query = "SELECT datetime, open, high, low, close, volume FROM stock_data WHERE symbol = ? AND datetime >= ? AND datetime <= ?"
        params = [symbol, start_date, end_date]
        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['datetime'])
        if not df.empty:
            # ADDED: Calculate Price SMA for trend analysis
            df['price_sma_200'] = df['close'].rolling(200).mean()
            df[f'price_ma_{self.config["MA_FAST_PERIOD"]}'] = df['close'].rolling(self.config["MA_FAST_PERIOD"]).mean()
            df[f'price_ma_{self.config["MA_SLOW_PERIOD"]}'] = df['close'].rolling(self.config["MA_SLOW_PERIOD"]).mean()
            df['value_cr'] = (df['close'] * df['volume']) / 1e7
            df['vol_sma_200'] = df['volume'].rolling(200).mean()
        return df

    def backtest_symbol(self, symbol, start_date, end_date):
        df = self.load_symbol_data(symbol, start_date, end_date)
        
        if df.empty or len(df) < 200: return []
        
        # Pre-calculate indicators for the entire dataframe once
        df.ta.atr(length=self.config['ATR_PERIOD'], append=True)
        df[f'price_ma_{self.config["MA_FAST_PERIOD"]}'] = df['close'].rolling(self.config["MA_FAST_PERIOD"]).mean()
        df[f'price_ma_{self.config["MA_SLOW_PERIOD"]}'] = df['close'].rolling(self.config["MA_SLOW_PERIOD"]).mean()

        trades = []
        
        daily_trades = 0
        current_day = None
        for i in range(200, len(df)):
            current_date = df.iloc[i]['datetime'].date()
            if current_date != current_day:
                current_day = current_date
                daily_trades = 0

            if daily_trades >= self.config['MAX_DAILY_TRADES']:
                continue

            setup = self.strategy.analyze_setup(df, i)
            setup = self.strategy.analyze_setup(df, i)
            if setup:
                exit_price, exit_time, exit_reason, pnl = self.strategy.simulate_exit(df, i, setup)
                if exit_price is not None:
                    # --- MODIFIED: Expand the trade dictionary with setup data ---
                    trades.append({
                        'Symbol': symbol, 'EntryTime': df.iloc[i]['datetime'], 'ExitTime': exit_time,
                        'Direction': setup['direction'], 'EntryPrice': setup['entry_price'],
                        'ExitPrice': exit_price, 'PnL': pnl, 'ExitReason': exit_reason,
                        # Add the new columns for analysis
                        'VolumeRatio': setup['volume_ratio'],
                        'ValueTraded': setup['value_traded'],
                        'ConsolidationRatio': setup['consolidation_ratio'],
                        'IsBreakingSR': setup['is_breaking_sr']
                    })
                    daily_trades += 1
        return trades

    def run_backtest(self, job_id, symbols, start_date, end_date):
        try:
            backtest_jobs[job_id]['status'] = 'running'
            # --- NEW: Save the configuration used for this backtest run ---
            config_filename = f'backtest_config_{job_id}.json'
            config_path = os.path.join('static', 'downloads', config_filename)
            with open(config_path, 'w') as f:
                # Convert non-serializable items to strings if necessary
                serializable_config = {k: str(v) for k, v in self.config.items()}
                json.dump(serializable_config, f, indent=4)
            # --- End of new code ---
            
            if not symbols:
                query = "SELECT DISTINCT symbol FROM stock_data"
                df_symbols = pd.read_sql_query(query, self.conn)
                symbols = df_symbols['symbol'].tolist()

            all_trades = []
            for symbol in symbols:
                try:
                    trades = self.backtest_symbol(symbol, start_date, end_date)
                    if trades:
                        all_trades.extend(trades)
                except Exception as e:
                    print(f"Error backtesting {symbol}: {e}")
                    traceback.print_exc()
            
            # --- START: THIS SECTION IS THE FIX ---
            if not all_trades:
                print(f"Job {job_id} completed with 0 trades.")
                backtest_jobs[job_id]['status'] = 'completed'
                # Create a specific 'results' object for the "no trades" case
                backtest_jobs[job_id]['results'] = {
                    'summary_html': '<div class="alert alert-warning">No trades were found for the selected symbols and parameters.</div>',
                    'summary_plot': None,
                    'csv_path': None,
                    'pattern_analysis': {} # Empty analysis
                }
                save_jobs(backtest_jobs)
                return # Stop execution here
            # --- END: THIS SECTION IS THE FIX ---

            df_trades = pd.DataFrame(all_trades)
            summary_html, image_data, csv_path = self.generate_summary(df_trades, job_id)
            
            # --- NEW: Run and save automated analysis ---
            analyzer = PatternAnalyzer(df_trades, job_id)
            analysis_results = analyzer.run_all_analyses()
            
            backtest_jobs[job_id]['status'] = 'completed'
            backtest_jobs[job_id]['results'] = {
                'summary_html': summary_html,
                'summary_plot': image_data,
                'csv_path': csv_path,
                'pattern_analysis': analysis_results # Store the new analysis
            }
            print("Save JOBS BACKTEST RESULTS ")
            # print(backtest_jobs)
            save_jobs(backtest_jobs)

        except Exception as e:
            backtest_jobs[job_id]['status'] = 'failed'
            backtest_jobs[job_id]['error'] = str(e)
            traceback.print_exc()

    def generate_summary(self, df_trades, job_id):
        if df_trades.empty:
            return pd.DataFrame(), None, None

        total_trades = len(df_trades)
        win_rate = (df_trades['PnL'] > 0).sum() / total_trades if total_trades > 0 else 0
        total_pnl = df_trades['PnL'].sum()
        
        summary_data = {
            'Metric': ['Total Trades', 'Win Rate', 'Total PnL (₹)'],
            'Value': [total_trades, f"{win_rate:.2%}", f"{total_pnl:,.2f}"]
        }
        df_summary = pd.DataFrame(summary_data)

        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Backtest Results', fontsize=16)

        axes[0].hist(df_trades['PnL'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('PnL Distribution')
        
        df_trades_sorted = df_trades.sort_values('EntryTime')
        df_trades_sorted['Cumulative_PnL'] = df_trades_sorted['PnL'].cumsum()
        axes[1].plot(df_trades_sorted['EntryTime'], df_trades_sorted['Cumulative_PnL'], color='green')
        axes[1].set_title('Cumulative PnL Over Time')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Save detailed CSV
        # --- START: THIS SECTION IS THE FIX ---
        
        # 1. Define the web-accessible relative path for the template/JS
        csv_filename = f'backtest_results_{job_id}.csv'
        relative_csv_path = f'static/downloads/{csv_filename}'

        # 2. Create the full, absolute path for saving the file from the backend
        # app.root_path gives the absolute path to where app.py is located
        absolute_csv_path = os.path.join(app.root_path, relative_csv_path)

        # 3. Ensure the directory exists
        os.makedirs(os.path.dirname(absolute_csv_path), exist_ok=True)
        
        # 4. Save the file using the ABSOLUTE path
        df_trades.to_csv(absolute_csv_path, index=False)
        print(f"Successfully saved results to: {absolute_csv_path}")

        # 5. Return the summary, the plot, and the RELATIVE path for the web link
        print("Return the summary, the plot, and the RELATIVE path for the web link ")
        summaryHTML = df_summary.to_html(classes='table table-striped')
        print(summaryHTML)
        return summaryHTML, plot_url, relative_csv_path

        # --- END: THIS SECTION IS THE FIX ---

# --- Flask Routes ---

@app.route('/')
def route_index():
    return render_template('index.html', scanner_status=SCANNER_STATUS)

@app.route('/api/opportunities')
def api_get_opportunities():
    return jsonify(current_opportunities)

@app.route('/scanner/start')
def scanner_start():
    global scanner_thread, scanner_stop_event, SCANNER_STATUS
    if not scanner_thread or not scanner_thread.is_alive():
        scanner_stop_event.clear()
        backtester = Backtester(APP_CONFIG)
        scanner_thread = threading.Thread(target=backtester.scan_for_breakouts, daemon=True)
        scanner_thread.start()
        SCANNER_STATUS = "Running"
    return redirect(url_for('route_index'))

@app.route('/scanner/stop')
def scanner_stop():
    global SCANNER_STATUS
    scanner_stop_event.set()
    SCANNER_STATUS = "Stopped"
    return redirect(url_for('route_index'))

@app.route('/download_opportunities')
def download_opportunities():
    csv_path = get_latest_opportunities_csv_path()
    if os.path.exists(csv_path):
        return send_from_directory('.', csv_path, as_attachment=True)
    return "No opportunities file found for today.", 404

# --- Flask Routes (Updated and New) ---

@app.route('/backtest', methods=['GET'])
def route_backtest():
    # Pass sorted jobs to the template
    sorted_jobs = sorted(backtest_jobs.items(), key=lambda item: item[1]['start_time'], reverse=True)
    return render_template('backtest.html', jobs=sorted_jobs)

# ... (all other routes and code)

@app.route('/backtest/run', methods=['POST'])
def run_backtest_endpoint():
    # --- START: THIS LOGIC WAS MISSING AND IS NOW ADDED ---
    symbols_str = request.form.get('symbols')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    symbols = [s.strip().upper() for s in symbols_str.split(',')] if symbols_str else None
    # --- END: ADDED LOGIC ---

    job_id = str(uuid.uuid4())
    backtest_jobs[job_id] = {
        'status': 'queued', 
        'start_time': datetime.now().isoformat(),
        'params': {'symbols': symbols_str, 'start_date': start_date, 'end_date': end_date},
        'results': None,
        # --- ADDED: Also save a snapshot of the config with the job ---
        'config': {k: str(v) for k, v in APP_CONFIG.items()}
    }
    
    # Save config file for legacy analysis script compatibility (optional but good practice)
    config_filename = f'backtest_config_{job_id}.json'
    config_path = os.path.join('static', 'downloads', config_filename)
    with open(config_path, 'w') as f:
        json.dump(backtest_jobs[job_id]['config'], f, indent=4)

    save_jobs(backtest_jobs) # Save the master jobs list

    # --- START: THIS LOGIC WAS MISSING AND IS NOW ADDED ---
    # Instantiate a new backtester with the CURRENT app config for this specific run
    backtester = Backtester(APP_CONFIG.copy()) 
    thread = threading.Thread(target=backtester.run_backtest, args=(job_id, symbols, start_date, end_date))
    thread.start()
    # --- END: ADDED LOGIC ---
    
    return jsonify({'job_id': job_id})

# ... (rest of app.py)


@app.route('/backtest/status/<job_id>')
def backtest_status(job_id):
    """Polls for the status of a RUNNING job and returns full results upon completion."""
    # --- THIS IS THE FIX ---
    # Also make this endpoint stateless for consistency.
    all_jobs = load_jobs()
    job = all_jobs.get(job_id)
    # --- END OF FIX ---
    # job = backtest_jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)

@app.route('/api/backtest/<job_id>')
def api_get_backtest_details(job_id):
    """Gets the full details of a specific historical backtest job."""
    # --- THIS IS THE FIX ---
    # Also make this endpoint stateless for consistency.
    all_jobs = load_jobs()
    job = all_jobs.get(job_id)
    # --- END OF FIX ---
    # job = backtest_jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({'error': 'Job not found or not completed'}), 404
    return jsonify(job)

@app.route('/static/downloads/<filename>')
def download_file(filename):
    return send_from_directory(os.path.join('static', 'downloads'), filename, as_attachment=True)


@app.route('/settings', methods=['GET', 'POST'])
def route_settings():
    if request.method == 'POST':
        for key in APP_CONFIG:
            if key in request.form:
                # Try to cast to original type (int, float, str)
                original_value = APP_CONFIG[key]
                form_value = request.form[key]
                try:
                    if isinstance(original_value, bool):
                        APP_CONFIG[key] = form_value.lower() in ['true', '1', 'yes']
                    elif isinstance(original_value, int):
                        APP_CONFIG[key] = int(form_value)
                    elif isinstance(original_value, float):
                        APP_CONFIG[key] = float(form_value)
                    else:
                        APP_CONFIG[key] = form_value
                except ValueError:
                    # Keep original value if cast fails
                    pass
        # Re-initialize strategy with new config
        global strategy
        strategy = HighVolumeBreakoutStrategy(APP_CONFIG)
        return redirect(url_for('route_settings'))

    return render_template('settings.html', config=APP_CONFIG)

# app.py - Add this new route to your existing file

# ... (keep all the other code from the previous response)

def get_chart_data(symbol, n_bars=200):
    """Fetches and formats chart data for a given symbol."""
    try:
        tv_client = get_tv_client()
        exchange = 'NSE'
        data = tv_client.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=n_bars)

        if data is None or data.empty:
            return []
            
        # The charting library expects a UNIX timestamp in seconds (not milliseconds).
        # We also ensure the timestamp is in UTC as required.
        data.index = pd.to_datetime(data.index).tz_localize('Asia/Kolkata').tz_convert('UTC')
        
        chart_data = []
        for timestamp, row in data.iterrows():
            chart_data.append({
                'time': int(timestamp.timestamp()), # Convert to UNIX timestamp (seconds)
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume']) if not pd.isna(row['volume']) else 0
            })
        return chart_data
    except Exception as e:
        print(f"Error fetching chart data for {symbol}: {e}")
        return []

@app.route('/api/chart/<symbol>')
def api_get_chart_data(symbol):
    """API endpoint to get chart data for a specific symbol."""
    chart_data = get_chart_data(symbol)
    return jsonify(chart_data)


# ... (rest of the app.py file)

if __name__ == '__main__':
    initialize_csv()
    # MODIFIED: Add use_reloader=False to disable the auto-reloader
    app.run(debug=True, host='0.0.0.0', port=5000)