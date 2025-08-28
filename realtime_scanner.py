# realtime_scanner.py - Real-time scanning using the unified strategy logic
from flask import Flask, render_template, jsonify, request, send_from_directory
from tradingview_screener import Query, col
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from strategy_core import HighVolumeBreakoutStrategy
from config import *
import csv

app = Flask(__name__)

# Global variables
current_opportunities = []
tv = None
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SCANS)
strategy = HighVolumeBreakoutStrategy()

def initialize_csv():
    """Initialize the CSV file with headers if it doesn't exist"""
    current_DATE = datetime.now().strftime('%d_%m_%y')
    csv_file_path = 'realtime_breakout_opportunities_' + str(current_DATE) + '.csv'
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'name', 'price', 'volume', 'value_traded', 
                         'avg_volume_10d', 'entry_price', 'stop_loss', 'target', 'position_size',
                         'risk_per_share', 'potential_pnl', 'confidence_score', 'breakout_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    return csv_file_path

def get_tv_client():
    """Get or create TradingView client"""
    global tv
    if tv is None:
        tv = TvDatafeed(TV_USERNAME or None, TV_PASSWORD or None)
    return tv

def get_chart_data(symbol, n_bars=100):
    """Fetch chart data for a symbol using tvDatafeed"""
    try:
        tv_client = get_tv_client()
        exchange = 'NSE' if symbol.endswith('-EQ') or symbol.endswith('-NS') else 'NSE'
        
        data = tv_client.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=n_bars)
        
        if data is not None and not data.empty:
            # Convert index to Unix timestamp in milliseconds
            data.index = pd.to_datetime(data.index)
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            data.index = data.index.map(lambda x: int(x.timestamp() * 1000))
            
            # Convert to list of dictionaries
            chart_data = []
            for index, row in data.iterrows():
                chart_data.append({
                    'time': index,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']) if not pd.isna(row['volume']) else 0
                })
            return chart_data
        else:
            return []
    except Exception as e:
        print(f"Error fetching chart data for {symbol}: {e}")
        return []

def analyze_breakout_opportunity(symbol, stock_data):
    """Analyze a potential breakout opportunity using the unified strategy logic"""
    try:
        tv_client = get_tv_client()
        exchange = 'NSE' if symbol.endswith('-EQ') or symbol.endswith('-NS') else 'NSE'
        
        # Fetch 5000 1-minute bars for detailed analysis
        data = tv_client.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=5000)
        
        if data is None or data.empty:
            return None
            
        df = data.reset_index()
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        
        # Add derived columns
        df['value_cr'] = (df['close'] * df['volume']) / 1e7  # Value in Crores
        df['vol_sma_200'] = df['volume'].rolling(200).mean()
        
        # Use the unified strategy logic to analyze setup
        setup = strategy.analyze_setup(df, len(df)-1)
        
        if setup is None:
            return None
            
        # Create opportunity dictionary
        opportunity = {
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
            'potential_pnl': TARGET_R_MULTIPLIER * RISK_PER_TRADE,
            'confidence_score': 80,  # Could be calculated based on setup quality
            'breakout_type': f"{setup['direction']} Breakout"
        }
        
        return opportunity
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def scan_for_breakouts():
    """Scan for breakout opportunities using TradingView Screener"""
    global current_opportunities
    
    try:
        # Create query for high-volume breakouts
        query = (Query()
                .set_markets('india')
                .select(
                    'name',
                    'close',
                    'volume|1',
                    'Value.Traded|1',
                    'average_volume_10d_calc|1',
                    'relative_volume_10d_calc'
                )
                .where(
                    col('is_primary') == True,
                    col('typespecs').has('common'),
                    col('type') == 'stock',
                    col('exchange') == 'NSE',
                    col('volume|1') > 10000,  # Minimum volume
                    col('close').between(2, 10000),  # Price range
                    col('active_symbol') == True,
                    col('Value.Traded|1') > MIN_VALUE_CR * 10000000,  # Value traded > 7 Crore
                )
                .order_by('Value.Traded', ascending=False)
                .limit(50)
                .set_property('preset', 'volume_leaders'))
        
        # Execute query
        _, df = query.get_scanner_data()
        
        if df.empty:
            return
            
        # Filter for high volume ratio
        df = df[df['volume|1'] > VOLUME_MULTIPLIER * df['average_volume_10d_calc|1']]
        
        if df.empty:
            return
            
        # Rename columns
        df_renamed = df.rename(columns={
            'ticker': 'symbol', 
            'close': 'price', 
            'volume|1': 'volume', 
            'Value.Traded|1': 'value_traded',
            'average_volume_10d_calc|1': 'avg_volume_10d'
        })
        
        # Analyze each potential opportunity
        new_opportunities = []
        
        for _, row in df_renamed.iterrows():
            symbol = row['symbol']
            if not symbol.startswith('BSE:'):
                opportunity = analyze_breakout_opportunity(symbol, row.to_dict())
                if opportunity:
                    new_opportunities.append(opportunity)
        
        # Update global opportunities
        current_opportunities = sorted(new_opportunities, 
                                    key=lambda x: x['confidence_score'], 
                                    reverse=True)
        
        # Save to CSV
        csv_file_path = initialize_csv()
        for opp in current_opportunities:
            with open(csv_file_path, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'symbol', 'name', 'price', 'volume', 'value_traded', 
                            'avg_volume_10d', 'entry_price', 'stop_loss', 'target', 'position_size',
                            'risk_per_share', 'potential_pnl', 'confidence_score', 'breakout_type']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(opp)
            
    except Exception as e:
        print(f"Error in scan_for_breakouts: {e}")

def start_scanning():
    """Start the scanning loop"""
    while True:
        scan_for_breakouts()
        time.sleep(SCAN_INTERVAL)

# Initialize CSV
csv_file_path = initialize_csv()

# Start scanning in background thread
scanning_thread = threading.Thread(target=start_scanning, daemon=True)
scanning_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/opportunities')
def get_opportunities():
    """API endpoint to get current breakout opportunities"""
    return jsonify(current_opportunities)

@app.route('/api/chart/<symbol>')
def get_chart(symbol):
    """API endpoint to get chart data for a specific symbol"""
    n_bars = request.args.get('n_bars', default=100, type=int)
    chart_data = get_chart_data(symbol, n_bars)
    return jsonify(chart_data)

@app.route('/download_opportunities')
def download_opportunities():
    """Download opportunities CSV"""
    return send_from_directory('.', csv_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)