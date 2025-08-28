# config.py - Centralized default configuration
import os

# --- Essential Paths ---
# It's recommended to use absolute paths or well-defined relative paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_FILE = os.path.join(BASE_DIR, "stock_data.db")
BACKTEST_CHARTS_DIR = os.path.join(BASE_DIR, "static", "backtest_charts")

# --- Strategy Parameters ---
VOLUME_MULTIPLIER = 5.0
MIN_VALUE_CR = 7.0
TARGET_R_MULTIPLIER = 8.0
RISK_PER_TRADE = 10000.0
MAX_CAPITAL_PER_TRADE = 200000.0
MAX_DAILY_TRADES = 3
EOD_CUTOFF = "15:15"
TRADING_START = "09:15"
TRADING_END = "15:15"

# --- Support/Resistance Parameters ---
SR_LOOKBACK = 50
SR_TOLERANCE = 0.015

# --- Real-time Scanning ---
SCAN_INTERVAL = 60  # seconds
MAX_CONCURRENT_SCANS = 5

# --- TradingView Credentials (Optional) ---
# For better reliability, set these as environment variables
TV_USERNAME = os.getenv("TV_USERNAME", "")
TV_PASSWORD = os.getenv("TV_PASSWORD", "")

# --- Create necessary directories ---
os.makedirs(os.path.join(BASE_DIR, "static", "downloads"), exist_ok=True)
os.makedirs(BACKTEST_CHARTS_DIR, exist_ok=True)


# --- Partial Profit & Trailing Stop Parameters ---
PARTIAL_EXIT_TARGETS_R = [2.0, 6.0] 
PARTIAL_EXIT_SIZES = [0.5, 1.0]

# --- NEW: Market Structure Trailing Stop ---
ENABLE_TRAILING_STOP = True
# How many bars to look back to find the last swing point.
TRAILING_SWING_LOOKBACK = 50
# How significant a swing must be to be considered. A higher number means the swing must be larger.
# This is a factor multiplied by the standard deviation of prices in the lookback period.
TRAILING_SWING_PROMINENCE_FACTOR = 1.0 # Lower for more sensitive, higher for less sensitive.


# --- NEW: ATR Stop Loss Configuration ---
ENABLE_ATR_STOP = True
# The lookback period for the ATR calculation. 14 is standard.
ATR_PERIOD = 14
# The multiplier for the ATR value. 1.5-2.5 is a common range.
# A higher number means a wider, more conservative stop loss.
ATR_MULTIPLIER = 2.0

# --- NEW: Enhanced Trend Filter Configuration ---
ENABLE_DUAL_MA_FILTER = True
MA_FAST_PERIOD = 9
MA_SLOW_PERIOD = 21


# --- NEW: Time Stop Configuration ---
ENABLE_TIME_STOP = True
# Exit a trade if it hasn't hit the first partial target after this many bars (minutes).
TIME_STOP_BARS = 30 # Exit after 1 hour if not yet profitable


# --- NEW: Opening Range Breakout (ORB) Filter ---
ENABLE_ORB_FILTER = True
# Defines the duration of the opening range in minutes (e.g., 30, 45, 60).
ORB_PERIOD_MINUTES = 30

# --- NEW: Breakout Volume Confirmation ---
# The volume of the breakout candle must be > (BREAKOUT_VOL_MULTIPLIER * avg volume of last N candles)
ENABLE_BREAKOUT_VOL_FILTER = True
BREAKOUT_VOL_LOOKBACK = 10 # Look at the average volume of the last 10 candles
BREAKOUT_VOL_MULTIPLIER = 1.5 # Breakout candle volume must be at least 1.5x the recent average

