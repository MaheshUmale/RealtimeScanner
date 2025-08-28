# strategy_core.py - Unified strategy logic
import numpy as np
from scipy import signal

import pandas as pd
import pandas_ta as ta

class HighVolumeBreakoutStrategy:
    def __init__(self, config):
        """Initialize the strategy with a configuration dictionary."""
        self.config = config

    def is_in_trading_window(self, time_str):
        return self.config['TRADING_START'] <= time_str < self.config['TRADING_END']
    
    def identify_support_resistance(self, df):
        lookback = self.config['SR_LOOKBACK']
        if len(df) < lookback: return [], []
        
        recent_data = df.tail(lookback).copy()
        highs = recent_data['high'].values
        lows = recent_data['low'].values

        # Find resistance levels
        strong_peak_prominence = np.std(highs) * 1.5
        strong_peaks, _ = signal.find_peaks(highs, distance=max(60, lookback // 6), prominence=strong_peak_prominence)
        resistances = [highs[i] for i in strong_peaks]
        
        # Find support levels
        strong_trough_prominence = np.std(lows) * 1.5
        strong_troughs, _ = signal.find_peaks(-lows, distance=max(60, lookback // 6), prominence=strong_trough_prominence)
        supports = [lows[i] for i in strong_troughs]
        
        return sorted(supports), sorted(resistances)

    def is_near_level(self, price, levels):
        tolerance = self.config['SR_TOLERANCE']
        return any(abs(price - level) / price < tolerance for level in levels) if levels else False

# strategy_core.py

# ... (imports and other methods remain the same)

    def analyze_setup(self, df, current_index):
        if current_index < 200: return None
            
        current_row = df.iloc[current_index]
        
        # --- ATR Calculation ---
        if 'ATRr_14' not in df.columns:
            df.ta.atr(length=14, append=True)

        # --- Time Window & Trend Filter ---
        current_time = current_row['datetime'].strftime('%H:%M')
        
        if current_time < "10:00":
            return None
        if not self.is_in_trading_window(current_time) or current_time > "11:00":
            return None

        is_bullish = current_row['close'] > current_row['open']
        is_bearish = current_row['close'] < current_row['open']
        
        is_uptrend = current_row['close'] > current_row['price_sma_200']
        if self.config.get('ENABLE_DUAL_MA_FILTER', True):
            fast_ma_col = f'price_ma_{self.config["MA_FAST_PERIOD"]}'
            slow_ma_col = f'price_ma_{self.config["MA_SLOW_PERIOD"]}'
            is_uptrend = current_row[fast_ma_col] > current_row[slow_ma_col]
        
        if (is_bullish and not is_uptrend) or (is_bearish and is_uptrend):
            return None
            
        # --- Volume & Value Checks ---
        volume_ratio = current_row['volume'] / current_row['vol_sma_200'] if current_row['vol_sma_200'] > 0 else 0
        if volume_ratio < self.config['VOLUME_MULTIPLIER'] or current_row['value_cr'] < self.config['MIN_VALUE_CR']:
            return None
            
        if not (is_bullish or is_bearish):
            return None

        # --- Consolidation & S/R Checks ---
        supports, resistances = self.identify_support_resistance(df.iloc[:current_index+1])
        recent_range = df['high'].iloc[current_index-20:current_index+1].max() - df['low'].iloc[current_index-20:current_index+1].min()
        avg_price = df['close'].iloc[current_index-20:current_index+1].mean()
        
        # --- THIS SECTION IS CORRECTED ---
        # 1. Calculate the numerical ratio first
        consolidation_ratio_value = recent_range / avg_price if avg_price > 0 else 0
        # 2. Then perform the boolean check for the logic
        in_consolidation = consolidation_ratio_value < 0.02
        # --- END OF CORRECTION ---
        
        breaking_resistance = is_bullish and self.is_near_level(current_row['high'], resistances)
        breaking_support = is_bearish and self.is_near_level(current_row['low'], supports)

        if not (in_consolidation and (breaking_resistance or breaking_support)):
            return None
            
        # --- Breakout Candle Volume Confirmation Filter ---
        if self.config.get('ENABLE_BREAKOUT_VOL_FILTER', True):
            lookback = self.config['BREAKOUT_VOL_LOOKBACK']
            multiplier = self.config['BREAKOUT_VOL_MULTIPLIER']
            if current_index > lookback:
                recent_avg_volume = df['volume'].iloc[current_index-lookback : current_index].mean()
                if current_row['volume'] < (recent_avg_volume * multiplier):
                    return None

        # --- Trigger & Trade Management ---
        entry_price = current_row['close']
        direction = "Long" if is_bullish else "Short"
        
        if self.config.get('ENABLE_ATR_STOP', True):
            atr_value = df['ATRr_14'].iloc[current_index]
            if pd.isna(atr_value): return None
            stop_loss = entry_price - (atr_value * self.config['ATR_MULTIPLIER']) if direction == "Long" else entry_price + (atr_value * self.config['ATR_MULTIPLIER'])
        else:
            stop_loss = current_row['low'] if direction == "Long" else current_row['high']
            
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0: return None
            
        position_size = int(self.config['RISK_PER_TRADE'] / risk_per_share)
        if position_size * entry_price > self.config['MAX_CAPITAL_PER_TRADE']:
            position_size = int(self.config['MAX_CAPITAL_PER_TRADE'] / entry_price)
        if position_size == 0: return None
            
        return {
            'entry_price': entry_price, 
            'stop_loss': stop_loss,
            'position_size': position_size, 
            'direction': direction,
            'volume_ratio': round(volume_ratio, 2),
            'value_traded': round(current_row['value_cr'], 2),
            # --- THIS LINE IS CORRECTED ---
            'consolidation_ratio': round(consolidation_ratio_value, 4), # Use the numerical value here
            # --- END OF CORRECTION ---
            'is_breaking_sr': (breaking_resistance or breaking_support)
        }
    



    def _find_last_swing_point(self, price_data, direction):
        """
        Helper to find the last swing low (for longs) or swing high (for shorts).
        """
        lookback = self.config['TRAILING_SWING_LOOKBACK']
        prominence_factor = self.config['TRAILING_SWING_PROMINENCE_FACTOR']
        
        # Ensure we have enough data
        if len(price_data) < lookback:
            return None
            
        recent_data = price_data.tail(lookback)
        
        if direction == 'Long':
            # Find swing lows by finding peaks in the inverted 'low' series
            prices = -recent_data['low'].values
            prominence = np.std(prices) * prominence_factor
            peaks, _ = signal.find_peaks(prices, distance=5, prominence=prominence)
            if len(peaks) > 0:
                # Return the actual low price of the last detected swing low
                return recent_data['low'].iloc[peaks[-1]]
        else: # Short
            # Find swing highs
            prices = recent_data['high'].values
            prominence = np.std(prices) * prominence_factor
            peaks, _ = signal.find_peaks(prices, distance=5, prominence=prominence)
            if len(peaks) > 0:
                # Return the high price of the last detected swing high
                return recent_data['high'].iloc[peaks[-1]]
                
        return None

    def simulate_exit(self, df, start_index, setup):
        initial_position_size = setup['position_size']
        remaining_position_size = initial_position_size
        pnl = 0.0
        exit_time = None
        exit_reason = "END"
        
        initial_risk_per_share = abs(setup['entry_price'] - setup['stop_loss'])
        price_targets = [
            setup['entry_price'] + (float(r) * initial_risk_per_share) if setup['direction'] == 'Long' else
            setup['entry_price'] - (float(r) * initial_risk_per_share)
            for r in self.config['PARTIAL_EXIT_TARGETS_R']
        ]

        targets_hit = [False] * len(price_targets)
        active_stop_loss = setup['stop_loss']

        for j in range(start_index + 1, len(df)):
            next_row = df.iloc[j]
            
            # --- NEW: Time Stop Logic ---
            if self.config.get('ENABLE_TIME_STOP', True) and not targets_hit[0]:
                bars_in_trade = j - start_index
                if bars_in_trade > self.config['TIME_STOP_BARS']:
                    exit_price = next_row['close']
                    pnl += ((exit_price - setup['entry_price']) * remaining_position_size if setup['direction'] == 'Long' else
                            (setup['entry_price'] - exit_price) * remaining_position_size)
                    exit_time, exit_reason, remaining_position_size = next_row['datetime'], "TIME_STOP", 0
                    break # Exit trade due to time stop
                    
            # --- Stop Loss Check ---
            if (setup['direction'] == 'Long' and next_row['low'] <= active_stop_loss) or \
               (setup['direction'] == 'Short' and next_row['high'] >= active_stop_loss):
                exit_price = active_stop_loss
                pnl += ((exit_price - setup['entry_price']) * remaining_position_size if setup['direction'] == 'Long' else
                        (setup['entry_price'] - exit_price) * remaining_position_size)
                exit_time, exit_reason, remaining_position_size = next_row['datetime'], "SL", 0
                break

            # --- Partial Profit Target Check ---
            for i in range(len(price_targets)):
                if not targets_hit[i]:
                    target_price = price_targets[i]
                    if (setup['direction'] == 'Long' and next_row['high'] >= target_price) or \
                       (setup['direction'] == 'Short' and next_row['low'] <= target_price):
                        size_to_exit = round(initial_position_size * self.config['PARTIAL_EXIT_SIZES'][i]) if i < len(price_targets) - 1 else remaining_position_size
                        size_to_exit = min(size_to_exit, remaining_position_size)

                        pnl += ((target_price - setup['entry_price']) * size_to_exit if setup['direction'] == 'Long' else
                                (setup['entry_price'] - target_price) * size_to_exit)
                        
                        remaining_position_size -= size_to_exit
                        targets_hit[i] = True
                        exit_reason, exit_time = f"PARTIAL_EXIT_{i+1}", next_row['datetime']
                        
                        # After first partial profit, move SL to breakeven
                        active_stop_loss = setup['entry_price']

                        if remaining_position_size <= 0: break
            if remaining_position_size <= 0: break

            # --- NEW: Market Structure Trailing Stop ---
            if self.config['ENABLE_TRAILING_STOP'] and targets_hit[0]: # Activate after first profit
                price_data_for_swing = df.iloc[:j+1]
                new_swing_stop = self._find_last_swing_point(price_data_for_swing, setup['direction'])
                
                if new_swing_stop:
                    if setup['direction'] == 'Long':
                        active_stop_loss = max(active_stop_loss, new_swing_stop) # Ensure stop only moves up
                    else: # Short
                        active_stop_loss = min(active_stop_loss, new_swing_stop) # Ensure stop only moves down
            
            # --- End of Day Cutoff ---
            if next_row['datetime'].strftime('%H:%M') >= self.config['EOD_CUTOFF']:
                exit_price = next_row['close']
                pnl += ((exit_price - setup['entry_price']) * remaining_position_size if setup['direction'] == 'Long' else
                        (setup['entry_price'] - exit_price) * remaining_position_size)
                exit_time, exit_reason, remaining_position_size = next_row['datetime'], "EOD", 0
                break

        if remaining_position_size > 0:
            last_row = df.iloc[-1]
            exit_price = last_row['close']
            pnl += ((exit_price - setup['entry_price']) * remaining_position_size if setup['direction'] == 'Long' else
                    (setup['entry_price'] - exit_price) * remaining_position_size)
            exit_time, exit_reason = last_row['datetime'], "END_OF_DATA"

        final_exit_price = pnl / initial_position_size + setup['entry_price'] if initial_position_size > 0 else setup['entry_price']
        return final_exit_price, exit_time, exit_reason, pnl