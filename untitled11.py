# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 23:34:37 2025

@author: Asus
"""

import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta
import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Streamlit page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Helper functions
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date, interval):
    """Fetches stock data using yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
        data['date'] = pd.to_datetime(data[date_col])
        data.drop(columns=[date_col], inplace=True)
        data.columns = data.columns.str.lower()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None
def calculate_profit(ticker, interval, start_date, end_date, indicators):
    """İndikatör sinyallerine göre alım-satım yaparak kazancı hesaplar."""
    # Veriyi çek
    data = fetch_stock_data(ticker, start_date, end_date, interval)
    if data is None:
        st.warning("No data found for the selected date range.")
        return

    # Sinyalleri saklamak için DataFrame
    data['buy_signal'] = False
    data['sell_signal'] = False

    # Her indikatör için sinyalleri topla
    for indicator in indicators:
        if indicator == 'SMA':
            fast_n = 8  # Varsayılan değerler
            short_n = 21
            data['sma_fast'] = data['close'].rolling(fast_n).mean()
            data['sma_short'] = data['close'].rolling(short_n).mean()
            data['sma_cross_up'] = (data['sma_fast'] > data['sma_short']) & (data['sma_fast'].shift(1) <= data['sma_short'].shift(1))
            data['sma_cross_down'] = (data['sma_fast'] < data['sma_short']) & (data['sma_fast'].shift(1) >= data['sma_short'].shift(1))
            data['buy_signal'] |= data['sma_cross_up']
            data['sell_signal'] |= data['sma_cross_down']

        elif indicator == 'EMA':
            fast_n = 8
            short_n = 21
            data['ema_fast'] = data['close'].ewm(span=fast_n, adjust=False).mean()
            data['ema_short'] = data['close'].ewm(span=short_n, adjust=False).mean()
            data['ema_cross_up'] = (data['ema_fast'] > data['ema_short']) & (data['ema_fast'].shift(1) <= data['ema_short'].shift(1))
            data['ema_cross_down'] = (data['ema_fast'] < data['ema_short']) & (data['ema_fast'].shift(1) >= data['ema_short'].shift(1))
            data['buy_signal'] |= data['ema_cross_up']
            data['sell_signal'] |= data['ema_cross_down']

        elif indicator == 'SuperTrend':
            atr_period = 10
            multiplier = 3
            change_atr = True
            def supertrend(df, atr_period, multiplier, change_atr):
                df = df.copy()
                df['hl2'] = (df['high'] + df['low']) / 2
                df['tr'] = np.maximum.reduce([
                    df['high'] - df['low'],
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1))
                ])
                df['atr'] = df['tr'].ewm(span=atr_period, adjust=False).mean() if change_atr else df['tr'].rolling(window=atr_period).mean()
                df['upper_band'] = df['hl2'] - (multiplier * df['atr'])
                df['lower_band'] = df['hl2'] + (multiplier * df['atr'])
                df['upper_band_final'] = 0.0
                df['lower_band_final'] = 0.0
                df['trend'] = 1
                df['buy_signal_st'] = False
                df['sell_signal_st'] = False
                df['upper_band_final'].iloc[0] = df['upper_band'].iloc[0]
                df['lower_band_final'].iloc[0] = df['lower_band'].iloc[0]
                for i in range(1, len(df)):
                    prev_upper = df['upper_band_final'].iloc[i-1]
                    current_upper = df['upper_band'].iloc[i]
                    df['upper_band_final'].iloc[i] = max(current_upper, prev_upper) if df['close'].iloc[i-1] > prev_upper else current_upper
                    prev_lower = df['lower_band_final'].iloc[i-1]
                    current_lower = df['lower_band'].iloc[i]
                    df['lower_band_final'].iloc[i] = min(current_lower, prev_lower) if df['close'].iloc[i-1] < prev_lower else current_lower
                    prev_trend = df['trend'].iloc[i-1]
                    df['trend'].iloc[i] = 1 if prev_trend == -1 and df['close'].iloc[i] > df['lower_band_final'].iloc[i-1] else \
                                         -1 if prev_trend == 1 and df['close'].iloc[i] < df['upper_band_final'].iloc[i-1] else prev_trend
                    df['buy_signal_st'].iloc[i] = df['trend'].iloc[i] == 1 and df['trend'].iloc[i-1] == -1
                    df['sell_signal_st'].iloc[i] = df['trend'].iloc[i] == -1 and df['trend'].iloc[i-1] == 1
                return df
            data = supertrend(data, atr_period, multiplier, change_atr)
            data['buy_signal'] |= data['buy_signal_st']
            data['sell_signal'] |= data['sell_signal_st']

        elif indicator == 'Tillson T3':
            length1 = 8
            a1 = 0.7
            def calculate_t3(df, length, volume_factor):
                input_series = (df['high'] + df['low'] + 2 * df['close']) / 4
                e1 = input_series.ewm(span=length, adjust=False).mean()
                e2 = e1.ewm(span=length, adjust=False).mean()
                e3 = e2.ewm(span=length, adjust=False).mean()
                e4 = e3.ewm(span=length, adjust=False).mean()
                e5 = e4.ewm(span=length, adjust=False).mean()
                e6 = e5.ewm(span=length, adjust=False).mean()
                c1 = -volume_factor ** 3
                c2 = 3 * volume_factor ** 2 + 3 * volume_factor ** 3
                c3 = -6 * volume_factor ** 2 - 3 * volume_factor - 3 * volume_factor ** 3
                c4 = 1 + 3 * volume_factor + volume_factor ** 3 + 3 * volume_factor ** 2
                return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
            data['t3'] = calculate_t3(data, length1, a1)
            data['t3_buy'] = (data['t3'] > data['t3'].shift(1)) & (data['t3'].shift(1) <= data['t3'].shift(2))
            data['t3_sell'] = (data['t3'] < data['t3'].shift(1)) & (data['t3'].shift(1) >= data['t3'].shift(2))
            data['buy_signal'] |= data['t3_buy']
            data['sell_signal'] |= data['t3_sell']

        elif indicator == 'MACD':
            short_ema = 12
            long_ema = 26
            signal_period = 9
            data['ema_short'] = data['close'].ewm(span=short_ema, adjust=False).mean()
            data['ema_long'] = data['close'].ewm(span=long_ema, adjust=False).mean()
            data['macd'] = data['ema_short'] - data['ema_long']
            data['signal_line'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
            data['macd_cross_up'] = (data['macd'] > data['signal_line']) & (data['macd'].shift(1) <= data['signal_line'].shift(1))
            data['macd_cross_down'] = (data['macd'] < data['signal_line']) & (data['macd'].shift(1) >= data['signal_line'].shift(1))
            data['buy_signal'] |= data['macd_cross_up']
            data['sell_signal'] |= data['macd_cross_down']

        elif indicator == 'RSI':
            period = 14
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            data['rsi_oversold'] = data['rsi'] <= 30
            data['rsi_overbought'] = data['rsi'] >= 70
            data['buy_signal'] |= data['rsi_oversold']
            data['sell_signal'] |= data['rsi_overbought']

    # Alım-satım simülasyonu
    initial_capital = 1000  # Başlangıç sermayesi
    capital = initial_capital
    position = 0  # Sahip olunan hisse sayısı
    trades = []  # İşlem geçmişi
    first_buy_found = False

    for i in range(len(data)):
        if data['buy_signal'].iloc[i] and not first_buy_found:
            # İlk buy sinyali
            position = capital / data['close'].iloc[i]  # Tüm sermaye ile alım
            capital = 0
            first_buy_found = True
            trades.append({
                'date': data['date'].iloc[i],
                'type': 'Buy',
                'price': data['close'].iloc[i],
                'shares': position,
                'capital': capital
            })
        elif first_buy_found:
            if data['buy_signal'].iloc[i] and position == 0:
                # Yeni alım sinyali
                position = capital / data['close'].iloc[i]
                capital = 0
                trades.append({
                    'date': data['date'].iloc[i],
                    'type': 'Buy',
                    'price': data['close'].iloc[i],
                    'shares': position,
                    'capital': capital
                })
            elif data['sell_signal'].iloc[i] and position > 0:
                # Satım sinyali
                capital = position * data['close'].iloc[i]
                trades.append({
                    'date': data['date'].iloc[i],
                    'type': 'Sell',
                    'price': data['close'].iloc[i],
                    'shares': position,
                    'capital': capital
                })
                position = 0

    # Son pozisyonu kapat (eğer açıksa)
    if position > 0:
        capital = position * data['close'].iloc[-1]
        trades.append({
            'date': data['date'].iloc[-1],
            'type': 'Sell (Close)',
            'price': data['close'].iloc[-1],
            'shares': position,
            'capital': capital
        })

    # Sonuçları göster
    final_capital = capital
    profit = final_capital - initial_capital
    profit_percentage = (profit / initial_capital) * 100

    st.subheader("Trading Performance")
    st.write(f"**Initial Capital**: {initial_capital:.2f}")
    st.write(f"**Final Capital**: {final_capital:.2f}")
    st.write(f"**Profit**: {profit:.2f}")
    st.write(f"**Profit Percentage**: {profit_percentage:.2f}%")
    st.write("**Trade History**:")
    trades_df = pd.DataFrame(trades)
    trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(trades_df)

    # Grafik: Sermaye değişimi
    capital_history = []
    current_capital = initial_capital
    current_position = 0
    for i in range(len(data)):
        if i in trades_df.index:
            trade = trades_df.iloc[i]
            if trade['type'].startswith('Buy'):
                current_position = trade['shares']
                current_capital = 0
            elif trade['type'].startswith('Sell'):
                current_capital = trade['capital']
                current_position = 0
        total_value = current_capital + (current_position * data['close'].iloc[i])
        capital_history.append(total_value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=capital_history, mode='lines', name='Portfolio Value', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'] * (initial_capital / data['close'].iloc[0]), mode='lines', name='Buy & Hold', line=dict(color='green', dash='dash')))
    fig.update_layout(
        title=f'{ticker} Portfolio Value vs Buy & Hold',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# Teknik analiz fonksiyonuna kazanç hesaplama entegrasyonu
def technical_analysis(ticker, interval, start_date, end_date):
    """Performs stock analysis with multiple selected technical indicators and signals."""
    indicators = st.sidebar.multiselect(
        'Indicators',
        options=[
            'SMA', 'EMA', 'SuperTrend', 'Fibonacci', 'Inverse Fisher Transform (STOCH,RSI,CCI)',
            'Tillson T3', 'MACD', 'RSI', 'Momentum', 'ATR', 'Bollinger Bands', 'SALMA'
        ],
        default=['SMA']
    )

    # Mevcut teknik analiz kodunu çalıştır
    # ... (orijinal technical_analysis kodunuz buraya gelecek)

    # Kazanç hesaplama butonu
    if st.sidebar.button('Calculate Profit'):
        calculate_profit(ticker, interval, start_date, end_date, indicators)