#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:34:33 2025

@author: charliewhite
"""

import matplotlib.pyplot as plt
import yfinance as yf 
import pandas as pd
import numpy as np

tickers = ['^IXIC', '^STOXX50E' ]
stock1 = tickers[0]
stock2 = tickers[1]

#yfinance blocked
def set_df(stock, start, end):

    ticker = [stock]

    data = yf.download(ticker, start=start, end=end)
    
    return data

#%% For Alpha Vantage

from alpha_vantage.timeseries import TimeSeries
import time

api_key = "2OFZI3DX24R6AVXJ"
ts = TimeSeries(key=api_key, output_format="pandas")

tickers = ['AAPL', 'AMZN']

def set_data(tickers, start=None, end=None):
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            # Get data
            ticker_data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            ticker_data.index = pd.to_datetime(ticker_data.index)
            ticker_data = ticker_data.sort_index()
            
            # Trim by date range
            if start:
                ticker_data = ticker_data[ticker_data.index >= pd.to_datetime(start)]
            if end:
                ticker_data = ticker_data[ticker_data.index <= pd.to_datetime(end)]
                
            # Concatenate
            data = pd.concat([data, ticker_data], axis=1)
            
            print(f"Fetched data for {ticker}: {len(ticker_data)} rows.")
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
        
        time.sleep(12)  # Safer: stay within rate limit (5 requests per min)

    data = data.ffill()
    return data

#%%

# Fetching historical stock data
data_1 = set_data([stock1], start='2022-01-01', end='2025-01-01')
data_2 = set_data([stock2], start='2022-01-01', end='2025-01-01')

data_1.columns = [col[0] if isinstance(col, tuple) else col for col in data_1.columns]
data_2.columns = [col[0] if isinstance(col, tuple) else col for col in data_2.columns]

print(data_1,data_2.head())

#%% Plot Bollinger bands

def plot_bollinger_bands(data, stock, window):
    plt.figure(figsize=(12, 6))
    
    # Plotting stock closing price
    plt.plot(data['4. close'], label='Close Price')

    # Calculating and plotting 20-day rolling average
    data['MA20'] = data['4. close'].rolling(window=50).mean()
    data['MA20'] = data['MA20'].dropna()
    plt.plot(data['MA20'], label= f'{window}-Day Rolling Average')

    # Calculating and plotting Bollinger Bands
    data['20D_MA'] = data['4. close'].rolling(window=50).mean()
    data['Upper_band'] = data['20D_MA'] + 2 * data['4. close'].rolling(window=window).std()
    data['Lower_band'] = data['20D_MA'] - 2 * data['4. close'].rolling(window=window).std()
    plt.plot(data['Upper_band'], label='Upper Bollinger Band', linestyle='--', color='red')
    plt.plot(data['Lower_band'], label='Lower Bollinger Band', linestyle='--', color='green')
    
    plt.title(f'Stock Data of {stock} with Rolling Average and Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
plot_bollinger_bands(data_1,stock1,50)
plot_bollinger_bands(data_2,stock2,50)

#%% Plot volatility using daily returns

# Calculating daily returns
data_1['Daily_Return'] = data_1['4. close'].pct_change()
data_2['Daily_Return'] = data_2['4. close'].pct_change()

# Calculating volatility
volatility_1 = data_1['Daily_Return'].std() * np.sqrt(252)
volatility_2 = data_2['Daily_Return'].std() * np.sqrt(252)

fig, axes = plt.subplots(2, 1, figsize=(20, 15))

# Plot volatility of each stock
axes[0].plot(data_1['Daily_Return'], label='Daily Return', alpha=0.5)
axes[0].set_title(f'Volatility {stock1}')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Return')
axes[0].legend()

axes[1].plot(data_2['Daily_Return'], label='Daily Return', alpha=0.5)
axes[1].set_title(f'Volatility of {stock2}')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Return')
axes[1].legend()

#%%  Define the momentum strategy

def calculate_momentum(data, period=1):
	return data['4. close'].pct_change(periods=period)

# Adding Momentum to DataFrame
data_1['Momentum'] = calculate_momentum(data_1)
data_2['Momentum'] = calculate_momentum(data_2)

data_1 = data_1.dropna(subset=['OBV', 'Momentum'])
data_2 = data_2.dropna(subset=['OBV', 'Momentum'])

def momentum_strategy(data,threshold_buy, threshold_sell):
    buy_signals = []
    sell_signals = []
    
    for i in range(len(data)):
        # Use momentum to decide trades
        if data['Momentum'][i] > threshold_buy:
            buy_signals.append(data['4. close'][i])
            sell_signals.append(np.nan)
        elif data['Momentum'][i] < - threshold_sell:
            sell_signals.append(data['4. close'][i])
            buy_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)  # No signal

    data['Buy Signals'] = buy_signals
    data['Sell Signals'] = sell_signals

    return data

data_1 = momentum_strategy(data_1, 0.02, 0.02)
data_2 = momentum_strategy(data_2, 0.01, 0.01)

# Create a figure and axis objects for two subplots
fig, axes = plt.subplots(2, 1, figsize=(20, 15))

# Plot for the first data
axes[0].plot(data_1['4. close'], label='Close Price', alpha=0.5)
axes[0].scatter(data_1.index, data_1['Buy Signals'], label='Buy Signal', marker='^', color='green')
axes[0].scatter(data_1.index, data_1['Sell Signals'], label='Sell Signal', marker='v', color='red')
axes[0].set_title(f'Momentum Trading Signals for {stock1}')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
axes[0].legend()

# Plot for the second data
axes[1].plot(data_2['4. close'], label='Close Price', alpha=0.5)
axes[1].scatter(data_2.index, data_2['Buy Signals'], label='Buy Signal', marker='^', color='green')
axes[1].scatter(data_2.index, data_2['Sell Signals'], label='Sell Signal', marker='v', color='red')
axes[1].set_title(f'Momentum Trading Signals for {stock2}')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price')
axes[1].legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.show()

#%% Backtest with the predefined momentum strategy (Strategy 2)

def backtest_with_momentum(portfolio, data):
    # Initialize position
    data['Position'] = 0

    # Use buy and sell signals for trading
    for i in range(1, len(data)):
        if not np.isnan(data['Buy Signals'].iloc[i]):
            data.iloc[i, data.columns.get_loc('Position')] = 1  # Long
        elif not np.isnan(data['Sell Signals'].iloc[i]):
            data.iloc[i, data.columns.get_loc('Position')] = -1  # Short
        else:
            data.iloc[i, data.columns.get_loc('Position')] = data.iloc[i - 1, data.columns.get_loc('Position')]  # Hold

    # Calculate returns
    data['Strategy Returns'] = data['Position'].shift(1) * data['4. close'].pct_change()

    # Portfolio and benchmark values
    data['Portfolio Value'] = portfolio * (1 + data['Strategy Returns']).cumprod()
    data['Benchmark Value'] = portfolio * (1 + data['4. close'].pct_change().cumsum())

    return data

data_1 = backtest_with_momentum(10000, data_1)
data_2 = backtest_with_momentum(10000, data_2)

print(data_1)

#%%

def plot_portfolio_value(data, stock):

    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data['Portfolio Value'], label='Momentum Strategy')
    plt.plot(data.index, data['Benchmark Value'], label='Benchmark (Buy & Hold)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(f'Portfolio Performance for {stock}')
    plt.legend()
    plt.show()

plot_portfolio_value(data_1,stock1)
plot_portfolio_value(data_2,stock2)

