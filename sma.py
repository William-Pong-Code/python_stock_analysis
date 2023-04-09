import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class SMAAnalyzer:
    """
    A class for analyzing stock data using simple moving averages (SMA).
    """
    def __init__(self, ticker, start='2015-01-01', end="2022-01-01", sma_short=50, sma_long=200):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.sma_short = sma_short
        self.sma_long = sma_long

    def download_data(self):
        """
        Download source data
        """
        return yf.download(self.ticker, start=self.start, end=self.end, progress=False).dropna()

    def buy_sell(self, data):
        """
        Buy and Sell signal
        """
        signal_Buy = []
        signal_Sell = []
        flag = -1
        for i in range(len(data)):
            if(data[f'SMA{self.sma_short}'][i] > data[f'SMA{self.sma_long}'][i]):
                if(flag != 1):
                    signal_Buy.append(data['Ticker'][i])
                    signal_Sell.append(np.nan)
                    flag = 1
                else:
                    signal_Buy.append(np.nan)
                    signal_Sell.append(np.nan)
            elif(data[f'SMA{self.sma_short}'][i] < data[f'SMA{self.sma_long}'][i]):
                if(flag != -1 and flag != 0):
                    signal_Buy.append(np.nan)
                    signal_Sell.append(data['Ticker'][i])
                    flag = 0
                else:
                    signal_Buy.append(np.nan)
                    signal_Sell.append(np.nan)
            else:
                signal_Buy.append(np.nan)
                signal_Sell.append(np.nan)
        self.data = data
        return signal_Buy, signal_Sell

    def plot_data(self):
        """
        Plots the stock data and the SMA strategy.
        """
        data = self.data
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12.5, 8))
        plt.plot(data['Ticker'], label=self.ticker.upper(), alpha=0.45)
        plt.plot(data[f'SMA{self.sma_short}'], label=f'SMA{self.sma_short}', alpha=0.45)
        plt.plot(data[f'SMA{self.sma_long}'], label=f'SMA{self.sma_long}', alpha=0.45)
        plt.scatter(data.index, data['Buy_Signal_Price'], label='Buy', marker='^', color='green')
        plt.scatter(data.index, data['Sell_Signal_Price'], label='Sell', marker='^', color='red')
        plt.legend(loc='upper left')
        plt.title(f'{self.ticker.upper()} Adj Close Price & Signals')
        plt.xlabel('DateTime')
        plt.ylabel('Adj Close Price')
        plt.show()

    def analyze_performance(self):
        """
        Analyzes the performance of the SMA strategy.
        """
        data = self.data
        buy = data['Buy_Signal_Price'].dropna().reset_index(drop=False)
        sell = data['Sell_Signal_Price'].dropna().reset_index(drop=False)
        merged_df = buy.join(sell, lsuffix='_Buy', rsuffix='_Sell')
        merged_df = merged_df[['Date_Buy', 'Buy_Signal_Price', 'Date_Sell', 'Sell_Signal_Price']].dropna()
        merged_df['Holding Days'] = merged_df['Date_Sell'] - merged_df['Date_Buy']
        merged_df['Profit'] = merged_df['Sell_Signal_Price'] - merged_df['Buy_Signal_Price']
        merged_df['Win'] = (merged_df['Profit'] > 0).astype(int)
        merged_df['Lose'] = (merged_df['Profit'] <= 0).astype(int)
        
        num_trades = len(merged_df)
        profitable_trades = merged_df['Win'].sum()
        average_holding_days = merged_df['Holding Days'].mean().days
        losing_trades = merged_df['Lose'].sum()
        total_profit = merged_df['Profit'].sum() 
        win_probability = f'{round(profitable_trades / num_trades * 100, 2)} %'

        print(merged_df, end='\n\n')
        print(f"Number of trades: {num_trades}")
        print(f"Average holding period: {average_holding_days} days")
        print(f"Number of profitable trades: {profitable_trades}")
        print(f"Number of losing trades: {losing_trades}")
        print(f"Total profit: {total_profit:.2f}")
        print(f"Win Probability: {win_probability}")     

    def show_performance(self):
        df = self.download_data()

        data = pd.DataFrame()
        data['Ticker'] = df['Adj Close']
        data[f'SMA{self.sma_short}'] = df['Adj Close'].rolling(window=self.sma_short).mean()
        data[f'SMA{self.sma_long}'] = df['Adj Close'].rolling(window=self.sma_long).mean()
        data['Buy_Signal_Price'], data['Sell_Signal_Price']  = self.buy_sell(data)

        self.analyze_performance()
        self.plot_data()