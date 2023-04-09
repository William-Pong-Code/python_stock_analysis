import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class SMA_Strategy:
    """
    A class for analyzing stock data using simple moving averages (SMA).
    """
    def __init__(self, ticker, start='2015-01-01', end="2022-01-01", sma_short=50, sma_long=200, stop_loss=True, stop_loss_percent=0.08):
        """
        ticker: stock code
        start: start day
        end: end day
        sma_short: eg.SMA50
        sma_long: eg.SMA200
        stop_loss: Set True if having a stop_loss_percent
        stop_loss_percent: loss percentage compared to the buying price
        """
        self.ticker = ticker
        self.start = start
        self.end = end
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.stop_loss = stop_loss
        self.stop_loss_percent = stop_loss_percent

    def download_data(self):
        """
        Download source data
        """
        return yf.download(self.ticker, start=self.start, end=self.end, progress=False).dropna()

    def buy_sell(self, data):
        """
        Buy and Sell signal
        """
        buy_signal = []
        sell_signal = []
        stop_loss_signal = []
        flag = -1
        stop_loss_price = 0
        stop_loss = self.stop_loss
        stop_loss_percent = self.stop_loss_percent
        for i in range(len(data)):
            if data[f'SMA{self.sma_short}'][i] > data[f'SMA{self.sma_long}'][i] and data['Ticker'][i] > stop_loss_price:
                stop_loss_signal.append(np.nan)
                if flag == 0:
                    buy_signal.append(data['Ticker'][i])
                    if stop_loss:
                        stop_loss_price = data['Ticker'][i] * (1 - stop_loss_percent)
                    sell_signal.append(np.nan)
                    flag = 1
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data[f'SMA{self.sma_short}'][i] < data[f'SMA{self.sma_long}'][i] or data['Ticker'][i] < stop_loss_price:
                if flag == 1:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Ticker'][i])
                    flag = -1
                    if data['Ticker'][i] < stop_loss_price:
                        stop_loss_signal.append(data['Ticker'][i])
                    else:
                        stop_loss_signal.append(np.nan)
                    stop_loss_price = 0
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
                    stop_loss_signal.append(np.nan)
                if data[f'SMA{self.sma_short}'][i] < data[f'SMA{self.sma_long}'][i]:
                    flag = 0
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
                stop_loss_signal.append(np.nan)
        self.data = data
        return buy_signal, sell_signal, stop_loss_signal

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
        plt.scatter(data.index, data['Stop_Loss'], label='Stop Loss', marker='x', color='red', linewidth=3)
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
        stop_loss = data['Stop_Loss'].dropna().reset_index(drop=False)
        merged_df = buy.join(sell, lsuffix='_Buy', rsuffix='_Sell')
        merged_df = merged_df[['Date_Buy', 'Buy_Signal_Price', 'Date_Sell', 'Sell_Signal_Price']].dropna()
        merged_df['Holding Days'] = merged_df['Date_Sell'] - merged_df['Date_Buy']
        merged_df['Profit'] = merged_df['Sell_Signal_Price'] - merged_df['Buy_Signal_Price']
        merged_df['RoR (%)'] = merged_df['Profit'] / merged_df['Buy_Signal_Price'] * 100
        merged_df['Win'] = (merged_df['Profit'] > 0).astype(int)
        merged_df['Lose'] = (merged_df['Profit'] <= 0).astype(int)
        merged_df['Stop Loss'] = merged_df['Date_Sell'].isin(stop_loss['Date']).astype(int)

        num_trades = len(merged_df)
        profitable_trades = merged_df['Win'].sum()
        average_holding_days = merged_df['Holding Days'].mean().days
        losing_trades = merged_df['Lose'].sum()
        total_profit = merged_df['Profit'].sum()
        average_profit = merged_df['Profit'].mean()
        average_ror = merged_df['RoR (%)'].mean()
        stop_loss = merged_df['Stop Loss'].sum()
        
        if num_trades > 0:
            win_probability = f'{round(profitable_trades / num_trades * 100, 2)} %'
            print(merged_df, end='\n\n')
        else:
            win_probability = '0 days'
            average_holding_days = 0

        print('Strategy: SMA')
        print(f"Number of trades: {num_trades}")
        print(f"Average holding period: {average_holding_days} days")
        print(f"Number of profitable trades: {profitable_trades}")
        print(f"Number of losing trades: {losing_trades}")
        print(f"Number of stop losses: {stop_loss}")
        print(f"Total profit: {total_profit:.2f}")
        print(f"Average profit: {average_profit:.2f}")
        print(f"Average RoR: {average_ror:.2f} %")
        print(f"Win rate: {win_probability}")     

    def show_performance(self):
        df = self.download_data()

        data = pd.DataFrame()
        data['Ticker'] = df['Adj Close']
        data[f'SMA{self.sma_short}'] = df['Adj Close'].rolling(window=self.sma_short).mean()
        data[f'SMA{self.sma_long}'] = df['Adj Close'].rolling(window=self.sma_long).mean()
        data['Buy_Signal_Price'], data['Sell_Signal_Price'], data['Stop_Loss']  = self.buy_sell(data)

        self.analyze_performance()
        self.plot_data()
