import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class Bollinger_Band:
    """
    A class for analyzing stock data using bb.
    """
    def __init__(self, ticker, start='2015-01-01', end="2022-01-01", sma=20, st=2, stop_loss=True, stop_loss_percent=0.08):
        """
        ticker: stock code
        start: start day
        end: end day
        sma: default: SMA20
        st: default: 2 standard deviation
        stop_loss: Set True if having a stop_loss_percent
        stop_loss_percent: loss percentage compared to the buying price
        """
        self.ticker = ticker
        self.start = start
        self.end = end
        self.sma = sma
        self.st = st
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
            if data['Ticker'][i] <= data['bollinger_down'][i] and data['Ticker'][i] > stop_loss_price:
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
            elif data['Ticker'][i] >= data['bollinger_up'][i] or data['Ticker'][i] < stop_loss_price:
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
                if data['Ticker'][i] >= data['bollinger_down'][i]:
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
        plt.figure(figsize=(12.5, 8))
        plt.plot(data['Ticker'], label=self.ticker.upper(), alpha=0.45)
        plt.plot(data['bollinger_up'], label='Bollinger Up', c='gray')
        plt.plot(data['bollinger_down'], label='Bollinger Down', c='gray')
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
        Analyzes the performance of the strategy.
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

        print('Strategy: Bollinger Band')
        print(f"Number of trades: {num_trades}")
        print(f"Average holding period: {average_holding_days} days")
        print(f"Number of profitable trades: {profitable_trades}")
        print(f"Number of losing trades: {losing_trades}")
        print(f"Number of stop losses: {stop_loss}")
        print(f"Total profit: {total_profit:.2f}")
        print(f"Average profit: {average_profit:.2f}")
        print(f"Average RoR: {average_ror:.2f} %")
        print(f"Win rate: {win_probability}")     
    
    def get_bollinger_bands(self, prices):
        
        def get_sma(prices, rate):
            return prices.rolling(rate).mean()
        
        sma = get_sma(prices, self.sma)
        std = prices.rolling(self.sma).std()
        bollinger_up = sma + std * self.st
        bollinger_down = sma - std * self.st
        return bollinger_up, bollinger_down

    def show_performance(self):
        df = self.download_data()

        data = pd.DataFrame()
        data['Ticker'] = df['Adj Close']
        data['bollinger_up'], data['bollinger_down'] = self.get_bollinger_bands(data['Ticker'])
        data['Buy_Signal_Price'], data['Sell_Signal_Price'], data['Stop_Loss']  = self.buy_sell(data)
        self.data = data
        self.analyze_performance()
        self.plot_data()
