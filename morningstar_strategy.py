import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


class MorningStarStrategy:
    """
    A class for buying at morning star and stop profit / loss at a percentage.
    """

    def __init__(self, ticker, start='2015-01-01', end="2022-01-01", stop_loss_percent=0.08):
        """
        ticker: stock code
        start: start day
        end: end day
        stop_loss_percent: loss percentage compared to the buying price, or compared to the local maximum price
        """
        self.ticker = ticker
        self.start = start
        self.end = end
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
        stop_loss_percent = self.stop_loss_percent
        for i in range(len(data)):
            if data['Ticker'][i] * (1 - stop_loss_percent) > stop_loss_price:
                stop_loss_price = data['Ticker'][i] * (1 - stop_loss_percent)  
            if data['Morning_Star'][i] == 1:
                stop_loss_signal.append(np.nan)
                if flag != 1:
                    buy_signal.append(data['Ticker'][i])
                    sell_signal.append(np.nan)
                    stop_loss_price = data['Ticker'][i] * (1 - stop_loss_percent)                   
                    flag = 1
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['Ticker'][i] < stop_loss_price:
                if flag == 1:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Ticker'][i])
                    flag = -1
                    stop_loss_price = 0
                    stop_loss_signal.append(data['Ticker'][i])
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
                    stop_loss_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
                stop_loss_signal.append(np.nan)
        self.data = data
        return buy_signal, sell_signal, stop_loss_signal

    def plot_data(self):
        """
        Plots stock data with the strategy.
        """
        data = self.data
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12.5, 8))
        plt.plot(data['Ticker'], label=self.ticker.upper(), alpha=0.45)
        plt.scatter(data.index, data['Buy_Signal_Price'], label='Buy', marker='^', color='green')
        plt.scatter(data.index, data['Sell_Signal_Price'], label='Sell', marker='^', color='red')
        plt.legend(loc='upper left')
        plt.title(f'{self.ticker.upper()} Adj Close Price & Signals')
        plt.xlabel('DateTime')
        plt.ylabel('Adj Close Price')
        plt.show()

    def analyze_performance(self):
        """
        Analyze performance of the strategy.
        """
        data = self.data
        buy = data['Buy_Signal_Price'].dropna().reset_index(drop=False)
        sell = data['Sell_Signal_Price'].dropna().reset_index(drop=False)
        stop_loss = data['Stop_Loss'].dropna().reset_index(drop=False)
        merged_df = buy.join(sell, lsuffix='_Buy', rsuffix='_Sell')
        merged_df = merged_df[['Date_Buy', 'Buy_Signal_Price', 'Date_Sell', 'Sell_Signal_Price']].dropna()
        merged_df['Holding Days'] = merged_df['Date_Sell'] - merged_df['Date_Buy']
        merged_df['Profit'] = merged_df['Sell_Signal_Price'] - merged_df['Buy_Signal_Price']
        merged_df['Win'] = (merged_df['Profit'] > 0).astype(int)
        merged_df['Lose'] = (merged_df['Profit'] <= 0).astype(int)
        merged_df['Stop Loss'] = merged_df['Date_Sell'].isin(stop_loss['Date']).astype(int)

        num_trades = len(merged_df)
        profitable_trades = merged_df['Win'].sum()
        average_holding_days = merged_df['Holding Days'].mean().days
        losing_trades = merged_df['Lose'].sum()
        total_profit = merged_df['Profit'].sum()
        stop_loss = merged_df['Stop Loss'].sum()

        if num_trades > 0:
            win_probability = f'{round(profitable_trades / num_trades * 100, 2)} %'
            print(merged_df, end='\n\n')
        else:
            win_probability = '0 days'
            average_holding_days = 0

        print(f"Number of trades: {num_trades}")
        print(f"Average holding period: {average_holding_days} days")
        print(f"Number of profitable trades: {profitable_trades}")
        print(f"Number of losing trades: {losing_trades}")
        print(f"Number of stop losses: {stop_loss}")
        print(f"Total profit: {total_profit:.2f}")
        print(f"Win rate: {win_probability}")

    def show_performance(self):
        df = self.download_data()

        data = pd.DataFrame()
        data['Ticker'] = df['Adj Close']
        data['Morning_Star'] = np.where(
            (
                (df['Open'].shift(2) > df['Close'].shift(2)) & (df['Close'].shift(2) >= df['Open'].shift()) & (df['Close'].shift(2) >= df['Close'].shift()) &
                (df['Close'] > df['Open']) & ((df['Open'] >= df['Open'].shift()) & (df['Open'] >= df['Close'].shift())) &
                ((df['Close'].shift() - df['Open'].shift()).abs() < (df['Open'].shift(2) - df['Close'].shift(2))) &
                ((df['Close'].shift() - df['Open'].shift()).abs() < (df['Close'] - df['Open'])) &
                ((df['High'] - df['Low']) >= ((df['High'].shift(2) - df['Low'].shift(2)))) &
                (df['High'] >= df['High'].shift(2))
            ), 1, 0)
        print(sum(data['Morning_Star']))
        data['Buy_Signal_Price'], data['Sell_Signal_Price'], data['Stop_Loss'] = self.buy_sell(data)

        self.analyze_performance()
        self.plot_data()
