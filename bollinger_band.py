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
    
    def get_bollinger_bands(self, prices):
        
        def get_sma(prices, rate):
            return prices.rolling(rate).mean()
        
        sma = get_sma(prices, self.sma)
        std = prices.rolling(self.sma).std()
        bollinger_up = sma + std * self.st
        bollinger_down = sma - std * self.st
        return bollinger_up, bollinger_down, sma
    
    def historical_simulation(self, data, percentile=0.99):
        latest_price = data['close'].iat[-1]
        number_of_share = 100 / latest_price
        simulation = latest_price * data['close'] / data['close'].shift(1)
        data['loss'] = 100 - simulation * number_of_share
        var_99_percent = data['loss'].quantile(percentile)
        return var_99_percent

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
        plt.title(f'{self.ticker.upper()} Close Price & Signals')
        plt.xlabel('DateTime')
        plt.ylabel('Close Price')
        plt.show()

        data['return'].plot(title=f'{self.ticker} Return Rate')
        plt.tight_layout()
        plt.show()

        # Creating dataset
        import math
        a = np.array(data['loss'])
        rounded_arr = [round(x * 2) / 2 for x in a if not math.isnan(x)]
        counts = {}
        for element in rounded_arr:
            counts[element] = counts.get(element, 0) + 1
        counts = dict(reversed(sorted(counts.items())))
        labels = list(counts.keys())
        values = list(counts.values())
        
        plt.hist(labels, weights=values)
        plt.xlim(max(labels), min(labels))
        plt.xlabel('Loss Distribution (%)')
        plt.ylabel('Frequency')
        plt.title(f'Historical Simualtion - {self.ticker} Probability Distribution of Losses')
        plt.show()

    def historical_simulation(self, data, percentile=0.99):
        latest_price = data['close'].iat[-1]
        number_of_share = 100 / latest_price
        simulation = latest_price * data['close'] / data['close'].shift(1)
        data['loss'] = 100 - simulation * number_of_share
        var_99_percent = data['loss'].quantile(percentile)
        return var_99_percent

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
        merged_df['Holding Days'] = (merged_df['Date_Sell'] - merged_df['Date_Buy']).dt.days.astype(int)
        merged_df['Profit'] = merged_df['Sell_Signal_Price'] - merged_df['Buy_Signal_Price']
        merged_df['RoR %'] = merged_df['Profit'] / merged_df['Buy_Signal_Price'] * 100
        merged_df['Win'] = (merged_df['Profit'] > 0).astype(int)
        merged_df['Lose'] = (merged_df['Profit'] <= 0).astype(int)
        merged_df['Stop Loss'] = merged_df['Date_Sell'].isin(stop_loss['Date']).astype(int)

        num_trades = len(merged_df)
        profitable_trades = merged_df['Win'].sum()
        losing_trades = merged_df['Lose'].sum()
        total_profit = merged_df['Profit'].sum()
        stop_loss = merged_df['Stop Loss'].sum()
        
        if num_trades > 0:
            win_probability = profitable_trades / num_trades * 100
            average_holding_days = merged_df['Holding Days'].mean()
            average_profit = merged_df['Profit'].mean()
            average_ror = merged_df['RoR %'].mean()
        else:
            win_probability = 0
            average_holding_days = 0
            average_ror = 0
            average_profit = 0

        var_99_percent = self.historical_simulation(data)

        print('Strategy: Bollinger Band')
        print(f"Trades: {num_trades}")
        print(f"Avg holding period: {average_holding_days}")
        print(f"Win trades: {profitable_trades}")
        print(f"Lose trades: {losing_trades}")
        print(f"Stop losses: {stop_loss}")
        print(f"Total profit: {total_profit:.2f}")
        print(f"Avg profit: {average_profit:.2f}")
        print(f"Avg RoR %%: {average_ror:.2f} %")
        print(f"Win rate %: {win_probability}")     
        print(f"1-day 99% VaR: {var_99_percent}")     

        merged_df.rename({'Date_Buy': 'Buy Date', 'Buy_Signal_Price': 'Buy Signal Price', 'Date_Sell': 'Sell Date', 'Sell_Signal_Price': 'Sell Signal Price'}, axis=1, inplace=True)
        merged_df.index += 1
        data_detail = merged_df.round(decimals=2)
        data_detail['Profit'] = data_detail['Profit'].apply(lambda x : x if x < 0 else f'+{x}')
        data_detail['RoR %'] = data_detail['RoR %'].apply(lambda x : x if x < 0 else f'+{x}')
        print(data_detail)

    def show_performance(self):
        df = self.download_data()

        data = pd.DataFrame()
        data['Ticker'] = df['Close']
        data['open'] = df['Open']
        data['high'] = df['High']
        data['low'] = df['Low']
        data['close'] = df['Close']
        data['vol'] = df['Volume']
        data['return'] = 100 * data['close'].pct_change()
        data['bollinger_up'], data['bollinger_down'], data['sma'] = self.get_bollinger_bands(data['Ticker'])
        data['Buy_Signal_Price'], data['Sell_Signal_Price'], data['Stop_Loss']  = self.buy_sell(data)

        self.analyze_performance()
        self.plot_data()

if __name__ == '__main__':
    stock_code = 'AAPL'
    start = '2022-01-01'
    end = '2023-05-01'
    stop_loss = True
    stop_loss_percent = 0.08

    sma = 20
    st = 2
    
    Bollinger_Band(ticker=stock_code, start=start, end=end, sma=sma, st=st, stop_loss=stop_loss, stop_loss_percent=stop_loss_percent).show_performance()
