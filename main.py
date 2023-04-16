from sma import SMA_Strategy
from bollinger_band import Bollinger_Band

if __name__ == '__main__':
    stock_code = 'AAPL'
    start = '2020-01-01'
    end = '2023-01-01'
    stop_loss = True
    stop_loss_percent = 0.08

    # For SMA
    sma_short = 10
    sma_long = 50

    # For Bollinger Band
    sma = 20
    st = 2
    
    SMA_Strategy(ticker=stock_code, start=start, end=end, sma_short=sma_short, sma_long=sma_long, stop_loss=stop_loss, stop_loss_percent=stop_loss_percent).show_performance()
    Bollinger_Band(ticker=stock_code, start=start, end=end, sma=sma, st=st, stop_loss=stop_loss, stop_loss_percent=stop_loss_percent).show_performance()