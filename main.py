from sma_strategy import SMA_Strategy
from morningstar_strategy import MorningStarStrategy

if __name__ == '__main__':
    stock_code = 'AAPL'
    start = '2020-01-01'
    end = '2023-01-01'
    stop_loss = True
    stop_loss_percent = 0.08
    sma_short = 10
    sma_long = 50
    
    SMA_Strategy(ticker=stock_code, start=start, end=end, sma_short=sma_short, sma_long=sma_long, stop_loss=stop_loss, stop_loss_percent=stop_loss_percent).show_performance()
    MorningStarStrategy(ticker=stock_code, start=start, end=end, stop_loss_percent=stop_loss_percent).show_performance()