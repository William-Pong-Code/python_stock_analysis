from sma import SMAAnalyzer

if __name__ == '__main__':
    analyzer = SMAAnalyzer(ticker='AAPL', start='2020-01-01', end='2023-01-01', sma_short=10, sma_long=50, stop_loss=True, stop_loss_percent=0.08)
    analyzer.show_performance()