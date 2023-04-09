from sma import SMAAnalyzer

if __name__ == '__main__':
    analyzer = SMAAnalyzer(ticker='AAPL', start='2015-01-01', end='2020-01-01', sma_short=10, sma_long=50)
    analyzer.show_performance()