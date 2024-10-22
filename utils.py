import yfinance as yf

def fetch_data(symbol, period, interval, auto_adjust, back_adjust):
    """Fetch OHLCV data from Yahoo! Finance."""
    return yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        actions=False,
        auto_adjust=auto_adjust,
        back_adjust=back_adjust
    )
