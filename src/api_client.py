import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class DataClient:
    """
    Handles fetching and caching of historical market data.
    Ensures data consistency and provides a local CSV cache to avoid redundant API calls.
    """
    
    def __init__(self, cache_dir='data/raw'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_data(self, tickers, start_date=None, end_date=None, use_cache=True):
        """
        Main entry point for data retrieval. Tries cache first, then falls back to yfinance.
        
        Args:
            tickers (list): List of ticker symbols.
            start_date (str): ISO format date (YYYY-MM-DD).
            end_date (str): ISO format date (YYYY-MM-DD).
            use_cache (bool): Whether to use local storage.
            
        Returns:
            pd.DataFrame: Adjusted Close prices for all tickers.
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        all_prices = {}
        
        for ticker in tickers:
            prices = self._fetch_single_ticker(ticker, start_date, end_date, use_cache)
            if prices is not None:
                all_prices[ticker] = prices
                
        if not all_prices:
            return pd.DataFrame()
            
        # Combine into single DataFrame and handle alignment
        df_combined = pd.DataFrame(all_prices)
        df_combined.index = pd.to_datetime(df_combined.index, utc=True)
        
        # Forward fill to handle occasional missing prices (holidays, etc.)
        return df_combined.ffill()

    def _fetch_single_ticker(self, ticker, start, end, use_cache):
        cache_path = os.path.join(self.cache_dir, f"{ticker}.csv")
        
        # 1. Try Cache First
        if use_cache and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not df.empty:
                    # If we're in a restricted environment or just want to use what we have
                    return df['Adj Close'] if 'Adj Close' in df.columns else df['close']
            except Exception as e:
                print(f"Cache read error for {ticker}: {e}")
        
        # 2. Fetch from yfinance (requires network)
        try:
            print(f"Fetching {ticker} from yfinance...")
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                return None
            
            # Save full fetch to cache
            df.to_csv(cache_path)
            return df['Adj Close']
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
