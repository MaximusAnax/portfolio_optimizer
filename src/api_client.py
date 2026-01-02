import os
import time
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
        
        for i, ticker in enumerate(tickers):
            prices = self._fetch_single_ticker(ticker, start_date, end_date, use_cache)
            if prices is not None:
                all_prices[ticker] = prices
            
            # Small delay between fetches to avoid rate limiting (except for last ticker)
            if i < len(tickers) - 1:
                time.sleep(0.5)
                
        if not all_prices:
            return pd.DataFrame()
        
        # Normalize all indices to datetime before combining
        normalized_prices = {}
        for ticker, prices in all_prices.items():
            if prices is not None and len(prices) > 0:
                # Ensure values are numeric (convert strings to numbers)
                prices = pd.to_numeric(prices, errors='coerce')
                # Remove any invalid values
                prices = prices.dropna()
                
                # Ensure index is datetime
                if not isinstance(prices.index, pd.DatetimeIndex):
                    prices.index = pd.to_datetime(prices.index, utc=True, errors='coerce')
                # Remove any rows with invalid dates
                prices = prices.dropna()
                
                if len(prices) > 0:
                    normalized_prices[ticker] = prices
        
        if not normalized_prices:
            return pd.DataFrame()
            
        # Combine into single DataFrame and handle alignment
        df_combined = pd.DataFrame(normalized_prices)
        
        # Forward fill to handle occasional missing prices (holidays, etc.)
        return df_combined.ffill()

    def _fetch_single_ticker(self, ticker, start, end, use_cache):
        cache_path = os.path.join(self.cache_dir, f"{ticker}.csv")
        
        # 1. Try Cache First
        if use_cache and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not df.empty:
                    # Handle different column name formats
                    if 'Adj Close' in df.columns:
                        return df['Adj Close']
                    elif 'close' in df.columns:
                        return df['close']
                    elif 'Close' in df.columns:
                        return df['Close']
                    else:
                        # Try to find any price-like column
                        price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
                        if price_cols:
                            return df[price_cols[0]]
                        print(f"Warning: {ticker} cache has no recognizable price column. Available: {df.columns.tolist()}")
                        return None
            except Exception as e:
                print(f"Cache read error for {ticker}: {e}")
                return None
        
        # 2. Fetch from yfinance (requires network)
        try:
            print(f"Fetching {ticker} from yfinance...")
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                return None
            
            # Handle MultiIndex columns (when downloading multiple tickers)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            # Try to get adjusted close, fallback to close
            if 'Adj Close' in df.columns:
                price_series = df['Adj Close']
            elif 'Close' in df.columns:
                price_series = df['Close']
            elif 'close' in df.columns:
                price_series = df['close']
            else:
                print(f"Warning: {ticker} - No price column found. Available: {df.columns.tolist()}")
                return None
            
            # Save full fetch to cache
            df.to_csv(cache_path)
            return price_series
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
