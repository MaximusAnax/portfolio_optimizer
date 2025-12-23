import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """
    Cleans market data and prepares features for regime detection.
    
    Key tasks:
    1. Calculate log returns for assets.
    2. Engineer 'Market' features (average return, rolling volatility) for HMM.
    3. Normalize features for stable HMM convergence.
    """
    
    def __init__(self, rolling_window=20):
        self.rolling_window = rolling_window
        self.scaler = StandardScaler()

    def get_returns(self, df_prices):
        """Calculate daily log returns."""
        return np.log(df_prices / df_prices.shift(1)).dropna()

    def prepare_hmm_features(self, df_prices):
        """
        Extracts features for HMM regime detection.
        Uses market-level (cross-sectional average) returns and volatility 
        to capture the broad market regime rather than idiosyncratic noise.
        """
        returns = self.get_returns(df_prices)
        
        # Cross-sectional average of returns (Market Return Proxy)
        market_return = returns.mean(axis=1)
        
        # Rolling volatility of the market proxy
        market_vol = market_return.rolling(window=self.rolling_window).std()
        
        # Combine into feature matrix
        features = pd.DataFrame({
            'return': market_return,
            'volatility': market_vol
        }).dropna()
        
        # Standardize features (HMM is sensitive to scale)
        scaled_features = self.scaler.fit_transform(features)
        
        return scaled_features, features.index

    def align_data(self, prices, features_index):
        """Align price data with the features index to ensure consistency."""
        return prices.loc[features_index]
