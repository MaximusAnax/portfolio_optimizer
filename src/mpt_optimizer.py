import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

class PortfolioOptimizer:
    """
    Implements regime-specific MPT optimization.
    
    Strategies:
    - State 0 (Low Vol): Maximize Sharpe Ratio (Growth)
    - State 1 (High Vol): Minimize Volatility (Defensive)
    
    Uses Ledoit-Wolf shrinkage for covariance to handle estimation error.
    """
    
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def optimize(self, df_prices, strategy='sharpe'):
        """
        Calculates optimal weights for a single regime's data.
        
        Args:
            df_prices (pd.DataFrame): Asset prices for the target regime.
            strategy (str): 'sharpe' or 'min_vol'
            
        Returns:
            dict: {ticker: weight}
        """
        if df_prices.shape[0] < 5: # Minimum observations
            return self._equal_weight(df_prices.columns)
            
        try:
            # 1. Expected Returns (Annualized)
            mu = expected_returns.mean_historical_return(df_prices)
            
            # 2. Robust Covariance (Ledoit-Wolf Shrinkage)
            S = risk_models.CovarianceShrinkage(df_prices).ledoit_wolf()
            
            # 3. Regularization for numerical stability
            S += np.eye(len(S)) * 1e-6
            
            # 4. Efficient Frontier
            ef = EfficientFrontier(mu, S)
            
            if strategy == 'sharpe':
                try:
                    ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                except ValueError:
                    # If no asset beats risk-free rate, fallback to min_vol
                    ef.min_volatility()
            else:
                ef.min_volatility()
                
            return ef.clean_weights()
            
        except Exception as e:
            print(f"Optimization error: {e}")
            return self._equal_weight(df_prices.columns)

    def blend_weights(self, weights_low, weights_high, prob_low):
        """
        Blends two sets of weights based on regime probability.
        w_blended = p * w_low + (1-p) * w_high
        """
        tickers = set(weights_low.keys()) | set(weights_high.keys())
        blended = {}
        
        prob_high = 1.0 - prob_low
        
        for t in tickers:
            w_l = weights_low.get(t, 0.0)
            w_h = weights_high.get(t, 0.0)
            blended[t] = prob_low * w_l + prob_high * w_h
            
        # Normalize
        total = sum(blended.values())
        return {t: w/total for t, w in blended.items()}

    def _equal_weight(self, tickers):
        n = len(tickers)
        return {t: 1.0/n for t in tickers}
