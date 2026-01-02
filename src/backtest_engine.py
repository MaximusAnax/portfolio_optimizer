import pandas as pd
import numpy as np
from src.hmm_detector import RegimeDetector
from src.mpt_optimizer import PortfolioOptimizer
from src.processor import DataProcessor

class BacktestEngine:
    """
    Orchestrates the walk-forward backtest.
    
    Logic:
    1. At each rebalance date, look back `window_size` days.
    2. Fit HMM and find optimal portfolios for both regimes.
    3. Predict current regime probability.
    4. Blend weights and hold for `rebalance_freq` days.
    5. CRITICAL: Use lagged weights (w_t-1) for returns (r_t) to avoid bias.
    """
    
    def __init__(self, tickers, window_size=252, rebalance_freq=21):
        self.tickers = tickers
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        
        self.processor = DataProcessor()
        self.optimizer = PortfolioOptimizer()
        
    def run(self, df_prices):
        """Execute the walk-forward backtest."""
        # 1. Prepare features and returns
        features, feature_idx = self.processor.prepare_hmm_features(df_prices)
        prices = df_prices.loc[feature_idx]
        asset_returns = self.processor.get_returns(prices)
        
        dates = feature_idx
        n_days = len(dates)
        
        # Dataframe to store weights over time
        all_weights = pd.DataFrame(index=dates, columns=self.tickers)
        regime_probas = pd.DataFrame(index=dates, columns=['prob_low_vol'])
        
        # 2. Walk-forward loop
        # Start after one full window
        for i in range(self.window_size, n_days, self.rebalance_freq):
            rebalance_date = dates[i]
            
            # --- TRAINING ---
            # Extract window: [i - window_size : i]
            win_features = features[i - self.window_size : i]
            win_prices = prices.iloc[i - self.window_size : i]
            
            # Fit HMM
            detector = RegimeDetector(n_states=2).fit(win_features)
            
            # Identify internal state sequences for optimization data split
            internal_states = detector.model.predict(win_features)
            # Map them to our [Low Vol (0), High Vol (1)] system
            mapped_states = np.array([0 if s == detector.state_map[0] else 1 for s in internal_states])
            
            # Separate prices by detected regime
            prices_low = win_prices.iloc[mapped_states == 0]
            prices_high = win_prices.iloc[mapped_states == 1]
            
            # Optimize portfolios for each regime
            opt_weights_low = self.optimizer.optimize(prices_low, strategy='sharpe')
            opt_weights_high = self.optimizer.optimize(prices_high, strategy='min_vol')
            
            # --- PREDICTION ---
            # Current probability (using most recent features available at rebalance)
            prob_low = detector.get_latest_proba(win_features)[0]
            
            # Blend
            blended = self.optimizer.blend_weights(opt_weights_low, opt_weights_high, prob_low)
            
            # --- APPLY ---
            # Set weights for the next period [i : i + rebalance_freq]
            end_idx = min(i + self.rebalance_freq, n_days)
            for ticker, weight in blended.items():
                all_weights.loc[dates[i:end_idx], ticker] = weight
            
            regime_probas.loc[dates[i:end_idx], 'prob_low_vol'] = prob_low

        # Clean up weight dataframe
        all_weights = all_weights.ffill().dropna()
        regime_probas = regime_probas.loc[all_weights.index]
        
        # 3. Calculate Strategy Returns
        # Strategy Return at t = Weights at t-1 * Returns at t
        strategy_returns = (all_weights.shift(1) * asset_returns).sum(axis=1)
        strategy_returns = strategy_returns.loc[all_weights.index].fillna(0)
        
        return {
            'strategy_returns': strategy_returns,
            'weights': all_weights,
            'probas': regime_probas,
            'asset_returns': asset_returns.loc[all_weights.index]
        }
