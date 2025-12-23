import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MetricsEngine:
    """
    Calculates financial performance metrics and handles visualization.
    """
    
    @staticmethod
    def calculate_metrics(returns, risk_free_rate=0.02):
        """
        Calculates key performance indicators.
        """
        if returns.empty:
            return {}
            
        # Annualized Return
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return)**(1/n_years) - 1
        
        # Annualized Volatility
        ann_vol = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0
        
        # Max Drawdown
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'Annual Return': ann_return,
            'Annual Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }

    @staticmethod
    def plot_results(strategy_returns, benchmark_returns, save_path='outputs/performance.png'):
        """Plots cumulative returns comparison."""
        plt.figure(figsize=(12, 6))
        
        cum_strategy = (1 + strategy_returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()
        
        plt.plot(cum_strategy, label='Regime-Switching Strategy', color='blue')
        plt.plot(cum_benchmark, label='Equal-Weight Benchmark', color='gray', linestyle='--')
        
        plt.title('Cumulative Returns: Strategy vs Benchmark')
        plt.ylabel('Growth of $1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
