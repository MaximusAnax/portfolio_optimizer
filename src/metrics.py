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
    def plot_results(strategy_returns, benchmark_returns, save_path='outputs/performance.png', 
                     benchmark_name='Benchmark'):
        """Plots cumulative returns comparison."""
        plt.figure(figsize=(12, 6))
        
        cum_strategy = (1 + strategy_returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()
        
        plt.plot(cum_strategy, label='Regime-Switching Strategy', color='blue')
        plt.plot(cum_benchmark, label=benchmark_name, color='gray', linestyle='--')
        
        plt.title('Cumulative Returns: Strategy vs Benchmark')
        plt.ylabel('Growth of $1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

    @staticmethod
    def plot_regime_probabilities(regime_probas, save_path='outputs/regime_probabilities.png'):
        """
        Plots regime probabilities over time.
        Shows the probability of being in the 'Low Volatility' regime.
        """
        if regime_probas.empty:
            print("No regime probabilities to plot.")
            return
        
        # Ensure index is datetime if it's not already
        if not isinstance(regime_probas.index, pd.DatetimeIndex):
            regime_probas.index = pd.to_datetime(regime_probas.index)
            
        plt.figure(figsize=(14, 6))
        
        # Extract the probability column
        prob_low_vol = regime_probas['prob_low_vol']
        
        # Plot the probability
        plt.plot(prob_low_vol.index, prob_low_vol.values, 
                color='green', linewidth=1.5, alpha=0.7, label='P(Low Vol Regime)')
        
        # Add a horizontal line at 0.5 (50% threshold)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
        
        # Fill area above/below threshold for visual clarity
        plt.fill_between(prob_low_vol.index, 0, 0.5, 
                        where=(prob_low_vol <= 0.5), 
                        alpha=0.2, color='red', label='High Vol Regime (P < 50%)')
        plt.fill_between(prob_low_vol.index, 0.5, 1.0, 
                        where=(prob_low_vol > 0.5), 
                        alpha=0.2, color='green', label='Low Vol Regime (P > 50%)')
        
        plt.title('Regime Probabilities Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Probability of Low Volatility Regime', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Regime probabilities plot saved to {save_path}")
