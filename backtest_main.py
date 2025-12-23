import os
import pandas as pd
from src.api_client import DataClient
from src.backtest_engine import BacktestEngine
from src.metrics import MetricsEngine

def main():
    # --- CONFIGURATION ---
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    WINDOW_SIZE = 252 # 1 Year training window
    REBALANCE_FREQ = 21 # Monthly rebalancing
    
    print("Initializing Regime-Switching Backtest...")
    
    # --- DATA ACQUISITION ---
    client = DataClient()
    df_prices = client.get_data(TICKERS)
    
    if df_prices.empty:
        print("Error: No data fetched.")
        return

    # --- EXECUTION ---
    engine = BacktestEngine(TICKERS, window_size=WINDOW_SIZE, rebalance_freq=REBALANCE_FREQ)
    results = engine.run(df_prices)
    
    strategy_returns = results['strategy_returns']
    asset_returns = results['asset_returns']
    
    # --- BENCHMARK ---
    # Equal-weight benchmark for comparison
    benchmark_returns = asset_returns.mean(axis=1)
    
    # --- ANALYSIS ---
    metrics_engine = MetricsEngine()
    
    strategy_metrics = metrics_engine.calculate_metrics(strategy_returns)
    benchmark_metrics = metrics_engine.calculate_metrics(benchmark_returns)
    
    print("\n" + "="*30)
    print("BACKTEST RESULTS")
    print("="*30)
    print(f"{'Metric':<20} | {'Strategy':<10} | {'Benchmark':<10}")
    print("-" * 45)
    for m in strategy_metrics.keys():
        s_val = f"{strategy_metrics[m]:.2%}" if 'Ratio' not in m else f"{strategy_metrics[m]:.2f}"
        b_val = f"{benchmark_metrics[m]:.2%}" if 'Ratio' not in m else f"{benchmark_metrics[m]:.2f}"
        print(f"{m:<20} | {s_val:<10} | {b_val:<10}")
    print("="*45)

    # --- SAVE OUTPUTS ---
    os.makedirs('outputs', exist_ok=True)
    results['weights'].to_csv('outputs/rebalance_weights.csv')
    results['probas'].to_csv('outputs/regime_probabilities.csv')
    
    # Final CSV for comparison
    df_compare = pd.DataFrame({
        'Strategy': strategy_returns,
        'Benchmark': benchmark_returns
    })
    df_compare.to_csv('outputs/daily_returns.csv')
    
    # Visuals
    metrics_engine.plot_results(strategy_returns, benchmark_returns)
    
    print("\nBacktest Complete. Outputs saved to /outputs directory.")

if __name__ == "__main__":
    main()
