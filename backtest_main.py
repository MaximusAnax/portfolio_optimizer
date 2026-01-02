import os
import pandas as pd
from src.api_client import DataClient
from src.backtest_engine import BacktestEngine
from src.metrics import MetricsEngine

def main():
    # --- CONFIGURATION ---
    # Diversified portfolio for regime-switching optimization:
    # - Broad Market: SPY (S&P 500 ETF) for market exposure
    # - Growth Stocks: AAPL, MSFT (tech growth, higher risk/reward)
    # - Defensive Assets: GLD (Gold ETF), TLT (Long-term Treasury Bonds)
    # - Utilities: XLU (Utilities sector, defensive equity)
    TICKERS = ['SPY', 'AAPL', 'MSFT', 'GLD', 'TLT', 'XLU']
    WINDOW_SIZE = 252 # 1 Year training window
    REBALANCE_FREQ = 21 # Monthly rebalancing
    
    print("Initializing Regime-Switching Backtest...")
    print(f"Portfolio: {', '.join(TICKERS)}")
    print("  - Broad Market: SPY")
    print("  - Growth: AAPL, MSFT")
    print("  - Defensive: GLD (Gold), TLT (Bonds), XLU (Utilities)")
    
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
    # Use SPY (S&P 500) as benchmark if available, otherwise equal-weight
    if 'SPY' in asset_returns.columns:
        benchmark_returns = asset_returns['SPY']
        benchmark_name = "S&P 500 (SPY)"
    else:
        benchmark_returns = asset_returns.mean(axis=1)
        benchmark_name = "Equal-Weight Portfolio"
    
    # --- ANALYSIS ---
    metrics_engine = MetricsEngine()
    
    strategy_metrics = metrics_engine.calculate_metrics(strategy_returns)
    benchmark_metrics = metrics_engine.calculate_metrics(benchmark_returns)
    
    print("\n" + "="*30)
    print("BACKTEST RESULTS")
    print("="*30)
    print(f"Benchmark: {benchmark_name}")
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
    metrics_engine.plot_results(strategy_returns, benchmark_returns, benchmark_name=benchmark_name)
    metrics_engine.plot_regime_probabilities(results['probas'])
    
    print("\nBacktest Complete. Outputs saved to /outputs directory.")

if __name__ == "__main__":
    main()
