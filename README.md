# Regime-Switching Portfolio Optimizer

A professional quantitative finance pipeline for regime-aware portfolio optimization. This project uses Hidden Markov Models (HMM) to detect market regimes (e.g., Calm vs. Crisis) and dynamically adjusts portfolio allocations using Modern Portfolio Theory (MPT).

## 🚀 Key Features

- **HMM Regime Detection**: Automatically identifies market states using cross-sectional average returns and rolling volatility.
- **Dynamic Optimization**:
    - **State 0 (Low Vol)**: Growth-oriented strategy maximizing the Sharpe Ratio.
    - **State 1 (High Vol)**: Defensive strategy minimizing portfolio volatility.
- **Robust Estimation**: Uses Ledoit-Wolf shrinkage for covariance matrices to reduce estimation error.
- **Probabilistic Blending**: Smoothly transitions between portfolios based on the probability of each regime, reducing turnover and transaction costs.
- **Walk-Forward Backtesting**: Implements a rigorous walk-forward methodology with zero look-ahead bias (using lagged weights for returns calculation).

## 📁 Project Structure

```text
reg_switch_portfolio_optimizer/
├── backtest_main.py       # Main entry point and orchestrator
├── environment.yml        # Conda environment definition
├── src/                   # Core source code
│   ├── api_client.py      # Data fetching and CSV caching
│   ├── processor.py       # Feature engineering and data cleaning
│   ├── hmm_detector.py    # HMM regime detection logic
│   ├── mpt_optimizer.py   # MPT optimization (Sharpe/Min Vol)
│   ├── backtest_engine.py # Walk-forward backtest orchestration
│   └── metrics.py         # Performance analytics and visualization
├── data/raw/              # Local cache for market data (CSV)
└── outputs/               # Generated charts and results (CSV/PNG)
```

## 🛠 Setup & Usage

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate fin_pipeline
```

### 2. Run Backtest
```bash
python backtest_main.py
```

## 📊 Outputs
- `outputs/performance.png`: Cumulative returns comparison vs. equal-weight benchmark.
- `outputs/daily_returns.csv`: Time series of daily strategy and benchmark returns.
- `outputs/rebalance_weights.csv`: Historical portfolio weights at each rebalance date.
- `outputs/regime_probabilities.csv`: Predicted probability of the "Low Vol" regime over time.

## 📝 Design Principles
1. **Modularity**: Each component (Data, HMM, MPT, Backtest) is isolated and independently testable.
2. **Temporal Integrity**: Strict adherence to "no look-ahead" rules using $w_{t-1} \cdot r_t$ logic.
3. **Robustness**: Multi-layer fallbacks in the optimization process to handle non-convexity or lack of data.

