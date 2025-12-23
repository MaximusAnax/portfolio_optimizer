# Project: Quant-Regime Portfolio Optimizer
**Role:** AI Engineering & Quant Mentor  
**Goal:** Build a regime-switching portfolio strategy using HMM and MPT.

## 1. Architecture & Data Flow

**Complete Pipeline:**
```
yfinance → api_client.py → processor.py → hmm_detector.py → mpt_optimizer.py
   ↓                                            ↓
 Cache                                   State Probabilities
(data/raw)                               → Blended Weights
```

- **Modular Pipeline:** `api_client.py` (fetch) → `processor.py` (transform) → `hmm_detector.py` (detect regimes) → `mpt_optimizer.py` (allocate) → `main.py` (orchestrate)
- **Data Source:** `yfinance` with `auto_adjust=True` (strips stock splits/dividends automatically)
- **Caching Strategy:** `data/raw/{SYMBOL}.csv` — always check cache before network requests
- **DataFrame Contract:** DatetimeIndex named 'date', columns are lowercase (`[open, high, low, close, volume]`)
  - Example: `df.index.name = 'date'` and `df.columns = [col.lower() for col in df.columns]`

## 2. Tech Stack & Setup
- **Python 3.10** in Conda environment `fin_pipeline` (`environment.yml` defines dependencies)
- **Core Libs:** `pandas`, `numpy`, `scikit-learn`, `yfinance`
- **ML/Finance:** `hmmlearn` (regime detection), `PyPortfolioOpt` (MPT optimization)
- **Utilities:** `python-dotenv`, `scipy`
- **Setup:** `conda env create -f environment.yml && conda activate fin_pipeline`

## 3. Completed Phases & Implementation

### Phase 1: Data Acquisition & Processing ✅
**File:** `src/api_client.py`, `src/processor.py`
- Fetch daily OHLCV data from yfinance with rate limiting (2-sec decorator)
- Standardize column names to lowercase, set DatetimeIndex named 'date'
- Cache to CSV for repeated runs without network calls
- Current assets: AAPL, MSFT, GOOGL, TSLA

### Phase 2: Regime Detection (HMM) ✅
**File:** `src/hmm_detector.py`
**Class:** `RegimeDetector`

**Input:** Feature matrix (N×2) from `prepare_hmm_features()`
- Log returns: $\ln(P_t / P_{t-1})$ (stationary, additive)
- Rolling volatility: 20-day rolling std (regime capture)
- Normalized via StandardScaler (equal weighting)

**HMM Model:**
- Gaussian emissions: $P(\text{features} | \text{state}) \sim \mathcal{N}(\mu, \Sigma)$
- Baum-Welch (EM) learns state parameters
- Viterbi decoding returns most likely state sequence

**Output:** `hidden_states` array (integer labels 0, 1)
**Characterization:** Mean/std of returns and volatility per state

**Key Methods:**
```python
detector = RegimeDetector(n_states=2, random_state=42)
hidden_states = detector.fit_predict(features)  # Viterbi
state_summary = detector.characterize_states(features, hidden_states)
detector.print_regime_summary(state_summary)
transmat = detector.get_transition_matrix()  # Regime stickiness
startprob = detector.get_start_probability()
```

### Phase 3: Regime-Aware Portfolio Optimization (MPT) ✅
**File:** `src/mpt_optimizer.py`
**Class:** `RegimeOptimizer`

**Input:** Historical prices + `hidden_states` from HMM

**Regime-Specific Optimization:**
- **State 0 (Low Vol):** Maximize Sharpe ratio (growth-oriented allocation)
- **State 1 (High Vol):** Minimize volatility (defensive posture)
- Uses `PyPortfolioOpt.EfficientFrontier` with long-only constraints

**Ledoit-Wolf Advantage:**
Shrinkage covariance $\Sigma_{LW} = (1-\alpha)\hat{\Sigma} + \alpha F$ reduces:
- Estimation error (low variance)
- Regime-specificity (robust across conditions)
- Weight extremes (stable allocations)

**Key Methods:**
```python
optimizer = RegimeOptimizer(prices, hidden_states=hidden_states)
optimizer.optimize_all_regimes(verbose=True)

# Probability-blended weights
blended = optimizer.get_blended_weights(
    current_state_probs=np.array([0.7, 0.3])  # 70% Low Vol
)
# Output: {0: 0.512, 1: 0.188, 2: 0.153, 3: 0.147}
```

**Weight Blending Formula:**
$$w_{blended} = P(\text{State 0}) \cdot w_{State 0} + P(\text{State 1}) \cdot w_{State 1}$$

Benefits:
- Smooth transitions (reduces transaction costs)
- Probabilistic hedging (no abrupt flips)
- Theoretically optimal under uncertainty

## 4. Critical Patterns

### Rate Limiting (Decorator Pattern)
```python
@rate_limit(2)  # 2-second wait between API calls
def fetch_daily_data(self, symbol):
    ticker = yf.Ticker(symbol)
    return ticker.history(period="max", interval="1d", auto_adjust=True)
```
- Applied in `api_client.py` to avoid throttling
- Decorator in `utils.py` uses closure to track last call time

### Feature Engineering for HMM
`prepare_hmm_features(df)` in `processor.py` transforms data into stationary features:
- **Log returns:** $\ln(P_t / P_{t-1})$ — mathematically additive, stationary
- **Rolling volatility:** 20-day rolling std of log returns—captures variance regimes
- **Normalization:** StandardScaler ensures both features are equally weighted in HMM
- **Returns:** Feature matrix (N×2) and cleaned DataFrame with preserved dates
```python
features, df_clean = prepare_hmm_features(df)  # features shape: (n_samples, 2)
# features[:, 0] = normalized log returns
# features[:, 1] = normalized volatility
```

### HMM Convergence (Why Long Time Series Matter)
- **Data Requirement:** ~250 observations/state/year (e.g., 5 years = 1250 obs for 2-state model)
- **EM Convergence:** Iterations 2-50 show rapid improvement; 50-500 plateaus
- **One Long Chain vs. Multiple:** Finance typically uses one continuous market history
- **Each Observation Refines:** More data = more stable parameter estimates

### MPT Weight Blending Workflow
1. Detect HMM regime probabilities: $P(\text{State 0}), P(\text{State 1})$
2. Get optimal weights for each regime separately
3. Blend: $w_{final} = P_0 \cdot w_0 + P_1 \cdot w_1$
4. Normalize to sum to 1.0

## 5. Implementation Rules
- **No Monoliths:** API logic (yfinance calls) ≠ Data processing ≠ HMM detection ≠ Portfolio optimization
- **Lazy Loading:** Check `os.path.exists(file_path)` before `client.fetch_daily_data()`
- **Error Handling:** Return `None` from processors on empty data; let `main.py` decide next step
- **Pedagogy:** Explain *why* (e.g., adjusted close corrects for dividends, log returns are stationary)
- **Known Limitation:** Symbol list is hardcoded in `main.py`—future: move to config file or CLI args

## 6. Logging & Testing
- **Directory:** `logs/` stores API usage metrics and rate-limit hit tracking
- **Tests:** 
  - `test_hmm_features.py` — Feature engineering validation
  - `test_hmm_detector.py` — HMM fitting, regime characterization, real data
  - `test_mpt_optimizer.py` — Regime optimization, weight blending, validation
  - `test_integration.py` — End-to-end pipeline (optional)

## 7. File Structure
```
src/
  __init__.py
  api_client.py       # StockDataClient (yfinance wrapper)
  processor.py        # process_stock_data(), prepare_hmm_features()
  hmm_detector.py     # RegimeDetector (Gaussian HMM, Viterbi, characterization)
  mpt_optimizer.py    # RegimeOptimizer (regime-specific portfolio optimization, blending)
  utils.py            # rate_limit() decorator

data/
  raw/                # {SYMBOL}.csv cached prices
  processed/          # (reserved for future use)

tests/
  test_hmm_features.py
  test_hmm_detector.py
  test_mpt_optimizer.py

main.py              # Orchestrator: fetch → process → detect → optimize
environment.yml      # Conda environment with all dependencies
```

## 8. Future Phases
- **Backtesting Engine:** Calculate Sharpe ratio, max drawdown, cumulative returns on blended portfolios
- **Risk Management:** Position limits, sector constraints, turnover penalties
- **Real-time Deployment:** Market data feed → regime detection → weight recommendation → execution
- **Multi-asset:** Extend to bonds, commodities, forex, multi-currency hedging