# True Range Prediction Model & Dynamic Position Sizing

## Overview
This quantitative trading model forecasts the daily True Range (volatility) of major US equity indices (SPY, QQQ) to dynamically scale capital exposure. By targeting a constant daily risk threshold, the engine automatically deleverages during periods of forecasted market stress and applies 1.5x leverage during low-volatility regimes.

The pipeline is engineered to institutional standards, featuring strict walk-forward validation to eliminate look-ahead bias, log transformations to normalize fat tails, and dynamic margin/friction accounting. It successfully neutralizes extreme market events—such as the 2000 Dot-Com crash and the 2008 Financial Crisis—generating significant Alpha through volatility drag reduction.

## Financial Logic & Methodology
1. **The Target (Log Volatility):** We define risk as the Daily True Range, capturing overnight gaps. The target is log-transformed to squash kurtosis and normalize the asymmetric distribution of market shocks.
2. **The Features:** 
   * *Autoregressive Volatility:* A 5-day rolling mean of historical Log True Range (capturing volatility clustering).
   * *Exogenous Signal:* A 4-day percentage change in the `^VIX` (incorporating forward-looking options market sentiment).
   * *Validation:* Features are mathematically orthogonalized, passing Variance Inflation Factor (VIF $\approx$ 1.0) checks for multicollinearity.
3. **The Model:** An Ordinary Least Squares (OLS) Regression (`statsmodels`) executed through a strict walk-forward out-of-sample loop, ensuring the model continuously trains and tests on rolling market windows without data leakage.
4. **The Sizer:** A Target Volatility algorithm. The portfolio risks exactly a predefined percentage of total capital per day, dynamically adjusting shares held:
   $$Shares = \frac{Target\_Dollar\_Risk}{Price \times Predicted\_True\_Range}$$
5. **Real-World Microstructure:** The backtester accounts for "Cash Drag" (earning the risk-free rate on uninvested capital), Margin Interest (deducting the risk-free rate when >1.0x levered), and VIX-scaled dynamic slippage (penalizing execution during liquidity vacuums).

## Full History Out-of-Sample Results
The model's geometric compounding (CAGR) was tested across the full available history for both the S&P 500 (SPY) and the Nasdaq 100 (QQQ), actively absorbing massive volatility events. 

### QQQ vs. Dynamic Sizing
By aggressively suppressing the Nasdaq's structural volatility, the model completely sidestepped the 83% Dot-Com wipeout, allowing geometric compounding to drastically outperform the benchmark's return while carrying less than half the risk.

| Metric | Dynamic Sizing (Model) | Buy & Hold QQQ (Benchmark) |
| :--- | :--- | :--- |
| **Annualized Return (CAGR)** | **10.17%** | 8.16% |
| **Annualized Volatility** | **12.91%** | 26.41% |
| **Max Drawdown** | **-31.22%** | -82.96% |
| **Sharpe Ratio** | **0.48** | 0.16 |
| **Sortino Ratio** | **0.67** | 0.22 |
| **Beta** | 0.41 | 1.00 |
| **Alpha** | **+4.48%** | 0.00% |

**QQQ Strategy Performance**:
<img width="640" height="480" alt="performance_chart_QQQ" src="https://github.com/user-attachments/assets/performance_chart_QQQ.png" />

---

### SPY vs. Dynamic Sizing
On the S&P 500, the strategy sacrifices a small margin of absolute return to build a highly defensive, risk-adjusted portfolio, cutting maximum drawdown by 20% and generating pure, uncorrelated edge.

| Metric | Dynamic Sizing (Model) | Buy & Hold SPY (Benchmark) |
| :--- | :--- | :--- |
| **Annualized Return (CAGR)** | 10.48% | **12.04%** |
| **Annualized Volatility** | **13.81%** | 18.76% |
| **Max Drawdown** | **-35.78%** | -55.19% |
| **Sharpe Ratio** | **0.47** | 0.43 |
| **Sortino Ratio** | **0.65** | 0.61 |
| **Beta** | 0.62 | 1.00 |
| **Alpha** | **+1.49%** | 0.00% |

**SPY Strategy Performance**:
<img width="640" height="480" alt="performance_chart_SPY" src="https://github.com/user-attachments/assets/performance_chart_SPY.png" />

## Tech Stack
* **Python** (pandas, numpy)
* **statsmodels** (for linear regression, VIF, Breusch-Pagan, and Jarque-Bera testing)
* **scikit-learn** (for walk-forward model training)
* **yfinance** (for historical index and VIX market data)
* **matplotlib** (for equity curve visualizations)
* **tqdm** (for testing loop telemetry)

## How to Run
1. Clone the repository locally.
2. Install the required packages via your terminal or IDE package manager: 
   `pip install pandas numpy statsmodels yfinance matplotlib scikit-learn tqdm`
3. Run the main quantitative pipeline: 
   `python main.py`