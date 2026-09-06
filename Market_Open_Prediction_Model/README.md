# Monday Open Prediction Model: Cross-Asset Momentum

## Overview
This quantitative research project explores cross-asset momentum by predicting the S&P 500 (SPY) Monday morning opening gap using Bitcoin (BTC) weekend price action. 

Because traditional equity markets are closed over the weekend, macroeconomic news and shifts in global sentiment often have no outlet for price discovery until Monday morning. By using a 24/7 crypto asset (BTC) as a proxy for weekend sentiment, this model successfully identifies a statistically significant leading indicator for traditional equity market gaps.

## Financial Logic & Methodology
1. **The Target:** The percentage gap between Friday's SPY Close and Monday's SPY Open.
2. **The Feature:** The percentage change in BTC over the weekend (aligned to traditional market hours).
3. **Data Alignment (Avoiding Look-Ahead Bias):** Vectorized index shifting ensures strict separation of timeframes. SPY data is shifted forward by 1 day, and BTC data is shifted forward by 2 days before filtering for Fridays (`dayofweek == 4`), guaranteeing that the model only uses data mathematically available *before* the Monday open.
4. **The Model:** An Ordinary Least Squares (OLS) Regression (`statsmodels`) and a standard Linear Regression (`scikit-learn`).
5. **Walk-Forward Validation:** To account for market regime shifts and alpha decay, the model utilizes a rolling-window training and testing framework rather than a static data split, ensuring the signal adapts to shifting market beta dynamically.

## Market Regime & Structural Break Diagnostics
To thoroughly investigate out-of-sample alpha decay and ensure model robustness, the pipeline includes statistical tests for market regime changes:
* **Chow Test:** Deployed to evaluate the equality of regression coefficients across different time periods. The test mathematically confirmed coefficient stability (failing to reject the null hypothesis), indicating that out-of-sample degradation was caused by natural edge compression and noise, rather than a sudden structural break in the asset correlation.
* **K-Means Clustering:** Unsupervised learning was applied to search for latent market regimes by evaluating the spread between SPY and BTC returns. Inertia and Silhouette score mapping confirmed a continuous distribution without distinct, isolated data clusters, corroborating the stability findings of the Chow test.

## Statistical Results & Edge
The model identified a highly significant linear relationship in the training data, proving that weekend crypto flows contain forward-looking information about traditional equities.

| Metric | Result  | Interpretation                                                                                                                  |
| :--- |:--------|:--------------------------------------------------------------------------------------------------------------------------------|
| **R-Squared** | 0.174   | The model explains ~17% of the variance in SPY Monday gaps in-sample (exceptionally high for a single-feature financial model). |
| **P-Value** | < 0.001 | The relationship is statistically significant and not the result of random chance.                                              |
| **Coefficient (Beta)** | 0.1347  | For every **1.00%** move in BTC over the weekend, SPY is expected to gap by **0.1347%** in the same direction on Monday.        |
| **Out-Of-Sample R²**| 0.0848  | The rolling model successfully explains ~8.5% of variance on completely unseen future data.                                     |
| **RMSE (Model)** | 0.0064  | The rolling model's average out-of-sample prediction error.                                                                     |
| **RMSE (Baseline)**| 0.0067  | The error rate if simply guessing the historical average gap. The model successfully beats this baseline.                       |

*Conclusion:* In the modern post-ETF market environment where institutional capital bridges crypto and traditional equities, weekend BTC momentum is a viable predictive signal for SPY Monday open trajectories.

# Implementing the Model into a Trading Strategy

## The Logic
* **Mean Reversion (Gap Theory):** Assume that extreme Monday Open Gaps tend to revert or close over the course of the week.
* Use the model's predictions to act as a noise filter for execution. 
* A predicted gap larger than **0.25%** triggers a **short** position, a gap smaller than **-0.25%** triggers a **long** position. Every prediction within that threshold is considered noise and results in no market exposure (staying in cash).

## The Out-Of-Sample Results (104 weeks)
| Metric               | Gap Trading (Model) | Buy & Hold SPY (Benchmark) |
|:---------------------|:--------------------|:---------------------------|
| **Total Returns**    | **10.05%**          | 40.54%                     |
| **Max Drawdown**     | **-11.24%**         | -18.76%                    |
| **Win Rate**         | **61.90%**          | 61.68%                     |
| **Number of Trades** | **42**              | 104                        |
| **Expected Value**   | **0.39%**           | 0.28%                      |

*Conclusion:* The strategy demonstrates a highly controlled, risk-adjusted edge. By remaining entirely out of the market ~60% of the time, it naturally underperforms the absolute return of a historic multi-year bull market. However, acting as a highly selective gap-closure engine, it generates a superior Expected Value per trade (0.39%) while cutting the benchmark's Max Drawdown nearly in half. 

## Tech Stack
* **Python** (pandas, numpy)
* **statsmodels** (for linear regression, statistical inference, and regime testing)
* **scikit-learn** (for model training, clustering, and evaluation metrics)
* **yfinance** (for historical SPY and BTC market data)

## How to Run
1. Clone the repository.
2. Install the required packages: `pip install numpy pandas yfinance matplotlib statsmodels seaborn scikit-learn chow-test`
3. Run the model diagnostics: `python Monday_Open_Prediction_Model.py`
4. Run the backtester: `python Gap_Theory_Trading.py`