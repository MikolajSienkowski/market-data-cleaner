# Financial Data Quality Engine: Automated Outlier Detection

## Overview
This project is a quantitative data cleaning pipeline designed to ingest high-frequency market data (TSLA), detect "bad ticks" (erroneous price spikes or drops), and reconstruct accurate pricing metrics. 

Financial data feeds often contain noise—glitches, flash crashes, or feed drops—that can ruin trading strategies. By mimicking the data integrity operations performed by institutional market data providers like LSEG (Refinitiv) or Bloomberg, this engine successfully acts as a robust filter to protect downstream algorithms.

## Financial Logic & Methodology
1. **The Target Benchmark:** Volume-Weighted Average Price (VWAP), a critical institutional execution metric that is highly sensitive to price and volume anomalies.
2. **The Chaos Generator:** A Monte Carlo simulation that dynamically infuses randomized extreme pricing errors (e.g., ~99% price drops or spikes mimicking "fat finger" errors and unadjusted splits) into real TSLA minute-level data.
3. **The Cleaner:** An Outlier Detection Algorithm designed to identify, isolate, and neutralize these anomalies without destroying the underlying true market variance.
4. **The Evaluation:** The system calculates the VWAP of the corrupted data and the cleaned data, comparing both against the "true" VWAP to measure the exact percentage of error reduction.

## Stress Test Results (1,000 Monte Carlo Iterations)
The algorithm was subjected to a 1,000-iteration Monte Carlo stress test, injecting randomized extreme price anomalies into historical TSLA data to quantify the algorithm's effectiveness.

| Metric | Result |
| :--- | :--- |
| **Original Error** | 2.51% (Average deviation from true VWAP) |
| **Cleaned Error** | 1.14% (Error after running algorithm) |
| **Improvement** | **54.3% reduction in pricing error** |

*Conclusion:* The outlier detection algorithm successfully eliminated over half of the synthetic pricing errors. For quantitative strategies running on high-frequency data, reducing VWAP deviation by 54% represents a massive reduction in false trading signals and execution slippage.

## Tech Stack
* **Python** (Core logic and simulation control)
* **pandas & numpy** (Vectorized time-series manipulation)
* **yfinance** (Historical market data sourcing)
* **Monte Carlo Simulation** (Statistical stress-testing architecture)

## How to Run
1. Clone the repository.
2. Install the required packages: `pip install pandas numpy yfinance`
3. Run the main pipeline: `python main.py`