Financial Data Quality Engine: Automated Outlier Detection

Project Overview

A quantitative data cleaning pipeline designed to ingest high-frequency market data (TSLA), detect "bad ticks" (erroneous price spikes), and reconstruct accurate pricing metrics. This project mimics the data integrity operations performed by market data providers like **LSEG (Refinitiv)** or **Bloomberg**.

Objective
Financial data feeds often contain noise—glitches, flash crashes, or feed drops—that can ruin trading strategies. The goal was to build a robust **Outlier Detection Algorithm** and quantify its impact on **Volume-Weighted Average Price (VWAP)** calculations using Monte Carlo methods.

Tech Stack
* **Python:** Core logic and simulation control.
* **pandas & numpy:** Vectorized time-series manipulation.
* **yfinance:** Historical market data sourcing.
* **Monte Carlo Simulation:** Statistical stress-testing.

Key Results (1,000 Iteration Stress Test)
I ran a Monte Carlo simulation infusing random 99% price drops ("fat finger" errors) into real TSLA minute-level data.

| Metric | Result |
| :--- | :--- |
| **Original Error** | **2.51%** (Average deviation from true VWAP) |
| **Cleaned Error** | **1.14%** (Error after running algorithm) |
| **Improvement** | **54.3%** reduction in pricing error |

> **Risk Note:** In rare edge cases (~0.6% of runs), the algorithm produced a negative improvement. This occurred during periods of extreme real volatility, where the model flagged valid price moves as "errors." This highlights the trade-off between **Data Quality** and **Signal Preservation**.

Methodology:
1.  **Data Ingestion:** Fetched 1-minute interval data for TSLA.
2.  **Noise Injection:** Deliberately corrupted 50 random data points per iteration to simulate feed instability.
3.  **Detection Logic:** Implemented a **Rolling Z-Score (Window=20)**. unlike a static threshold, this adapts to changing market volatility to minimize false positives.
4.  **Reconstruction:** Used Forward-Filling and VWAP re-calculation to establish "Fair Value."
