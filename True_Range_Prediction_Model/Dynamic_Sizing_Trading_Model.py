import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from True_Range_Prediction_Model import main

CAPITAL = 10000
TARGET_RISK_PCT = 0.01
RISK_FREE_RATE = 0.04
LOW_VOLATILITY_LEVERAGE = 1.5

def portfolio_dynamic_sizing(df, next_prediction, capital=CAPITAL, target_risk_pct=TARGET_RISK_PCT):
    start_date = next_prediction.index[0]
    df = df.loc[start_date:].copy()
    target_dollar_risk = capital * target_risk_pct
    df['Target Capital'] = target_dollar_risk / next_prediction
    df['Target Capital'] = df['Target Capital'].clip(upper=CAPITAL)
    df['Target Shares'] = df['Target Capital'] / df['Close']

    print("--- POSITION SIZING FOR TOMORROW ---")
    print(f"Predicted True Range: {next_prediction.iloc[-1]:.4f}")
    print(f"Capital Allocated:    ${df['Target Capital'].iloc[-1]:.2f}")
    print(f"Shares to Buy:        {df['Target Shares'].iloc[-1]:.2f}")

    return df

def test_strategy(df, ticker, capital=CAPITAL, risk_free_rate=RISK_FREE_RATE, leverage=LOW_VOLATILITY_LEVERAGE):
    # Calculate the position weights
    df['Weight'] = df['Target Capital'] / capital
    df['Weight'] = df['Weight'].shift(1)

    df['Exposure'] = np.where(df['Weight'] == 1, leverage, df['Weight'])
    df['Free Capital Weight'] = 1 - df['Exposure']

    ''' We calculate our exposure if we allocate full capital we take advantage of low volatility
         using 1.5x leverage. '''

    # Calculate transaction costs
    base_commission = 0.00005
    baseline_vix = 15
    baseline_slippage = 0.00005
    dynamic_slippage = baseline_slippage * (df['VIX Close'] / baseline_vix)

    df['Friction Rate'] = base_commission + dynamic_slippage
    df['Turnover'] = df['Exposure'].diff().abs()
    df['Friction'] = df['Turnover'] * df['Friction Rate']

    # Calculate the position we take (SPY + ST-TBILLS)
    df['Close Change'] = df['Close'].pct_change()
    df['Weighted Position'] = df['Exposure'] * df['Close Change']
    df['Free Capital Position'] = df['Free Capital Weight'] * (risk_free_rate / 252)
    df['Position'] = df['Weighted Position'] + df['Free Capital Position'] - df['Friction']

    # Calculate and plot the results
    df['Strategy Returns'] = capital * (1 + df['Position']).cumprod()
    df['Benchmark Returns'] = capital * (1 + df['Close Change']).cumprod()

    plt.plot(df['Benchmark Returns'])
    plt.plot(df['Strategy Returns'])
    plt.legend(['Benchmark Returns', 'Strategy Returns'])
    plt.ylabel('Growth of $10,000')
    plt.title(f'{ticker} vs. Dynamic Sizing Portfolio')
    plt.show()

    return df.dropna()

def evaluate_strategy(df, risk_free_rate=RISK_FREE_RATE):
    # Sharpe Ratio
    years = len(df) / 252

    p_final = df['Strategy Returns'].iloc[-1]
    p_initial = df['Strategy Returns'].iloc[0]
    annualized_portfolio_returns = (p_final / p_initial) ** (1 / years) - 1

    p_std = df['Position'].std() * np.sqrt(252)
    p_sharpe_ratio = (annualized_portfolio_returns - risk_free_rate) / p_std

    b_final = df['Benchmark Returns'].iloc[-1]
    b_initial = df['Benchmark Returns'].iloc[0]
    annualized_benchmark_returns = (b_final / b_initial) ** (1 / years) - 1

    b_std = df['Close Change'].std() * np.sqrt(252)
    b_sharpe_ratio = (annualized_benchmark_returns - risk_free_rate) / b_std

    # Sortino Ratio
    p_downside_deviation = np.sqrt(np.mean(np.minimum(0, df['Position'].dropna()) ** 2)) * np.sqrt(252)
    p_sortino_ratio = (annualized_portfolio_returns - risk_free_rate) / p_downside_deviation
    b_downside_deviation = np.sqrt(np.mean(np.minimum(0, df['Close Change'].dropna()) ** 2)) * np.sqrt(252)
    b_sortino_ratio = (annualized_benchmark_returns - risk_free_rate) / b_downside_deviation

    # Max Drawdown
    p_running_max = df['Strategy Returns'].cummax()
    p_max_drawdown = ((df['Strategy Returns'] / p_running_max) - 1).min()
    b_running_max = df['Benchmark Returns'].cummax()
    b_max_drawdown = ((df['Benchmark Returns'] / b_running_max) - 1).min()

    # Calmar Ratio
    p_calmar_ratio = annualized_portfolio_returns / abs(p_max_drawdown)
    b_calmar_ratio = annualized_benchmark_returns / abs(b_max_drawdown)

    # Beta
    portfolio_returns = df['Position'].dropna()
    benchmark_returns = df['Close Change'].dropna()

    common_index = benchmark_returns.index.intersection(portfolio_returns.index)
    benchmark_returns = benchmark_returns.loc[common_index]
    portfolio_returns = portfolio_returns.loc[common_index]

    covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
    covariance = covariance_matrix[0, 1]
    market_variance = covariance_matrix[1, 1]

    beta = covariance / market_variance

    # Alpha
    alpha = annualized_portfolio_returns - (risk_free_rate + beta * (annualized_benchmark_returns - risk_free_rate))

    # Daily VaR
    p_var = abs(np.percentile(df['Position'].dropna(), 1))
    b_var = abs(np.percentile(df['Close Change'].dropna(), 1))

    # Results Table
    metrics_data = {
        'Metric': [
            'Annualized Return',
            'Annualized Volatility',
            'Value at Risk (Daily)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown',
            'Calmar Ratio',
            'Beta',
            'Alpha'
        ],
        'DPS': [
            f'{annualized_portfolio_returns:.2%}',
            f'{p_std:.2%}',
            f'{p_var:.2%}',
            f'{p_sharpe_ratio:.2f}',
            f'{p_sortino_ratio:.2f}',
            f'{p_max_drawdown:.2%}',
            f'{p_calmar_ratio:.2f}',
            f'{beta:.2f}',
            f'{alpha:.2%}'
        ],
        '   Benchmark': [
            f'{annualized_benchmark_returns:.2%}',
            f'{b_std:.2%}',
            f'{b_var:.2%}',
            f'{b_sharpe_ratio:.2f}',
            f'{b_sortino_ratio:.2f}',
            f'{b_max_drawdown:.2%}',
            f'{b_calmar_ratio:.2f}',
            '1.00',  # Benchmark Beta is always 1
            '0.00%'  # Benchmark Alpha is always 0
        ]
    }

    comparison_df = pd.DataFrame(metrics_data)
    comparison_df.set_index('Metric', inplace=True)

    print('-' * 45)
    print('             DPS vs. Benchmark')
    print('-' * 45)
    print(comparison_df)
    print('-' * 45)

    return

def main_dps():
    df, next_prediction, ticker = main(verbose=False)
    df = portfolio_dynamic_sizing(df, next_prediction)
    df = test_strategy(df, ticker)
    evaluate_strategy(df)

    return df

if __name__ == '__main__':
    main_dps()