import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Monday_Open_Prediction_Model import main

LEVERAGE = 1
GAP = 0.25

def add_strategy(df_org, next_prediction, leverage=LEVERAGE, gap=GAP):
    # Assign the Prediction column for easier use and slice the DataFrame accordingly to next_prediciton variable
    df = df_org
    next_prediction.index = next_prediction.index + pd.Timedelta(days=1) # fix the dates (we move our predictions from Sunday to Monday)
    start_date = next_prediction.index[0]
    df = df.loc[start_date:].copy()
    df['Prediction'] = next_prediction * 100
    df['Prediction'] = df['Prediction'].ffill()

    # Trade with the Gap Theory logic - the gap will be closed (reversed)
    df['Strategy'] = np.where(df['Prediction'] > gap, -leverage, np.where(df['Prediction'] < -gap, leverage, 0))
    df['Strategy'] = df['Strategy'].shift(1)

    return df.dropna(), start_date

def test_strategy(df):
    # Make a Daily Return column that assumes we hold the trade from Monday Open to Friday Close
    df['Daily Return'] = df['Close'].pct_change()
    first_days = df.groupby(pd.Grouper(freq='W')).head(1).index
    df.loc[first_days, 'Daily Return'] = (df.loc[first_days, 'Close'] - df.loc[first_days, 'Open']) / df.loc[
        first_days, 'Open']

    df['Strategy Returns'] = (1 + df['Daily Return'] * df['Strategy']).cumprod()
    df['Benchmark Returns'] = (1 + df['Close'].pct_change()).cumprod()

    plt.plot(df['Strategy Returns'])
    plt.plot(df['Benchmark Returns'])
    plt.title('Gap Theory Trading vs. Benchmark (SPY)')
    plt.legend(['Strategy Returns', 'Benchmark Returns'])
    plt.show()

    return df

def evaluate_performance(df):
    # Total Returns
    s_ret = df['Strategy Returns'].iloc[-1] - 1
    b_ret = df['Benchmark Returns'].iloc[-1] - 1

    # Max Drawdown
    s_running_max = df['Strategy Returns'].cummax()
    s_max_drawdown = ((df['Strategy Returns'] / s_running_max) - 1).min()
    b_running_max = df['Benchmark Returns'].cummax()
    b_max_drawdown = ((df['Benchmark Returns'] / b_running_max) - 1).min()

    # Win Rate
    weekly_df = df.groupby(pd.Grouper(freq='W')).agg(
        Open=('Open', 'first'),
        Close=('Close', 'last'),
        Strategy=('Strategy', 'first')
    ).dropna()

    weekly_df['Weekly Returns'] = (weekly_df['Close'] - weekly_df['Open']) / weekly_df['Open']
    weekly_df['Trade Returns'] = weekly_df['Weekly Returns'] * weekly_df['Strategy']

    trades = weekly_df[weekly_df['Strategy'] != 0]
    benchmark = weekly_df

    n_trades = len(trades)
    s_w = len(trades[trades['Trade Returns'] > 0])
    s_wr = s_w / n_trades if n_trades > 0 else 0

    n_weeks = len(benchmark)
    b_w = len(benchmark[benchmark['Weekly Returns'] > 0])
    b_wr = b_w / n_weeks if n_weeks > 0 else 0

    # Expected Value
    s_wEV = trades[trades['Trade Returns'] > 0]['Trade Returns'].mean()
    s_lEV = trades[trades['Trade Returns'] < 0]['Trade Returns'].mean()
    s_wEV = 0 if pd.isna(s_wEV) else s_wEV
    s_lEV = 0 if pd.isna(s_lEV) else s_lEV
    s_EV = s_wEV * s_wr + s_lEV * (1 - s_wr)

    b_wEV = benchmark[benchmark['Weekly Returns'] > 0]['Weekly Returns'].mean()
    b_lEV = benchmark[benchmark['Weekly Returns'] < 0]['Weekly Returns'].mean()
    b_wEV = 0 if pd.isna(b_wEV) else b_wEV
    b_lEV = 0 if pd.isna(b_lEV) else b_lEV
    b_EV = b_wEV * b_wr + b_lEV * (1 - b_wr)

    metrics_data = {
        'Metric': [
            'Total Returns',
            'Max Drawdown',
            'Win Rate',
            'N Trades',
            'Weeks of backtest',
            'Expected Value'
    ],
        'Strategy': [
            f'{s_ret:.2%}',
            f'{s_max_drawdown:.2%}',
            f'{s_wr:.2%}',
            f'{n_trades}',
            f'{n_weeks}',
            f'{s_EV:.2%}',
        ],
        'Benchmark': [
            f'{b_ret:.2%}',
            f'{b_max_drawdown:.2%}',
            f'{b_wr:.2%}',
            f'{n_weeks}',
            f'{n_weeks}',
            f'{b_EV:.2%}'
        ]
    }

    comparison_df = pd.DataFrame(metrics_data)
    comparison_df.set_index('Metric', inplace=True)

    print()
    print('-' * 45)
    print('             Strategy vs. Benchmark')
    print('-' * 45)
    print(comparison_df)
    print('-' * 45)

    return df

def main_gtt():
    next_prediction, df_org = main(verbose=False)
    df, start_date = add_strategy(df_org, next_prediction)
    df = test_strategy(df)
    df = evaluate_performance(df)

    return df

if __name__ == '__main__':
    main_gtt()