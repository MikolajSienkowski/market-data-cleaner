import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

TICKER = 'SPY'

def get_data(ticker=TICKER):
    df = yf.download(tickers=ticker, period='max', interval='1d', auto_adjust=True)
    vix = yf.download(tickers='^VIX', period='max', interval='1d', auto_adjust=True)

    df.columns = df.columns.get_level_values(0)
    vix.columns = vix.columns.get_level_values(0)

    for c in vix.columns:
        vix[f'VIX {c}'] = vix[f'{c}']

    vix = vix.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    df = df.join(vix)

    return df.dropna(), ticker

def feature_engineering(df):
    df['H-L'] = (df['High'] - df['Low'])
    df['H-C'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-C'] = (df['Low'] - df['Close'].shift(1)).abs()

    df['True Range'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['True Range Pct'] = df['True Range'] / df['Close']
    df['5-day True Range Mean'] = df['True Range Pct'].rolling(5).mean()

    df['Log True Range Pct'] = np.log(df['True Range Pct'])
    df['5-day Log True Range Mean'] = df['Log True Range Pct'].rolling(5).mean()

    df['VIX Close Change'] = df['VIX Close'].pct_change(4)

    df['Target'] = df['Log True Range Pct'].shift(-1)

    return df.replace([np.inf, -np.inf], np.nan).dropna()

def test_hypothesis(df, verbose=True):
    X = df[['5-day Log True Range Mean', 'VIX Close Change']]
    y = df['Target']
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    if verbose:
        print(model.summary())

    return X, y, model

def check_for_stationarity(x, variable:str):
    _, adf1, _, _, _, _ = adfuller(x)
    if adf1 < 0.05:
        print(f'We reject the null hypothesis - {variable} is stationary.')
    else:
        print(f'We do not reject the null hypothesis - {variable} is non-stationary.')

    return

def check_for_multicollinearity(X):
    vif_data = pd.DataFrame({
        'Feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    return vif_data

def check_assumptions(df, model):
    # Test for normality of residuals
    residuals = model.resid
    _, jb, skew, kurt = jarque_bera(residuals)

    sm.qqplot(residuals, line='s', label=f'P-value: {jb:.4f}, Skew: {skew:.2f}, Kurtosis: {kurt:.2f}')
    plt.title('Check for Normality of Residuals')
    plt.legend(loc='best')
    plt.show()

    # Check for homoscedasticity
    _, _, _, bp = het_breuschpagan(residuals, model.model.exog)

    fitted_values = model.predict()
    plt.scatter(x=fitted_values, y=residuals, label=f'P-value: {bp:.4f}')
    plt.title('Check for Homoscedasticity')
    plt.legend(loc='best')
    plt.show()

    # Check for stationarity
    check_for_stationarity(df['Target'], 'Target')
    for n in ['5-day Log True Range Mean', 'VIX Close Change']:
        check_for_stationarity(df[f'{n}'], f'{n}')

    # Check for multicollinearity
    X = df[['5-day Log True Range Mean', 'VIX Close Change']]
    vif = check_for_multicollinearity(X)
    print(vif)

    return


def train_and_test_model(X, y, verbose=True):
    training_size = 250
    testing_size = 5
    n_splits = math.ceil((len(X) - training_size) / testing_size)

    metrics_list = []
    all_predictions_list = []
    all_y_test_list = []

    for i in tqdm(range(n_splits), desc='Progress', unit='folds'):
        start_idx = i * testing_size
        train_end = start_idx + training_size
        test_end = train_end + testing_size

        if test_end > len(X):
            test_end = len(X)
            if train_end >= test_end:
                break

        X_train = X.iloc[start_idx:train_end]
        y_train = y.iloc[start_idx:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        y_guess = pd.Series(y_train.iloc[-1], index=y_test.index).dropna()

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = pd.Series(model.predict(X_test), index=X_test.index)

        predictions_raw = np.exp(predictions)
        y_test_raw = np.exp(y_test)
        y_guess_raw = np.exp(y_guess)

        metrics_list.append({
            'Fold': i + 1,
            'Test_RMSE': np.sqrt(mean_squared_error(y_test_raw, predictions_raw)),
            'Guess_RMSE': np.sqrt(mean_squared_error(y_test_raw, y_guess_raw)),
        })

        all_predictions_list.append(predictions_raw)
        all_y_test_list.append(y_test_raw)

    results_df = pd.DataFrame(metrics_list)
    all_predictions = pd.concat(all_predictions_list)
    all_y_test = pd.concat(all_y_test_list)

    global_r2 = r2_score(all_y_test, all_predictions)

    if verbose:
        print('')
        print(f'--- OLS WALK-FORWARD REPORT ---')
        print(f'Global Out-of-Sample R²: {global_r2:.4f}')
        print(f'RMSE:               {results_df["Test_RMSE"].mean():.8f} (Average error)')
        print(f'Guessing (RMSE):    {results_df["Guess_RMSE"].mean():.8f} (Average error)')
        print(f'Next Prediction:    {all_predictions.iloc[-1]:.4f}')
        print('-' * 30)

        plt.figure(figsize=(12, 6))
        plt.plot(all_y_test.index, all_y_test, label='Actual Volatility', alpha=0.5)
        plt.plot(all_predictions.index, all_predictions, label='Predicted Volatility', color='red', alpha=0.7)
        plt.title(f'{TICKER} True Range - Walk-Forward OLS Predictions')
        plt.legend()
        plt.show()

    return all_predictions

def main(verbose=True):
    df, ticker = get_data()
    df = feature_engineering(df)
    X, y, model  = test_hypothesis(df, verbose=verbose)
    if verbose:
        check_assumptions(df, model)
    next_prediction= train_and_test_model(X, y, verbose=verbose)

    return df, next_prediction, ticker

if __name__ == '__main__':
    main()