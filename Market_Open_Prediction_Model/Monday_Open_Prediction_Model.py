import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
import chow_test as ct
import math

TICKER = 'SPY'

def get_data(ticker=TICKER):
    start = dt.datetime(2024, 1, 1)
    end = dt.datetime(2026, 9, 1)

    df = yf.download(tickers=ticker, start=start, end=end, interval='1d', auto_adjust=True)
    btc = yf.download(tickers='BTC-USD', start=start, end=end, interval='1d', auto_adjust=True)

    df.columns = df.columns.get_level_values(0)
    btc.columns = btc.columns.get_level_values(0)

    df.index = df.index.tz_localize(None).normalize()
    btc.index = btc.index.tz_localize(None).normalize()

    return df.dropna(), btc.dropna()

def feature_engineering(df, btc):
    df_org = df.copy()

    weekly_spy = df.groupby(pd.Grouper(freq='W')).agg(
        First_Open=('Open', 'first'),
        Last_Close=('Close', 'last'),
    ).dropna()

    target = (weekly_spy['First_Open'].shift(-1) - weekly_spy['Last_Close']) / weekly_spy['Last_Close']
    target.name = 'Target'

    btc['Feature'] = (btc['Close'].shift(-2) - btc['Close']) / btc['Close']
    btc_feature = btc[btc.index.dayofweek == 4][['Feature']].resample('W').last()

    df_model = pd.concat([target, btc_feature], axis=1).dropna()

    return df_model, df_org

def test_hypothesis(df, verbose=True):
    y = df['Target']
    X = df[['Feature']]
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
    check_for_stationarity(df['Target'], 'target')
    check_for_stationarity(df['Feature'], 'feature')

    return

def train_and_test_model(X, y, verbose=True):
    training_size = 32
    testing_size = 4
    n_splits = math.ceil((len(X) - training_size) / testing_size)

    metrics_list = []
    all_predictions_list = []
    all_y_test_list = []

    for i in range(n_splits):
        start_idx = i * testing_size
        train_end = start_idx + training_size
        test_end = train_end + testing_size

        X_train = X.iloc[start_idx:train_end]
        y_train = y.iloc[start_idx:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]
        y_train_mean = y_train.mean()
        y_guess = pd.Series(y_train_mean, index=y_test.index)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = pd.Series(model.predict(X_test), index=X_test.index)

        metrics_list.append({
            'Fold': i + 1,
            'Train_R2': model.score(X_train, y_train),
            'Test_R2': r2_score(y_test, predictions),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'Guess_RMSE': np.sqrt(mean_squared_error(y_test, y_guess)),
        })

        all_predictions_list.append(predictions)
        all_y_test_list.append(y_test)

    results_df = pd.DataFrame(metrics_list)
    all_predictions = pd.concat(all_predictions_list)
    all_y_test = pd.concat(all_y_test_list)

    global_r2 = r2_score(all_y_test, all_predictions)

    if verbose:
        print('')
        print(f'--- MODEL REPORT ---')
        print(f'Global Out-of-Sample R²: {global_r2:.4f}')
        print(f'RMSE:               {results_df['Test_RMSE'].mean():.8f} (Average error)')
        print(f'Guessing (RMSE):    {results_df['Guess_RMSE'].mean():.8f} (Average error)')
        print(f'Next Prediction:    {all_predictions.iloc[-1]:.4f}')
        print('-' * 30)
        print('*guessing is described as assuming that the gap on Monday open')
        print('would be the size of an average gap from the period')

    return all_predictions

def chow_test(X, y):
    # Perform a chow-test to see if the regimes are the same between train and test data
    print()
    ct.chow_test(
        X, y,
        int(len(X) * 0.75),
        int((len(X) * 0.75) + 1),
        0.05)

    return

def test_regimes(df_org, btc):
    # Feature engineering
    weekly_spy = df_org.groupby(pd.Grouper(freq='W')).agg(
        First_Open=('Open', 'first'),
        Last_Close=('Close', 'last'),
    )
    weekly_spy['Difference'] = (weekly_spy['Last_Close'] - weekly_spy['First_Open']) / weekly_spy['First_Open']
    weekly_spy = weekly_spy.resample('W').last()

    weekly_btc = btc.groupby(pd.Grouper(freq='W')).agg(
        First_Open=('Open', 'first'),
        Last_Close=('Close', 'last'),
    )
    weekly_btc['Difference'] = (weekly_btc['Last_Close'] - weekly_btc['First_Open']) / weekly_btc['First_Open']

    X = pd.DataFrame()
    X['Diff'] = weekly_spy['Difference'] - weekly_btc['Difference']

    X['SPY std'] = weekly_spy['Difference'].rolling(3).std()
    X['BTC std'] = weekly_btc['Difference'].rolling(3).std()
    X.dropna(inplace=True)

    # Check for optimal clusters
    n = range(2, 10, 1)
    k_inertia = []
    for i in n:
        kmsi = KMeans(n_clusters=i, random_state=42).fit(X)
        inertia = kmsi.inertia_
        k_inertia.append(inertia)

    k_sil = []
    for i in n:
        kmss = KMeans(n_clusters=i, random_state=42).fit(X)
        silhouette = silhouette_score(X, labels=kmss.labels_)
        k_sil.append(silhouette)

    sns.lineplot(x=n, y=inertia, marker='o')
    plt.title('Inertia for K-clusters')
    plt.xlabel('K-clusters')
    plt.ylabel('Inertia')
    plt.show()

    sns.lineplot(x=n, y=silhouette, marker='o')
    plt.title('Silhouette Score for K-clusters')
    plt.xlabel('K-clusters')
    plt.ylabel('Silhouette')
    plt.show()

    return


def main(verbose=True):
    df, btc = get_data()
    df, df_org = feature_engineering(df, btc)

    X, y, model = test_hypothesis(df, verbose=verbose)

    if verbose:
        check_assumptions(df, model)

    all_predictions = train_and_test_model(X, y, verbose=verbose)

    if verbose:
        chow_test(X, y)
        test_regimes(df_org, btc)

    return all_predictions, df_org

if __name__ == '__main__':
    all_predictions, df_org = main()

