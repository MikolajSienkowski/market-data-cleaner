import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

    return df.dropna()

def add_feature(df):
    df['H-L'] = (df['High'] - df['Low'])
    df['H-C'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-C'] = (df['Low'] - df['Close'].shift(1)).abs()

    df['True Range'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['True Range Pct'] = df['True Range'] / df['Close']
    df['5-day True Range Mean'] = df['True Range Pct'].rolling(5).mean()

    df['VIX Close Change'] = df['VIX Close'].pct_change(4)

    return df.dropna()

def add_target(df):
    df['Target'] = df['True Range Pct'].shift(-1)

    return df.dropna()

def test_hypothesis(df):
    X = df[['True Range Pct', '5-day True Range Mean', 'VIX Close Change']]
    y = df['Target']
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    print(model.summary())

    return X, y

def train_and_test_model(X, y):
    split = int(len(X) * 0.75)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    avg_true_range = predictions.mean()
    next_prediction = pd.Series(predictions, index=X_test.index)

    print(f'--- MODEL REPORT ---')
    print(f'Training R²:        {model.score(X_train, y_train):.4f}')
    print(f'Testing R²:         {r2:.4f}')
    print(f'Avg True Range:     {avg_true_range:.4f}')
    print(f'RMSE:               {np.sqrt(mse):.8f} (Average error)')
    print(f'Guessing (RMSE):    {np.sqrt(mean_squared_error(y_test, X_test['True Range Pct'])):.8f}')
    print(f'Next Prediction:    {next_prediction.iloc[-1]:.4f}')
    print('-' * 30)
    plt.plot(y_test.index, y_test, label='Actual Volatility', alpha=0.5)
    plt.plot(y_test.index, predictions, label='Predicted Volatility', color='red', alpha=0.7)
    plt.title(f'{TICKER} True Range Prediction')
    plt.legend()
    plt.show()

    return next_prediction, split

def main():
    df = get_data()
    df = add_feature(df)
    df = add_target(df)
    X, y = test_hypothesis(df)
    next_prediction, split = train_and_test_model(X, y)

    return df, next_prediction, split

if __name__ == '__main__':
    main()