import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def random_walk_forecast(df, origin_idx, horizon):
    return df["brent_spot"].iloc[origin_idx]

def ar1_forecast(df, origin_idx, horizon, lookback = 500):
    window = df["log_return"].iloc[origin_idx - lookback + 1: origin_idx  + 1]
    fitted = AutoReg(window.values, lags=1).fit()
    c, phi = fitted.params[0], fitted.params[1]

    last_return = window.iloc[-1] # takes latest return - 1 day
    predicted_return = c + phi * last_return # predict latest return (AR1 formula basically)

    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert back to latest predicted price
