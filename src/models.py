import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

def random_walk_forecast(df, origin_idx, horizon, fit_idx):
    return df["brent_spot"].iloc[origin_idx] # prediction is just today's price

def historical_mean_forecast(df, origin_idx, horizon, fit_idx, lookback=500):
    window = df["log_return"].iloc[fit_idx - lookback + 1: fit_idx  + 1] # Window up to 30 Sep 2026 max
    predicted_return = window.mean()
    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert return to latest predicted price

def ar1_forecast(df, origin_idx, horizon, fit_idx, lookback = 3000):
    window = df["log_return"].iloc[fit_idx - lookback + 1: fit_idx  + 1]
    fitted = AutoReg(window.values, lags=1).fit()
    c, phi = fitted.params[0], fitted.params[1]

    last_return = df["log_return"].iloc[origin_idx] # takes latest return - 1 day
    predicted_return = c + phi * last_return # predict latest return (AR1 formula basically)

    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert return to latest predicted price

def var_forecast(df, origin_idx, horizon, fit_idx, cols, lags=3, lookback=3000):
    window = df[cols].iloc[fit_idx - lookback + 1: fit_idx + 1]
    fitted = VAR(window.values).fit(lags)
    recent = df[cols].iloc[origin_idx - lags + 1: origin_idx + 1].values  # latest `lags` days of all variables (in col), used to forecast
    predicted_return =  fitted.forecast(recent, steps=1)[0,0] #brents (log) return tomorrow in the table produced.
    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert return to latest predicted price