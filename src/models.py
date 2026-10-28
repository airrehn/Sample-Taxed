import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

from xgboost import XGBRegressor

def random_walk_forecast(df, origin_idx, horizon, fit_idx):
    return df["brent_spot"].iloc[origin_idx] # prediction is just today's price

def historical_mean_forecast(df, origin_idx, horizon, fit_idx, lookback=500):
    window = df["log_return"].iloc[fit_idx - lookback + 1: fit_idx  + 1] # Window up to 30 Sep 2026 max
    predicted_return = horizon * window.mean()   # average daily return × number of days
    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert return to latest predicted price

def ar1_forecast(df, origin_idx, horizon, fit_idx, lookback = 3000):
    window = df["log_return"].iloc[fit_idx - lookback + 1: fit_idx  + 1]
    fitted = AutoReg(window.values, lags=1).fit()
    c, phi = fitted.params[0], fitted.params[1]

    r = df["log_return"].iloc[origin_idx]   # today's return, the starting point
    predicted_return = 0
    for _ in range(horizon):
        r = c + phi * r # predict the next day from the previous one
        # e.g.
        # day 1: 0.06 (phi) × 0.02 (r) = 0.0012 (new r)
        # day 2: 0.06 × 0.0012 (new r) = 0.000072
        # day 3: 0.06 × 0.000072 ≈ 0.000004
        predicted_return += r
        # 5-day log return = 0.0012 + 0.000072 + 0.000004 + 0 + 0 ≈ 0.0013

    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert back return to latest predicted price

def var_forecast(df, origin_idx, horizon, fit_idx, cols, lags=3, lookback=3000):
    window = df[cols].iloc[fit_idx - lookback + 1: fit_idx + 1]
    fitted = VAR(window.values).fit(lags)
    recent = df[cols].iloc[origin_idx - lags + 1: origin_idx + 1].values  # pulls latest `lags` days of all variables (in col), used to forecast
    predicted_return = fitted.forecast(recent, steps=horizon)[:, 0].sum() #brents (log return) tomorrow in the table produced, sum the log returns
    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert return to latest predicted price

def xgb_forecast(df, origin_idx, horizon, fit_idx, features, lookback=3000, n_estimators=200, max_depth=2, learning_rate=0.05, min_child_weight=1):
    X_all = features.drop(columns="target")
    y_all = df["log_return"].rolling(horizon).sum().shift(-horizon) # sums log return over h days before it, then shifts df back by h days, so it aligns nicely as h-th days log return. 
    train_X = X_all.iloc[fit_idx - lookback: fit_idx - horizon + 1] # last row's h-day target ends exactly at fit_idx
    train_y = y_all.iloc[fit_idx - lookback: fit_idx - horizon + 1]
    keep = train_X.notna().all(axis=1) & train_y.notna() # same job as dropna()
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, random_state=0, n_jobs=1)
    model.fit(train_X[keep], train_y[keep])
    today = X_all.iloc[[origin_idx]]
    predicted_return = model.predict(today)[0] # it returns a list of 1 item, just specify the [0] anyways
    price_now = df["brent_spot"].iloc[origin_idx]
    return price_now * np.exp(predicted_return)