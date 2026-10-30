import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

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
    X_all = features
    y_all = df["log_return"].rolling(horizon).sum().shift(-horizon) # sums log return over h days before it, then shifts df back by h days, so it aligns nicely as h-th days log return. 
    train_X = X_all.iloc[fit_idx - lookback: fit_idx - horizon + 1] # last row's h-day target ends exactly at fit_idx
    train_y = y_all.iloc[fit_idx - lookback: fit_idx - horizon + 1]
    keep = train_X.notna().all(axis=1) & train_y.notna() # same job as dropna(). makes Boolean flags for each row, we only train on the True rows.
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, random_state=0, n_jobs=1)
    model.fit(train_X[keep], train_y[keep])
    today = X_all.iloc[[origin_idx]]
    predicted_return = model.predict(today)[0] # it returns a list of 1 item, just specify the [0] anyways
    price_now = df["brent_spot"].iloc[origin_idx]
    return price_now * np.exp(predicted_return)

def direct_forecast(df, origin_idx, horizon=60, fit_idx=None, lookback=3000):
    fit_idx = origin_idx if fit_idx is None else fit_idx
    assert fit_idx <= origin_idx, "fit_idx must not be after origin_idx"
    assert fit_idx - lookback + 1 - horizon >= 0, "not enough history"

    # R_t = sum of the last `horizon` daily log returns (ends at t)
    R = df["log_return"].rolling(horizon).sum()

    # Training pairs: predictor R_t, target R_{t+horizon}; target must be known by fit_idx
    t_start = fit_idx - lookback + 1
    t_end = fit_idx - horizon
    X = R.iloc[t_start: t_end + 1].values
    Y = R.iloc[t_start + horizon: t_end + horizon + 1].values

    fit = sm.OLS(Y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    a, b = fit.params

    pred_ret = a + b * R.iloc[origin_idx]
    p_t = df["brent_spot"].iloc[origin_idx]

    return p_t * np.exp(pred_ret)


def direct_lags_forecast(df, origin_idx, horizon=60, fit_idx=None, lookback=3000,
                        short=1, long=60, nonoverlap=False, use_intercept=True):
    
    fit_idx = origin_idx if fit_idx is None else fit_idx
    assert fit_idx <= origin_idx, "fit_idx must not be after origin_idx (lookahead)"
    start = fit_idx - lookback + 1
    assert start >= 0, "not enough history for the lookback window"

    lp = np.log(df["brent_spot"].to_numpy(dtype=float))

    def feats(t):
        r_short = lp[t] - lp[t - short]                  # trailing 5-day return
        r_long = lp[t] - lp[t - long]                    # trailing 60-day return
        if nonoverlap:
            r_long = lp[t - short] - lp[t - long]        # days t-59 to t-5
        return np.column_stack([r_short, r_long])

    # Features known at t; target = log(P_{t+h}/P_t), which must be known by fit_idx
    t = np.arange(start + long, fit_idx - horizon + 1)
    assert len(t) > 4 * horizon, "too few samples; increase lookback"
    X, y = feats(t), lp[t + horizon] - lp[t]
    if use_intercept:
        X = sm.add_constant(X)

    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})

    x0 = feats(np.array([origin_idx]))
    if use_intercept:
        x0 = np.column_stack([np.ones(1), x0])
    pred_ret = float(x0[0] @ fit.params)

    p_t = df["brent_spot"].iloc[origin_idx]
    k = 1 if use_intercept else 0
    return p_t * np.exp(pred_ret)


def spread_mr_forecast(df, origin_idx, horizon=60, fit_idx=None, lookback=1500,
                       winsor=(0.01, 0.99), shrink=1.0):
    fit_idx = origin_idx if fit_idx is None else fit_idx
    assert fit_idx <= origin_idx, "fit_idx must not be after origin_idx (lookahead)"
    start = fit_idx - lookback + 1
    assert start >= 0, "not enough history for the lookback window"

    r = df["log_return"].to_numpy(dtype=float)
    B = df["brent_spot"].to_numpy(dtype=float)
    W = df["wti_spot"].to_numpy(dtype=float)
    assert not np.isnan(r[start: origin_idx + 1]).any(), "NaNs in Brent returns"
    cs = np.cumsum(r)

    # Spread as a fraction of Brent: defined even when WTI <= 0 (Apr 2020)
    sp = (B - W) / B
    lo, hi = np.quantile(sp[start: fit_idx + 1], winsor)   # clip bounds from fit window only
    sp = np.clip(sp, lo, hi)

    # Samples: predictor known at t, target = sum of Brent r[t+1..t+horizon], known by fit_idx
    t = np.arange(start, fit_idx - horizon + 1)
    assert len(t) > 4 * horizon, "too few samples; increase lookback"
    X, y = sp[t], cs[t + horizon] - cs[t]

    fit = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    a, b = fit.params

    pred_ret = shrink * (a + b * sp[origin_idx])
    p_t = B[origin_idx]
    
    return p_t * np.exp(pred_ret)

## XGBoost direct forecast: features = lagged returns, momentum, volatility
def _make_features(r):
    s = pd.Series(r)
    F = pd.DataFrame({f"lag{k}": s.shift(k) for k in range(5)})  # last 5 daily returns
    for w in (5, 20, 60):
        F[f"ret{w}"] = s.rolling(w).sum()    # momentum over several horizons
    for w in (20, 60):
        F[f"vol{w}"] = s.rolling(w).std()    # volatility regime
    return F

def xgb_direct_forecast(df, origin_idx, horizon=60, fit_idx=None, lookback=3000,
                        val_frac=0.2, seed=0):
    
    fit_idx = origin_idx if fit_idx is None else fit_idx
    assert fit_idx <= origin_idx, "fit_idx must not be after origin_idx (lookahead)"
    start = fit_idx - lookback + 1
    assert start >= 0, "not enough history for the lookback window"

    r = df["log_return"].to_numpy(dtype=float)[: origin_idx + 1]
    cs = np.cumsum(np.nan_to_num(r))
    F = _make_features(r)

    # Sample at time t: features known at t, target = sum of r[t+1 .. t+horizon]
    t_all = np.arange(max(start, 60), fit_idx - horizon + 1)   # 60 = rolling warm-up
    X_all = F.iloc[t_all].to_numpy()
    y_all = cs[t_all + horizon] - cs[t_all]
    n = len(t_all)

    # Purged chronological split: gap of `horizon` between train and validation
    n_val = int(n * val_frac)
    n_tr = n - n_val - horizon
    assert n_tr > 250, "too few samples; increase lookback"
    tr, va = np.arange(n_tr), np.arange(n - n_val, n)

    params = dict(learning_rate=0.02, max_depth=2, min_child_weight=20,
                  subsample=0.7, colsample_bytree=0.7, reg_lambda=10.0,
                  random_state=seed, n_jobs=-1)

    # Step 1: early stopping on the validation block picks the number of trees
    m = XGBRegressor(n_estimators=1500, early_stopping_rounds=50, **params)
    m.fit(X_all[tr], y_all[tr], eval_set=[(X_all[va], y_all[va])], verbose=False)
    best_n = m.best_iteration + 1

    # Step 2: refit on all samples with that number of trees
    final = XGBRegressor(n_estimators=best_n, **params).fit(X_all, y_all)

    x0 = F.iloc[[origin_idx]]
    assert not x0.isna().any().any(), "NaNs in origin features"
    pred_ret = float(final.predict(x0)[0])

    p_t = df["brent_spot"].iloc[origin_idx]
    
    return p_t * np.exp(pred_ret)


## Direct ridge regression: features = lagged returns, momentum, volatility
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def make_features(df, lvl_cols, windows):
    lb = np.log(df["brent_spot"])
    F = {"spread": (df["brent_spot"] - df["wti_spot"]) / df["brent_spot"]}   # WTI-safe
    for w in windows:
        F[f"brent_{w}d"] = lb.diff(w)                      # Brent trailing log return
        for c in lvl_cols:
            F[f"{c}_{w}d"] = df[c].diff(w)                 # change in log level
    return pd.DataFrame(F)

def direct_ridge_forecast(df, origin_idx, horizon=60, fit_idx=None, lookback=3000,
                          lvl_cols=("usd", "sp500", "ovx", "vix"), windows=(20, 60),
                          alphas=np.logspace(1, 5, 9)):
    
    fit_idx = origin_idx if fit_idx is None else fit_idx
    assert fit_idx <= origin_idx, "fit_idx must not be after origin_idx (lookahead)"
    start = fit_idx - lookback + 1
    assert start >= 0, "not enough history for the lookback window"

    lb = np.log(df["brent_spot"])
    F = make_features(df, lvl_cols, windows)
    y = lb.shift(-horizon) - lb # next-60-day Brent log return

    # Features known at t; target must be known by fit_idx
    t = np.arange(start + max(windows), fit_idx - horizon + 1)
    assert len(t) > 4 * horizon, "too few samples; increase lookback"
    X, yt = F.iloc[t].to_numpy(), y.iloc[t].to_numpy()
    assert not (np.isnan(X).any() or np.isnan(yt).any()), "NaNs in fit window"
    x0 = F.iloc[[origin_idx]].to_numpy()
    assert not np.isnan(x0).any(), "NaNs in origin features"

    # Scale -> clip at +/-4 sd -> ridge, all refitted inside each CV fold (no leakage)
    pipe = make_pipeline(StandardScaler(),
                         FunctionTransformer(lambda Z: np.clip(Z, -4, 4)),
                         Ridge(fit_intercept=False))
    cv = TimeSeriesSplit(n_splits=5, gap=horizon)          # gap purges overlapping targets
    gs = GridSearchCV(pipe, {"ridge__alpha": alphas}, cv=cv,
                      scoring="neg_root_mean_squared_error").fit(X, yt)

    pred_ret = float(gs.predict(x0)[0])
    p_t = df["brent_spot"].iloc[origin_idx]
    
    return p_t * np.exp(pred_ret)

## Combination model
from functools import partial
from scipy import stats

def combination_forecast(df, origin_idx, horizon=60, fit_idx=None, members=None, weights=None):
    """Weighted average of member forecasts.
    Members are existing forecast functions with signature (df, origin_idx, horizon, fit_idx).
    Averaging is done on the predicted 60-day LOG return, then converted back to a price,
    so the combination is a geometric mean of the member price forecasts."""
    fit_idx = origin_idx if fit_idx is None else fit_idx
    assert fit_idx <= origin_idx, "fit_idx must not be after origin_idx (lookahead)"

    if members is None:   # choose members BEFORE looking at results; keep RW in as a shrinkage anchor
        members = [
            random_walk_forecast,
            partial(direct_lags_forecast, short=1, long=horizon, lookback=3000),
            partial(xgb_direct_forecast, lookback=3000),
            partial(direct_forecast, lookback=3000),
        ]
        
    w = np.full(len(members), 1 / len(members)) if weights is None else np.asarray(weights, float)
    assert len(w) == len(members) and np.isclose(w.sum(), 1), "weights must match members and sum to 1"

    p_t = df["brent_spot"].iloc[origin_idx]
    rets = np.array([np.log(fn(df, origin_idx, horizon=horizon, fit_idx=fit_idx) / p_t)
                     for fn in members])
    return p_t * np.exp(w @ rets)

