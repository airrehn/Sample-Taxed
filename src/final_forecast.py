"""
final_forecast.py
-----------------
Final h-day-ahead Brent forecast from ANY model in models.py.

    direct_report()   -> the "Direct forecast" model (R_{t+h} = a + b * R_t) with the full
                         OLS table, parameter CI and prediction interval (replaces cell 20).
    final_forecast()  -> generic: point forecast from any model + GARCH prediction interval.
    make_model()      -> builds the model function by name, so you only change one string.

If origin_date is None the origin is the LAST row of `df`, so re-running after the data
reloads gives the forecast from the newest observation automatically.
"""
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy import stats

try:                                   # inside your project (notebooks use src.*)
    from src.models import (random_walk_forecast, direct_forecast, direct_lags_forecast,
                            xgb_direct_forecast, combination_forecast)
except ImportError:                    # standalone
    from models import (random_walk_forecast, direct_forecast, direct_lags_forecast,
                        xgb_direct_forecast, combination_forecast)


# --------------------------------------------------------------------------- helpers
def _origin_index(df, origin_date=None):
    """Position of the last trading day on or before origin_date (None = last row)."""
    if origin_date is None:
        return len(df) - 1
    idx = df.index.searchsorted(pd.Timestamp(origin_date), side="right") - 1
    if idx < 0:
        raise ValueError(f"{origin_date} is before the first date in the data")
    return idx


def _target_date(origin_date, horizon):
    return pd.bdate_range(origin_date, periods=horizon + 1)[-1]   # approx: ignores holidays


# --------------------------------------------------------------------------- model factory
def make_model(name, horizon, lookback=3000, **kw):
    """Return f(df, origin_idx, horizon, fit_idx) -> price forecast.

    Works with the `data` frame (brent_spot + log_return). Mean-revert and ridge need the
    wider frames df_mr / df_rr from your notebooks: pass partial(spread_mr_forecast, ...) or
    partial(direct_ridge_forecast, ...) straight into final_forecast() together with that frame.
    """
    models = {
        "random_walk": random_walk_forecast,
        "direct":      partial(direct_forecast, lookback=lookback),
        # nonoverlap=True matches the regression shown in your final-forecast cell
        "direct_lags": partial(direct_lags_forecast, short=kw.get("short", 1),
                               long=kw.get("long", horizon), lookback=lookback,
                               nonoverlap=kw.get("nonoverlap", True)),
        "xgboost":     partial(xgb_direct_forecast, lookback=lookback),
        "combination": partial(combination_forecast, weights=kw.get("weights")),
    }
    if name not in models:
        raise KeyError(f"unknown model '{name}'. Choose from {list(models)}")
    return models[name]


# --------------------------------------------------------------------------- GARCH interval
def garch_interval_forecast(df, origin_idx, horizon, fit_idx=None, lookback=3000,
                            center_fn=None, level=0.95, vol="garch", gjr=False):
    """Prediction interval for Brent at origin+horizon (same logic as your notebook cell).
    centre = log-return forecast of center_fn (None = random walk);
    width  = sqrt of summed h-step GARCH(1,1)-t variances (vol="constant": sd*sqrt(h))."""
    fit_idx = origin_idx if fit_idx is None else fit_idx
    assert fit_idx <= origin_idx, "fit_idx must not be after origin_idx (lookahead)"
    start = fit_idx - lookback + 1
    assert start >= 0, "not enough history for the lookback window"

    r = df["log_return"].to_numpy(float)
    assert not np.isnan(r[start: origin_idx + 1]).any(), "NaNs in return window"
    p_t = df["brent_spot"].iloc[origin_idx]

    used = vol
    if vol == "garch":
        spec = dict(mean="Zero", vol="GARCH", p=1, o=int(gjr), q=1, dist="t", rescale=False)
        fit = arch_model(r[start: fit_idx + 1] * 100, **spec).fit(disp="off")
        if fit.convergence_flag != 0:
            used = "constant (garch failed)"
        else:
            filt = arch_model(r[start: origin_idx + 1] * 100, **spec).fix(fit.params)
            var_path = filt.forecast(horizon=horizon, reindex=False).variance.to_numpy()[-1]
            sigma_h = np.sqrt(var_path.sum()) / 100
    if used != "garch":
        sigma_h = r[start: fit_idx + 1].std(ddof=1) * np.sqrt(horizon)

    pred_ret = 0.0 if center_fn is None else np.log(center_fn(df, origin_idx, horizon, fit_idx) / p_t)
    z = stats.norm.ppf(0.5 + level / 2)
    return {"forecast": p_t * np.exp(pred_ret),
            "lower": p_t * np.exp(pred_ret - z * sigma_h),
            "upper": p_t * np.exp(pred_ret + z * sigma_h),
            "sigma_h": sigma_h, "vol_used": used}


# --------------------------------------------------------------------------- generic final forecast
def final_forecast(df, model_fn, horizon, origin_date=None, lookback=3000,
                   level=0.95, vol="garch", verbose=True):
    """Final forecast from any model_fn(df, origin_idx, horizon, fit_idx) -> price.

    The model is fitted ONLY on data known at the origin (fit_idx = origin_idx),
    exactly as in the walk-forward backtest.
    """
    origin_idx = _origin_index(df, origin_date)
    o_date = df.index[origin_idx]
    p0 = float(df["brent_spot"].iloc[origin_idx])

    point = float(model_fn(df, origin_idx, horizon, origin_idx))          # model runs once
    iv = garch_interval_forecast(df, origin_idx, horizon, origin_idx, lookback,
                                 center_fn=lambda *_: point, level=level, vol=vol)

    out = {"origin_date": o_date, "origin_price": p0, "horizon": horizon,
           "target_date": _target_date(o_date, horizon), "forecast": point,
           "pred_log_return": float(np.log(point / p0)),
           "lower": iv["lower"], "upper": iv["upper"], "level": level,
           "sigma_h": iv["sigma_h"], "vol_used": iv["vol_used"], "random_walk": p0}

    elapsed = len(df) - 1 - origin_idx          # how the forecast is doing so far
    if elapsed > 0:
        out["elapsed_days"] = elapsed
        out["latest_date"] = df.index[-1]
        out["latest_price"] = float(df["brent_spot"].iloc[-1])
        out["realised_log_return"] = float(np.log(out["latest_price"] / p0))

    if verbose:
        _print_forecast(out)
    return out


def _print_forecast(o):
    print(f"Origin {o['origin_date'].date()}   price = {o['origin_price']:.2f}")
    print(f"Horizon: {o['horizon']} trading days  (target ~ {o['target_date'].date()})")
    print(f"Predicted {o['horizon']}-day log return = {o['pred_log_return']:+.4f}")
    print(f"Point forecast : {o['forecast']:.2f}   (random walk: {o['random_walk']:.2f})")
    print(f"{o['level']:.0%} prediction interval ({o['vol_used']}): "
          f"{o['lower']:.2f} to {o['upper']:.2f}")
    if "elapsed_days" in o:
        print(f"\n{o['elapsed_days']} trading days elapsed. Latest {o['latest_date'].date()}: "
              f"{o['latest_price']:.2f} (realised log return {o['realised_log_return']:+.4f})")


# --------------------------------------------------------------------------- Direct model report
def direct_report(df, horizon, origin_date=None, lookback=3000, level=0.95, verbose=True):
    """Direct forecast:  R_{t+h} = a + b * R_t,   R_t = sum of the last h daily log returns.

    Rebuilds the training set exactly as direct_forecast() does, so the OLS table, the
    parameter uncertainty and the prediction interval can be reported, then asserts that the
    point forecast equals the function used in the backtest.
    """
    origin_idx = _origin_index(df, origin_date)
    fit_idx = origin_idx
    o_date = df.index[origin_idx]
    assert fit_idx - lookback + 1 - horizon >= 0, "not enough history"

    R = df["log_return"].rolling(horizon).sum()
    t_start, t_end = fit_idx - lookback + 1, fit_idx - horizon        # last target = R[fit_idx]
    X = R.iloc[t_start: t_end + 1].values
    Y = R.iloc[t_start + horizon: t_end + horizon + 1].values

    fit = sm.OLS(Y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    beta = np.asarray(fit.params)
    x0 = np.array([1.0, R.iloc[origin_idx]])

    pred_ret = float(x0 @ beta)
    se_mean = float(np.sqrt(x0 @ np.asarray(fit.cov_params()) @ x0))   # uncertainty of E[return]
    sigma = float(np.sqrt(fit.scale))                                  # residual sd of an h-day return
    sd_total = float(np.sqrt(se_mean ** 2 + sigma ** 2))
    z = stats.norm.ppf(0.5 + level / 2)
    p0 = float(df["brent_spot"].iloc[origin_idx])
    price = lambda r: p0 * np.exp(r)

    # sanity check against the backtest function
    fc_fn = direct_forecast(df, origin_idx, horizon=horizon, fit_idx=fit_idx, lookback=lookback)
    assert np.isclose(price(pred_ret), fc_fn), "report and direct_forecast() disagree"

    ci = np.asarray(fit.conf_int(alpha=1 - level))
    table = pd.DataFrame({"coef": beta, "HAC se": np.asarray(fit.bse), "t": np.asarray(fit.tvalues),
                          "p-value": np.asarray(fit.pvalues), "CI low": ci[:, 0], "CI high": ci[:, 1]},
                         index=["const (a)", f"R_past_{horizon}d (b)"])

    out = {"origin_date": o_date, "origin_price": p0, "horizon": horizon,
           "target_date": _target_date(o_date, horizon),
           "feature_R": float(x0[1]), "pred_log_return": pred_ret,
           "forecast": float(price(pred_ret)),
           "mean_ci": (float(price(pred_ret - z * se_mean)), float(price(pred_ret + z * se_mean))),
           "lower": float(price(pred_ret - z * sd_total)), "upper": float(price(pred_ret + z * sd_total)),
           "level": level, "params": table, "r2": float(fit.rsquared),
           "wald_p": float(fit.f_pvalue), "n_obs": len(Y),
           "n_independent": len(Y) // horizon, "random_walk": p0}

    elapsed = len(df) - 1 - origin_idx
    if elapsed > 0:
        out.update(elapsed_days=elapsed, latest_date=df.index[-1],
                   latest_price=float(df["brent_spot"].iloc[-1]),
                   realised_log_return=float(np.log(df["brent_spot"].iloc[-1] / p0)))

    if verbose:
        print(fit.summary())
        print("\n", table.round(4))
        print(f"\nobs = {out['n_obs']}  (~{out['n_independent']} non-overlapping {horizon}-day windows)")
        print(f"R2 = {out['r2']:.4f}   Wald p-value = {out['wald_p']:.4f}")
        print(f"\nOrigin {o_date.date()}  price = {p0:.2f}")
        print(f"Feature: past {horizon}-day log return R_t = {out['feature_R']:+.4f}")
        print(f"Predicted {horizon}-day log return = {pred_ret:+.4f}  "
              f"(expected-return {level:.0%} CI {pred_ret - z * se_mean:+.4f} to {pred_ret + z * se_mean:+.4f})")
        print(f"Point forecast for ~{out['target_date'].date()}: {out['forecast']:.2f}")
        print(f"{level:.0%} prediction interval (OLS, normal): {out['lower']:.2f} to {out['upper']:.2f}")
        print(f"Random-walk forecast: {p0:.2f}")
        if elapsed > 0:
            print(f"\n{elapsed} trading days elapsed. Latest {out['latest_date'].date()}: "
                  f"{out['latest_price']:.2f} (realised log return {out['realised_log_return']:+.4f})")
    return out
