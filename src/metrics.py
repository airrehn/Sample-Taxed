import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats
import matplotlib.pyplot as plt



def accuracy_metrics(results):
    err = results["actual"] - results["forecast"]
    mae = err.abs().mean() # mean abs error
    rmse = (err ** 2).mean() ** 0.5 # root mse
    mape = (err.abs() / results["actual"]).mean() * 100 # mean abs percentage err
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def directional_metrics(results):
    actual_dir = np.sign(results["actual"] - results["origin_price"])  # +1 if price rose, -1 if it fell (like a column of labels)
    forecast_dir = np.sign(results["forecast"] - results["origin_price"])  # same, but for the predicted price

    accuracy = (actual_dir == forecast_dir).mean()  # % of days direction was correctly called
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_dir, forecast_dir, labels=[1, -1], zero_division=0
    )  # scores computed separately for "up" (1) and "down" (-1) calls

    return {
        "directional_accuracy": accuracy, #(TP + TN) / Total entries
        "precision_up": precision[0], "recall_up": recall[0], "f1_up": f1[0],  # how accurate were "up" calls. Measured with Precision, Recall, F1
        "precision_down": precision[1], "recall_down": recall[1], "f1_down": f1[1],  # how accurate were "down" calls.
    }

def economic_value_metrics(results, transaction_cost = 0.0001, horizon = 1):
    results = results.iloc[::horizon]
    forecasted_position= np.sign(results["forecast"] - results["origin_price"])
    gross_return = forecasted_position * (results["actual"] - results["origin_price"]) / results ["origin_price"]

    trades = forecasted_position.diff().fillna(forecasted_position.iloc[0]).abs() # 0 trade: hold, 1 trade: long or short, 2 trades: flip pos
    net_return = gross_return - trades * transaction_cost # default: 0.01% per trade
    portfolio = (1 + net_return).cumprod()
    drawdown = (portfolio/portfolio.cummax()) - 1

    annualized_return = net_return.mean() * (252 / horizon)
    annualized_vol = net_return.std() * np.sqrt(252 / horizon)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan

    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.min(),
        "num_trades": trades.sum(),
    }

def summarise(results, horizon=1):
    summary = {}
    summary.update(accuracy_metrics(results))
    summary.update(directional_metrics(results))
    summary.update(economic_value_metrics(results, horizon=horizon))
    return summary

def compare_models(results_dict, horizon=1):
    rows = []
    for name, results in results_dict.items():
        row = summarise(results, horizon=horizon)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def vol_regime(results, volatility_all):
    vol_sorted = volatility_all.reindex(results["origin_date"]).values #grabs only volatility of dates of interest from the results df.
    median_vol = np.nanmedian(vol_sorted)
    is_high_vol = vol_sorted >= median_vol

    return { # 2 different accuracy metrics split by high and low vol periods
        "high_vol": accuracy_metrics(results[is_high_vol]),
        "low_vol": accuracy_metrics(results[~is_high_vol]),
    }


def compare_vol_regimes(results_dict, volatility): # one row per model, columns = each metric split by vol regime
    rows = {}
    for name, results in results_dict.items():
        regime = vol_regime(results, volatility) # nested dict, e.g. {"high_vol": {"MAE": 2.1, ...}, "low_vol": {"MAE": 0.9, ...}}
        row = {}
        for reg, metrics in regime.items(): # each regime (high_vol / low_vol)
            for metric, value in metrics.items(): # each metric inside that regime (MAE, RMSE, MAPE)
                row[f"{metric}_{reg}"] = value # flatten into one key, e.g. "MAE_high_vol"
        rows[name] = row
    return pd.DataFrame(rows).T # dict keys become columns, so transpose to put models on rows


def diebold_mariano(results_a, results_b, horizon=1, loss="ape"): # Not necessary but useful when models are super close (e.g. horizon 1 model's are ~ 1% from each other), to test if the difference is just noise or statistically significant
    a = results_a.set_index("target_date") # line both models up by the day being predicted
    b = results_b.set_index("target_date")
    if loss == "ape": # absolute % error, matches MAPE
        loss_a = (a["actual"] - a["forecast"]).abs() / a["actual"]
        loss_b = (b["actual"] - b["forecast"]).abs() / b["actual"]
    else: # squared error, matches RMSE
        loss_a = (a["actual"] - a["forecast"]) ** 2
        loss_b = (b["actual"] - b["forecast"]) ** 2
    d = (loss_a - loss_b).dropna() # daily gap, negative = A was better that day
    n = len(d)

    d_centered = d - d.mean()
    variance = (d_centered ** 2).mean()
    for k in range(1, horizon): # h>1 forecasts overlap, so neighbouring gaps are correlated: add that in
        variance += 2 * (d_centered.iloc[k:].values * d_centered.iloc[:-k].values).mean()

    dm_stat = d.mean() / np.sqrt(variance / n) # like a t-stat: average gap / its standard error
    dm_stat *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n) # small-sample correction (Harvey et al.)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n - 1)
    return {"DM_stat": dm_stat, "p_value": p_value, "n": n}

def dm_table(results_dict, benchmark="Random Walk", horizon=1, loss="ape"):
    rows = {}
    for name, results in results_dict.items():
        if name != benchmark:
            rows[name] = diebold_mariano(results, results_dict[benchmark], horizon=horizon, loss=loss)
    return pd.DataFrame(rows).T

def economic_value_staggered(results, horizon, transaction_cost=0.0001):
    """Trade at EVERY origin date, hold `horizon` days, 1/horizon of capital per trade.

    Equivalent to a portfolio of `horizon` staggered sub-strategies. Returns are measured DAILY
    on the combined portfolio, so observations do not overlap and Sharpe = mean/std * sqrt(252).
    """
    # daily price path rebuilt from the backtest output (origins are consecutive trading days)
    px = pd.concat([results.set_index("origin_date")["origin_price"],
                    results.set_index("target_date")["actual"]])
    px = px[~px.index.duplicated()].sort_index()
    ret = px.pct_change()                                    # daily simple return of Brent

    # signal formed at the close of the origin date; +1 long, -1 short, 0 flat
    sig = np.sign(results["forecast"] - results["origin_price"])
    sig.index = results["origin_date"]
    sig = sig.reindex(px.index)                              # NaN after the last origin

    # exposure on day d = average of the signals opened on days d-h ... d-1 (still held on d)
    exposure = sig.shift(1).rolling(horizon).mean()

    turnover = exposure.diff().abs()                         # net trading needed to move exposure
    net = exposure * ret - transaction_cost * turnover

    # only days where all `horizon` sub-positions are live (no ramp-up / ramp-down)
    full = sig.notna().to_numpy() & exposure.notna().to_numpy()
    net = net[full].iloc[1:]
    exposure = exposure[full].iloc[1:]

    ann_ret = net.mean() * 252
    ann_vol = net.std() * np.sqrt(252)
    equity = (1 + net).cumprod()
    return {
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe_ratio": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "avg_gross_exposure": exposure.abs().mean(),
        "pct_time_net_long": (exposure > 0).mean(),
        "annual_turnover": turnover[full].iloc[1:].mean() * 252,
        "n_days": len(net),
    }

def summarise_staggered(results, horizon=1):
    summary = {}
    summary.update(accuracy_metrics(results))
    summary.update(directional_metrics(results))
    summary.update(economic_value_staggered(results, horizon=horizon))
    return summary

def compare_models_staggered(results_dict, horizon=1):
    rows = []
    for name, results in results_dict.items():
        row = summarise_staggered(results, horizon=horizon)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# one fixed colour per model, keyed by the first word of its name (e.g. "VAR FS lag=2" uses "VAR")
ACTUAL_COLOUR = "#1f4e9c"
MODEL_COLOURS = {"Random": "#d62728", "Historical": "#c23fa8", "AR(1)": "#a8730a", "VAR": "#00a087", "XGB": "#7a6ad8",
                 "2022": "#00a087", "2024-25": "#7a6ad8"}   # the same VAR picked on two validation periods

def plot_forecasts(test_results, title, models=None):
    actual = test_results["Random Walk"].set_index("target_date")["actual"]
    if models is None:
        models = list(test_results)   # default: plot every model

    plt.figure(figsize=(11, 5))
    plt.plot(actual, linewidth=3, color=ACTUAL_COLOUR, label="Actual Brent")
    for name in models:
        r = test_results[name]
        plt.plot(r["target_date"], r["forecast"], linewidth=1.5, alpha=0.8, color=MODEL_COLOURS[name.split()[0]], label=name)

    plt.title(title)
    plt.xlabel("Target Date")
    plt.ylabel("Brent Spot Price (USD/barrel)")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(models) + 1, 4))   # below the chart, so it never hides a line
    plt.tight_layout()
    plt.show()
