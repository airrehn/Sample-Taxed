import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats


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

def economic_value_metrics(results, transaction_cost = 0.0001):
    forecasted_position= np.sign(results["forecast"] - results["origin_price"])
    gross_return = forecasted_position * (results["actual"] - results["origin_price"]) / results ["origin_price"]

    trades = forecasted_position.diff().fillna(forecasted_position.iloc[0]).abs() # 0 trade: hold, 1 trade: long or short, 2 trades: flip pos
    net_return = gross_return - trades * transaction_cost # default: 0.1% per trade
    portfolio = (1 + net_return).cumprod()
    drawdown = (portfolio/portfolio.cummax()) - 1

    annualized_return = net_return.mean() * 252
    annualized_vol = net_return.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan

    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.min(),
        "num_trades": trades.sum(),
    }

def summarise(results):
    summary = {}
    summary.update(accuracy_metrics(results))
    summary.update(directional_metrics(results))
    summary.update(economic_value_metrics(results))
    return summary

def compare_models(results_dict):
    rows = []
    for name, results in results_dict.items():
        row = summarise(results)
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

 


