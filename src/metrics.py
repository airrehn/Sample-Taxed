import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

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

def vol_regime(results, volatility_all):
    vol_sorted = volatility_all.reindex(results["origin_date"]).values #grabs only volatility of dates of interest from the results df.
    median_vol = np.nanmedian(vol_sorted)
    is_high_vol = vol_sorted >= median_vol

    return { # 2 different accuracy metrics split by high and low vol periods
        "high_vol": accuracy_metrics(results[is_high_vol]),
        "low_vol": accuracy_metrics(results[~is_high_vol]),
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

def summarise(results, volatility = None):
    summary = {}
    summary.update(accuracy_metrics(results))
    summary.update(directional_metrics(results))
    summary.update(economic_value_metrics(results))
    if volatility is not None:
        summary["regime"] = vol_regime(results, volatility)
    return summary

def compare_models(results_dict, volatility=None):
    rows = []
    for name, results in results_dict.items():
        row = summarise(results, volatility=volatility)
        row = {k: v for k, v in row.items() if k != "regime"}
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


 


