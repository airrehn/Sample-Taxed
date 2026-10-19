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
    actual_dir = np.sign(results["actual"]-results["origin_price"])
    forecast_dir = np.sign(results["forecast"]-results["origin_price"])

    accuracy = (actual_dir == forecast_dir).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_dir, forecast_dir, labels=[1,-1], zero_division=0
    )

    return {
        "directional_accuracy": accuracy,
        "precision_up": precision[0], "recall_up": recall[0], "f1_up": f1[0],
        "precision_down": precision[1], "recall_down": recall[1], "f1_down": f1[1],
    }