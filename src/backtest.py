import pandas as pd

def walk_forward(df, horizon, start_date, end_date, forecast_funct,  fit_end_date=None, target_end_date=None, **kwargs):
    dates = df.index
    start_idx = dates.searchsorted(start_date)
    end_idx = dates.searchsorted(end_date)
    fit_end_idx = None if fit_end_date is None else dates.searchsorted(fit_end_date, side="right") - 1
    # Freezes fitting, no fits after 30 Sep 2026.

    records = []
    for i in range(start_idx, end_idx):
        target_idx = i + horizon
        if target_idx >= len(dates):
            break
        if target_end_date is not None and dates[target_idx] > pd.Timestamp(target_end_date):
            break # Last forecast shouldnt be beyond 23 Oct 2023
        fit_idx = i if fit_end_idx is None else min(i, fit_end_idx)
        forecast = forecast_funct(df, i, horizon, fit_idx, **kwargs) #models from models.py > returns a pred price
        actual = df["brent_spot"].iloc[target_idx]
        origin_price = df["brent_spot"].iloc[i]
        records.append({
            "origin_date": dates[i],
            "target_date": dates[target_idx],
            "origin_price": origin_price,
            "forecast": forecast,
            "actual": actual,
        })

    return pd.DataFrame(records)
