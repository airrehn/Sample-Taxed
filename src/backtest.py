import pandas as pd

def walk_forward(df, horizon, start_date, end_date, forecast_funct, **kwargs):
    dates = df.index
    start_idx = dates.searchsorted(start_date)
    end_idx = dates.searchsorted(end_date)

    records = []
    for i in range(start_idx, end_idx):
        target_idx = i + horizon
        if target_idx >= len(dates):
            break
        forecast = forecast_funct(df, i, horizon, **kwargs) #models from models.py > returns a pred price
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
