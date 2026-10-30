import pandas as pd
import numpy as np
import pandas_datareader.data as web
import yfinance as yf

import io
import requests


def load_brent_spot(start="2000-01-01", end=None):
    df = web.DataReader("DCOILBRENTEU", "fred", start, end)
    df = df.rename(columns={"DCOILBRENTEU": "brent_spot"})
    df = df.dropna()
    return df

def load_brent_futures(start="2000-01-01", end=None):
    df = yf.download("BZ=F", start=start, end= end, progress=False)
    df = df[["Close"]]
    df.columns = ["brent_futures"]
    return df


def load_sp500(start="1987-01-01", end=None):
    df = yf.download("^GSPC", start=start, end=end, progress=False)
    df = df[["Close"]]
    df.columns = ["sp500"]
    return df

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

def load_gpr():
    response = requests.get(GPR_URL, timeout=60)  # geopolitical risk index
    response.raise_for_status()
    df = pd.read_excel(io.BytesIO(response.content))
    df = df.set_index("date")[["GPRD"]]
    return df.rename(columns={"GPRD": "gpr"})

EXOG_SERIES = {
    "DCOILWTICO": "wti",
    "DTWEXBGS": "usd",
    "VIXCLS": "vix",
    "OVXCLS": "ovx",  # oil volatility index ("VIX for oil")
    "USEPUINDXD": "epu", # US Policy Uncertainty index
}

def load_exogenous(start="1987-01-01", end=None):
    df = web.DataReader(list(EXOG_SERIES), "fred", start, end)
    return df.rename(columns=EXOG_SERIES)


INVENTORY_URL = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"

def load_inventory():
    response = requests.get(INVENTORY_URL, timeout=60)  # EIA weekly US crude stocks, excluding SPR (thousand barrels)
    response.raise_for_status()
    df = pd.read_excel(io.BytesIO(response.content), sheet_name="Data 1", skiprows=2)
    df.columns = ["week_ending", "stocks"]
    df.index = pd.to_datetime(df["week_ending"]) + pd.Timedelta(days=6)  
    df.index.name = "public_info_from"
    # [week_ending] is measured on Fri (and uses fridays date), but its only published the following Wed (made public); so we use the following Thurs to be safe
    df["invt_weekly_change"] = np.log(df["stocks"]).diff()   # week-on-week % change; must be done on weekly rows, before going daily
    return df[["invt_weekly_change"]]


def load_dataset(start="1987-01-01", end=None):
    brent = load_brent_spot(start=start, end=end)
    exog = load_exogenous(start=start, end=end)
    sp500 = load_sp500(start=start, end=end)
    gpr = load_gpr()
    combined = brent.join([exog, sp500, gpr], how="outer").ffill(limit=5)
    combined = combined.loc[brent.index]
    inv = load_inventory()
    combined["invt_weekly_change"] = inv["invt_weekly_change"].reindex(combined.index, method="ffill") # published week-on-week % change (repeats until the next release)
    return combined