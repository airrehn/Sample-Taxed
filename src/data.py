import pandas as pd
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

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

EXOG_SERIES = {
    "DCOILWTICO": "wti",
    "DTWEXBGS": "usd",
    "VIXCLS": "vix",
    "OVXCLS": "ovx",  # oil volatility index ("VIX for oil")
    "USEPUINDXD": "epu", # US Policy Uncertainty index
}

def load_sp500(start="1987-01-01", end=None):
    df = yf.download("^GSPC", start=start, end=end, progress=False)
    df = df[["Close"]]
    df.columns = ["sp500"]
    return df

def load_gpr():
    response = requests.get(GPR_URL, timeout=60)  # geopolitical risk index
    response.raise_for_status()
    df = pd.read_excel(io.BytesIO(response.content))
    df = df.set_index("date")[["GPRD"]]
    return df.rename(columns={"GPRD": "gpr"})

def load_exogenous(start="1987-01-01", end=None):
    df = web.DataReader(list(EXOG_SERIES), "fred", start, end)
    return df.rename(columns=EXOG_SERIES)


def load_dataset(start="1987-01-01", end=None):
    brent = load_brent_spot(start=start, end=end)
    exog = load_exogenous(start=start, end=end)
    sp500 = load_sp500(start=start, end=end)
    gpr = load_gpr()
    combined = brent.join([exog, sp500, gpr], how="outer").ffill(limit=5)
    return combined.loc[brent.index]