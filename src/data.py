import pandas as pd
import pandas_datareader.data as web
import yfinance as yf


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