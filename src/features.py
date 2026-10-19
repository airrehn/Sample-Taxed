import numpy as np

def log_returns(df, col="brent_spot"):
    return np.log(df[col]).diff() # ln(P_t / P_t-1) i.e. log return = ln(P_t) - ln(P_t-1)

def realised_volatility(df, col="log_return", window = 20): # simple proxy for how volatile it's been recently, later will built GARCH
    return df[col].rolling(window).std()