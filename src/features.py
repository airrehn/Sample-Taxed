import numpy as np

def log_returns(df, col="brent_spot"):
    return np.log(df[col]).diff()