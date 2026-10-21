import numpy as np
from arch import arch_model

def log_returns(df, col="brent_spot"):
    return np.log(df[col]).diff() # ln(P_t / P_t-1) i.e. log return = ln(P_t) - ln(P_t-1)

def realised_volatility(df, col="log_return", window = 20): # simple proxy for how volatile it's been recently, later will built GARCH
    return df[col].rolling(window).std()

def garch_volatility(df, col="log_return"):
    returns_pct = df[col] * 100 # scaled as percentage (2.0 instead of 0.02)
    model = arch_model(returns_pct, vol="GARCH", p=1, q=1, mean = "Zero", dist="normal")
    fitted = model.fit(disp="off")
    return fitted.conditional_volatility/100 # undoes the pct scaling
