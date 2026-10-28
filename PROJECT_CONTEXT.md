# Brent Oil Price Forecasting: Project Context

> **How to use this file:** it is a self-contained summary of our QF5210 group project as of **29 Oct 2026**. It covers the brief, the rules the lecturer clarified, the data, the code (full source in the appendix), the results so far and what is still open. Read it fully before suggesting changes. Most of the important decisions are explained in sections 2 and 5.

---

## 1. The assignment (QF5210 Part II, alternative group project)

**Course:** QF5210 Financial Time Series (NUS). The alternative group project is worth 25%.

**Objective (from the brief):**
- Build a data-driven forecasting system for crude oil prices and/or returns, 1 to 60 days ahead, using daily data.
- Include at least one major benchmark (WTI or Brent). Other variables are optional: oil benchmarks, energy indices, futures, ETFs, commodity prices, financial/macro variables.
- Models must **update automatically** as new daily data arrives.
- We are encouraged to compare conventional time-series models with ML/AI.
- Any language is allowed (we use Python).

**Approaches listed in the brief:**
- ARIMA, VAR, GARCH
- Random forest, gradient boosting, XGBoost
- LSTM, Transformer
- Regime switching
- Multi-factor models
- News/sentiment
- Hybrids/ensembles

We should justify our choice of target, predictors, method and benchmarks.

**Evaluation (brief, page 9).** The emphasis is on genuine out-of-sample performance. *The test period must not be used for model estimation or parameter selection*, and we must avoid look-ahead bias and leakage. Metrics:
1. **Accuracy:** MAE, RMSE, MAPE.
2. **Directional accuracy:** % of correct up/down calls, plus precision/recall/F1 for up and down.
3. **Relative to benchmarks:** compare against random walk/no-change, historical mean and AR/ARIMA; report the % improvement in MAE/RMSE.
4. **Robustness:** performance in high- vs low-volatility regimes, and stability across out-of-sample windows.
5. **Economic value:** a simple trading rule with P&L, annualised return, Sharpe, max drawdown and transaction costs. This is secondary to forecasting.

> A low in-sample error alone does not count as success.

**Submission (Sat 31 Oct 2026):** a zip on Canvas with:
- editable slides
- data in CSV: the oil series and the main variables, with daily observations
- code
- an optional video

We must document the data sources, variable definitions, sample period, train/validation/test split and preprocessing.

**Presentation:** 8 min plus 2 min Q&A, and every member presents.

---

## 2. Test design (rules clarified with the lecturer)

- **Horizons:** h = 1, 5, 10, 15, 20, 30 and 60 days ahead, counted in **trading days**.
- **Test targets:** forecasts are scored when their **target date** falls between **1 and 23 Oct 2026**, which is 17 trading days.
- **h = 1 and h = 5:**
  - The forecast origin must be **on or after 1 Oct**.
  - The model is fitted once on data up to the **30 Sep 2026 close** and then **frozen** (no refitting in October).
  - New October prices may still be used as inputs, up to each forecast's origin.
  - Number of forecasts with all data in: 16 for h=1 (targets 2–23 Oct) and 12 for h=5 (origins 1–16 Oct, targets 8–23 Oct).
- **h > 5 (10 to 60):**
  - No restriction. The origin can fall before 1 Oct; for example, an h=60 forecast for 1 Oct is made in early July.
  - The lecturer allowed evaluating these horizons over a long walk-forward history, so there are enough test points.
- **Validation (model selection):** walk-forward from 2024-01-01 to 2026-09-30, refitting at each origin. All validation targets must be ≤ 30 Sep 2026.
- **Selection criterion:** **MAPE** is the main criterion, used at every horizon. Tie rule: if two variable sets are within about 0.002 MAPE, pick the simpler one.

**Example (h=5 test):**
- The model is fitted once on data up to 30 Sep.
- At the 1 Oct close it predicts 8 Oct, using 1 Oct's prices as inputs.
- At the 2 Oct close it predicts 9 Oct, and so on until 16 Oct → 23 Oct.

---

## 3. Data

**Target series:** **Brent spot**, from FRED `DCOILBRENTEU` (EIA daily Brent).
- We chose Brent over WTI because WTI went negative on 20 Apr 2020 (−$36.98), which breaks log returns.
- **Target:** the cumulative log return over h days, log(P_{t+h} / P_t), converted back to a price as P_t · exp(predicted return). All metrics are computed on price levels.

**Explanatory variables** (`src/data.py`, `load_dataset()`):

| Column | Source | Meaning | Transform |
|---|---|---|---|
| `brent_spot` | FRED DCOILBRENTEU | Brent spot $/bbl | log return (renamed `log_return`) |
| `wti` | FRED DCOILWTICO | WTI spot | log return |
| `usd` | FRED DTWEXBGS (from 2006) | broad USD index | log return |
| `sp500` | Yahoo `^GSPC` (FRED's SP500 only starts in 2016) | S&P 500 | log return |
| `vix` | FRED VIXCLS | equity volatility index | log level |
| `ovx` | FRED OVXCLS (from 2007-05-10) | oil volatility index | log level |
| `epu` | FRED USEPUINDXD | US economic policy uncertainty | log level |
| `gpr` | Caldara & Iacoviello daily GPR (.xls) | geopolitical risk | log level |

**Preprocessing:**
- Everything is joined on Brent's trading days, with a forward fill of at most 5 days for series that publish with a lag (USD is effectively weekly; GPR has about a 3-day lag).
- **Prices → log returns:** they are non-stationary as levels.
- **Indices → log levels:** ADF rejects a unit root, p ≤ 0.001.
- `dropna` then removes incomplete rows, so the joint sample starts on **2007-05-10**, when OVX begins. That gives about 4,200 rows before 2024.
- **Lookback (training window):** 3,000 trading days (about 12 years).

**Data quirks:**
- FRED/EIA Brent spot updates with a lag of about a week. As of 28 Oct 2026, data ended on 20 Oct, so the test must be rerun once 21–23 Oct appear.
- In October 2026, Brent spot and Brent futures (`BZ=F`) differ by $10–30 (e.g. on 2 Oct: spot 135.51 vs futures 102.25).
- Reading the GPR .xls needs `xlrd`, and is fetched with `requests` because of SSL certificate issues on python.org builds of Python.

---

## 4. Code structure

```
src/data.py       load_brent_spot, load_brent_futures, load_sp500, load_gpr, load_exogenous, load_dataset
src/features.py   log_returns, realised_volatility, garch_volatility (GARCH(1,1)), make_lag_features
src/models.py     random_walk_forecast, historical_mean_forecast, ar1_forecast, var_forecast, xgb_forecast
src/backtest.py   walk_forward (the backtest loop)
src/metrics.py    accuracy / directional / economic metrics, compare_models(results_dict, horizon=1), vol_regime,
                  compare_vol_regimes, diebold_mariano, dm_table
notebooks/aa_forecast1.ipynb   h=1: data checks, ACF/seasonality, VAR selection, XGB tuning, regime split, October test
notebooks/aa_forecast5.ipynb   h=5: non-overlapping ACF check, VAR sweep (XGB sweep next)
```

**`walk_forward(df, horizon, start_date, end_date, forecast_funct, fit_end_date=None, target_end_date=None, **kwargs)`**
- Loops over **origin** dates i from `start_date` (inclusive) to `end_date` (exclusive), with target = i + horizon trading days.
- **`fit_end_date`** freezes fitting. Each model receives `fit_idx = min(origin, fit_end_idx)` and may only fit on data up to `fit_idx`.
- **`target_end_date`** stops the loop once a target would pass that date.
- It stops automatically when the data runs out, which is how partial October data is handled.
- It returns a DataFrame with the columns `origin_date`, `target_date`, `origin_price`, `forecast` and `actual`.

**Model function signature:** `f(df, origin_idx, horizon, fit_idx, **kwargs)`, returning a **price** forecast. All models are horizon-aware:

| Model | How it forecasts h days ahead |
|---|---|
| Random walk | Today's price. "No change" is the same at any horizon. |
| Historical mean | h × the mean daily log return over the lookback |
| AR(1) | Fits c and φ on daily returns, then iterates h steps and sums |
| VAR | statsmodels VAR on the chosen columns; forecasts `steps=horizon` and sums the Brent column (iterated) |
| XGBoost | **Direct.** The target is built inside the function as `log_return.rolling(h).sum().shift(-h)` |

**XGBoost `features`: inputs only.**
- `xgb_forecast` uses **every column** of `features` as an input.
- So never add a `"target"` (future-return) column to it. That would feed the answer in as an input, which is leakage.
- Build `features` with `make_lag_features` and pass it as is.

**XGBoost training window:** training rows end at `fit_idx - horizon + 1`, so the last training row's h-day target ends exactly at the fit date. Without this, at h=5 the training rows would include targets built from returns after the fit date, which is leakage. `target_end_date` does **not** prevent this, because it only limits which forecasts are scored.

**Economic metrics need the horizon: always call `compare_models(results, horizon=h)`.**
- **The problem it fixes:** at h > 1, each row is an h-day return and a new position opens every day. The positions overlap, so each day's move is counted up to h times, and annualising with 252 inflates the Sharpe by about √h.
  - Example: at h=5, the historical mean's Sharpe showed 1.27, against 0.55 once corrected.
- **What `horizon=h` does in `economic_value_metrics`:**
  - It trades only every h-th origin (`results.iloc[::horizon]`): hold h days, then trade again.
  - It annualises with 252 / h holding periods per year.
- **Effect on other metrics:** accuracy and directional metrics still use every row. At h=1 nothing changes.
- **The Sharpe is noisy.** It depends on which day the h-day trading grid starts.
  - Example (h=5, VAR S1 p=3): the 5 possible start days give Sharpes of 0.29, 0.38, 1.10, 0.50 and 0.45.
  - Its standard error over the 2.7-year validation is about ±0.6.
  - So don't select models on Sharpe.

**Template for any horizon** (`df_var` holds the transformed variables plus the Brent price):

```python
import sys; sys.path.append("..")
import numpy as np, pandas as pd
from src.data import load_dataset
from src.features import make_lag_features
from src.backtest import walk_forward
from src.models import random_walk_forecast, historical_mean_forecast, ar1_forecast, var_forecast, xgb_forecast
from src.metrics import compare_models

H = 5
data = load_dataset()
X = pd.DataFrame(index=data.index)
for col in ["brent_spot", "wti", "usd", "sp500"]:
    X[col] = np.log(data[col]).diff()          # prices -> log returns
for col in ["vix", "ovx", "epu", "gpr"]:
    X[col] = np.log(data[col])                 # indices -> log levels
X = X.dropna()
df_var = X.rename(columns={"brent_spot": "log_return"})
df_var["brent_spot"] = data["brent_spot"]      # price column, needed by walk_forward and the models

feats_s1 = make_lag_features(df_var, ["log_return", "wti", "sp500"], lags=3)   # inputs only: no "target" column

val_args  = dict(horizon=H, start_date="2024-01-01", end_date="2026-09-30", target_end_date="2026-09-30")
test_args = dict(horizon=H, start_date="2026-10-01", end_date="2026-10-23",
                 fit_end_date="2026-09-30", target_end_date="2026-10-23")   # h=1 and h=5 rule

results = {
    "Random Walk":     walk_forward(df_var, forecast_funct=random_walk_forecast, **val_args),
    "Historical Mean": walk_forward(df_var, forecast_funct=historical_mean_forecast, lookback=3000, **val_args),
    "AR(1)":           walk_forward(df_var, forecast_funct=ar1_forecast, lookback=3000, **val_args),
    "VAR S1 p=2":      walk_forward(df_var, forecast_funct=var_forecast, cols=["log_return", "wti", "sp500"], lags=2, **val_args),
    "XGB S1":          walk_forward(df_var, forecast_funct=xgb_forecast, features=feats_s1,
                                    n_estimators=50, max_depth=1, min_child_weight=1, **val_args),
}
table = compare_models(results, horizon=H)   # horizon=H is needed for correct economic metrics
for m in ["MAE", "RMSE", "MAPE"]:
    table[f"{m}_vs_RW_%"] = (1 - table[m] / table.loc["Random Walk", m]) * 100   # positive = better than RW
```

**Important for h ≥ 10:** add `target_end_date="2026-09-30"` to validation so validation targets don't reach into October. For October targets with long horizons, the origins fall before 1 Oct: set `start_date` early enough, then keep the rows with `target_date` between 1 and 23 Oct. Keeping `fit_end_date="2026-09-30"` means no forecast is ever fitted on October data.

---

## 5. Findings so far

### 5.1 Benchmarks and univariate structure (h=1)
- The **ACF/PACF of daily Brent log returns is flat** (white noise to lag 20).
  - AR/MA/ARMA have nothing to use.
  - **ARIMA(p,1,q) on log prices is the same model as ARMA(p,q) on log returns**, and the data picks p=q=0. So ARIMA reduces to ARIMA(0,1,0), which is the random walk.
- **AR(1) converges to the random walk** as the lookback grows; the RMSE plateaus from about 3,000 days. Lookback 3000 is used.
- **Seasonality / calendar effects:**
  - **Month of year:** no effect (Kruskal-Wallis p = 0.19).
  - **Day of week:**
    - It is significant over the full sample (p ≈ 0.01) but not in 2015–2023 (p = 0.60).
    - The effect is tiny: about a 0.1% shift against about 2.7% daily noise.
    - At h=5 it cancels out, because every 5-trading-day window contains each weekday once.
  - So we use no calendar dummies.
- At h=5, the **non-overlapping** 5-day-return ACF/PACF is also flat. AR/MA are dropped as candidate models, and AR(1) is kept only as a benchmark.

### 5.2 Variable selection (Lecture 8 tools, on pre-2024 data)
- **Cross-correlation with the next day's Brent return** (significance band ±0.031):

  | Variable (lag 1) | Correlation | Significant? |
  |---|---|---|
  | WTI | 0.169 | yes |
  | S&P 500 | 0.088 | yes |
  | Brent's own | 0.063 | yes |
  | EPU | 0.039 | borderline |
  | Others | — | no |

  **Explanation:** US markets close after Brent is assessed in London, so Brent catches up the next day.
- **VAR order:** BIC picks 3 for the 8-variable system.
- **Granger causality into Brent** (VAR, F-test):

  | Variable | p-value | Result |
  |---|---|---|
  | WTI | 9.5e-69 | keep |
  | S&P | 2.0e-5 | keep |
  | USD | 0.0012 | keep |
  | VIX | 0.0043 | keep |
  | EPU | 0.010 | keep |
  | OVX | 0.049 | keep |
  | GPR | 0.84 | **dropped** |
- **Nested variable stages:**
  - **S1:** Brent + WTI + S&P
  - **S2:** S1 + USD + VIX
  - **S3:** S2 + EPU + OVX

  The lag order p is chosen by walk-forward MAPE, not by BIC.
- **Terminology:** in a VAR all variables are **endogenous** (each has its own equation). At h=1, the VAR's Brent equation is numerically identical to an ARX.

### 5.3 h=1 validation (walk-forward, 695 forecasts, targets 3 Jan 2024 – 30 Sep 2026)

| Model | MAE | RMSE | MAPE | Dir. acc. | Sharpe |
|---|---|---|---|---|---|
| Random walk | 1.6042 | 2.5742 | 1.8826 | n/a | n/a |
| Historical mean (lb 3000) | 1.6043 | 2.5745 | 1.8830 | 0.506 | 0.57 |
| AR(1) (lb 3000) | 1.6100 | 2.5875 | 1.8891 | 0.521 | 0.53 |
| **VAR S1 p=2** | **1.5901** | 2.5901 | **1.8667** | 0.553 | 0.66 |
| XGB S1 (n=50, depth=1, mcw=1) | 1.6025 | 2.5772 | 1.8795 | 0.515 | 0.35 |

- **VAR sweep:**
  - S1 p=1: MAPE 1.8740. It is the only VAR that also beats the random walk on RMSE (2.5608).
  - S1 p=2: MAPE 1.8667.
  - S2 p=2: MAPE 1.8655, a tie with S1, so S1 was chosen as the simpler set.
  - S3 is worse than the random walk at every p.
- **XGBoost:**
  - Untuned (200 trees, depth 2, learning rate 0.05) is worse than the random walk.
  - Grid: n ∈ {50, 200}, depth ∈ {1, 2}, min_child_weight ∈ {1, 100}.
  - Best was S2 with n=50, depth 1, mcw 1 (MAPE 1.8789). S1 with the same settings (1.8795) ties it, so S1 was chosen.
- **Size of the gains:** every edge is under 1%.
  - The Diebold-Mariano test (APE loss, optional) is not significant: VAR p = 0.31, XGB p = 0.70.
  - We report this honestly: at h=1 "tomorrow ≈ today".
- **Regime split** (validation days split at the median of GARCH(1,1) volatility), MAPE:

  | Model | High vol | Low vol |
  |---|---|---|
  | Random walk | 2.4878 | 1.2757 |
  | Historical mean | 2.4878 | 1.2764 |
  | AR(1) | 2.4969 | 1.2796 |
  | VAR S1 p=2 | 2.4860 | 1.2457 |
  | XGB S1 | 2.4857 | 1.2716 |

  Errors roughly double in high volatility. The VAR's edge comes only from calm periods.

### 5.4 h=1 October test (frozen at 30 Sep; data to 20 Oct, so 13 forecasts with targets 2–20 Oct)

| Model | MAE | RMSE | MAPE | MAPE vs RW | Dir. acc. |
|---|---|---|---|---|---|
| **Random walk** | 4.0000 | 6.9416 | **3.1263** | 0 | n/a |
| Historical mean | 4.0207 | 6.9430 | 3.1435 | −0.55% | 0.31 |
| AR(1) | 4.1694 | 7.1280 | 3.2593 | −4.25% | 0.31 |
| XGB S1 | 4.1749 | 7.0309 | 3.2752 | −4.76% | 0.46 |
| VAR S1 p=2 | 4.6447 | 7.0109 | 3.6688 | −17.35% | 0.38 |

- **October 2026 is entirely high-volatility:**
  - Average GARCH vol is 5.0% per day, against a validation median of 1.95%.
  - All 13 days fall in the high-vol regime.
- This is consistent with the regime split: the VAR's edge only existed in calm periods.
- **To do:** rerun once 21–23 Oct data is published.

### 5.5 h=5 (in progress)
- The models were made horizon-aware (commit `618ca12`). For h=1, the new code gives forecasts identical to the old code.
- **Plan:** keep the same lineup (RW, HM, AR(1), VAR, XGB), then re-pick at h=5:
  - the variable set (S1/S2/S3)
  - the VAR lag p
  - the XGB grid

  ACF/PACF, BIC and Granger don't need redoing, because they describe the daily process.

**VAR sweep on validation** (691 forecasts, targets 9 Jan 2024 – 30 Sep 2026):

| Model | MAE | RMSE | MAPE | MAPE vs RW | Dir. acc. |
|---|---|---|---|---|---|
| **VAR S1 p=3** | **3.6574** | **5.5645** | **4.3132** | **+0.64%** | 0.528 |
| VAR S2 p=3 | 3.6590 | 5.5720 | 4.3145 | +0.61% | 0.546 |
| VAR S1 p=5 | 3.6631 | 5.5779 | 4.3201 | +0.48% | 0.528 |
| VAR S2 p=2 | 3.6642 | 5.5842 | 4.3207 | +0.46% | 0.538 |
| VAR S1 p=2 | 3.6664 | 5.5755 | 4.3230 | +0.41% | 0.551 |
| VAR S1 p=4 | 3.6693 | 5.5720 | 4.3281 | +0.29% | 0.533 |
| VAR S1 p=1 | 3.6766 | 5.5874 | 4.3367 | +0.10% | 0.522 |
| Random walk | 3.6828 | 5.6054 | 4.3408 | 0 | n/a |
| Historical mean | 3.6878 | 5.6075 | 4.3487 | −0.18% | 0.507 |
| AR(1) | 3.6882 | 5.6123 | 4.3494 | −0.20% | 0.499 |
| VAR S3, p = 1 to 5 | 3.75–3.88 | 5.71–5.89 | 4.43–4.60 | −2.1% to −6.1% | 0.50–0.53 |

- **The pick is VAR S1 p=3.**
  - It has the best MAPE.
  - S2 p=3 is 0.0013 behind, within the tie rule, so we keep the simpler set.
- **S1 beats the random walk on MAE, RMSE and MAPE at every lag from 1 to 5.** At h=1, it beat the random walk on MAE and MAPE at every lag, but on RMSE only at p=1.
- **S3 is worse than the random walk at every p.** Each added variable adds parameters to every equation, so the model overfits.
- **The random walk's RMSE** is 5.6054 at h=5, against 2.5742 at h=1. That ratio is 2.18, close to √5, as theory predicts.
- **Trading value** (corrected Sharpe, h-day trades):

  | Model | Sharpe | Sharpe averaged over the 5 trade start days |
  |---|---|---|
  | Historical mean | 0.55 | — |
  | VAR S1 p=3 | 0.29 | 0.54 |
  | VAR S2 p=3 | 1.02 | 0.59 |

  So the VAR's small accuracy edge doesn't add trading value over "always long".
- **Next:**
  1. the XGB sweep at h=5
  2. the regime split
  3. the October test (frozen at 30 Sep)

### 5.6 Theory notes for longer horizons
- **Expected error growth:** under a random walk, error variance grows with h, so RMSE grows like √h. Expect h=5 RMSE to be about 2.2× the h=1 value.
- **Overlap:** consecutive h-day returns overlap, which creates fake autocorrelation (an MA(h−1) pattern).
  - Don't read ACF/PACF of overlapping returns.
  - The effective sample size is about n/h, and significance tests need HAC corrections (`diebold_mariano` takes `horizon`).
- **Fast vs slow signals:**
  - Fast signals (e.g. WTI's 1-day catch-up effect) count once while the noise grows, so they weaken relative to the noise.
  - Slow signals (levels of OVX, EPU, mean reversion) add up across days, so they matter relatively more at long horizons.
  - So re-pick the variables at each horizon.
- **Iterated vs direct:**
  - Iterated forecasts (VAR/AR stepped forward) are efficient if the model is right.
  - Direct forecasts (one model per horizon) are more robust if it's wrong.
  - At short horizons they are usually similar.
- **Leakage warning:** at 30 and 60 days, matching the random walk is an acceptable, honest result. Long-horizon results that are suspiciously good usually mean leakage.

---

## 6. Known issues and caveats

1. **`garch_volatility` is fitted on the full sample**, including the test period. That's fine for labelling regimes after the fact, but it must not be used as a forecasting input.
2. **`historical_mean_forecast` defaults to `lookback=500`.** The project uses 3000, so always pass `lookback=3000`.
3. **The random walk's directional metrics are 0, and its Sharpe is NaN, by construction.** It never predicts a move, so it never trades.
4. **Test-period economic metrics are very noisy.**
   - At h=1, annualised return and Sharpe come from only 13–17 test days.
   - At h=5, the 12 October forecasts give only 3 non-overlapping trades.
   - So report economic value from validation instead.
5. **The default transaction cost** is `transaction_cost=0.0001` = **0.01% (1 basis point) per trade**.

## 7. Open questions
- Does 1 Oct count as an h=1 target?
  - If yes: `start_date="2026-09-30"`, giving 17 forecasts.
  - If no: `start_date="2026-10-01"`, giving 16. This is the current setting.
- Which Brent price is scored: spot (current) or futures? They diverge a lot in October 2026.
- For h ≥ 10: confirm the selection/reporting split. Proposal: pick models on an earlier period (e.g. 2015–2023) and report on 2024 onwards, so the reported period isn't also the selection period.

## 8. Environment
- Python 3.11, with a venv (`source venv/bin/activate`) and `pip install -r requirements.txt`.
  - Key packages: pandas, numpy, statsmodels, arch, xgboost 3.2, scikit-learn, pandas-datareader, yfinance, xlrd, joblib.
- **XGBoost on macOS** needs OpenMP (`brew install libomp`). Use `n_jobs=1` inside `XGBRegressor`: it is about 9× faster for these small fits. Parallelise across grid settings with `joblib.Parallel` instead.
- **Repo:** GitHub `airrehn/Sample-Taxed`. `data/` and `*.csv` are gitignored.

---

## Appendix: full source code (`src/`)

### `src/data.py`

```python
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
```

### `src/features.py`

```python
import numpy as np
from arch import arch_model
import pandas as pd

def log_returns(df, col="brent_spot"):
    return np.log(df[col]).diff() # ln(P_t / P_t-1) i.e. log return = ln(P_t) - ln(P_t-1)

def realised_volatility(df, col="log_return", window = 20): # simple proxy for how volatile it's been recently, later will built GARCH
    return df[col].rolling(window).std()

def garch_volatility(df, col="log_return"):
    returns_pct = df[col] * 100 # scaled as percentage (2.0 instead of 0.02)
    model = arch_model(returns_pct, vol="GARCH", p=1, q=1, mean = "Zero", dist="normal")
    fitted = model.fit(disp="off")
    return fitted.conditional_volatility/100 # undoes the pct scaling

def make_lag_features(df, cols, lags):
    features = pd.DataFrame(index=df.index)
    for col in cols:
        for k in range(lags):
            features[f"{col}_lag{k}"] = df[col].shift(k)
    return features


```

### `src/models.py`

```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

from xgboost import XGBRegressor

def random_walk_forecast(df, origin_idx, horizon, fit_idx):
    return df["brent_spot"].iloc[origin_idx] # prediction is just today's price

def historical_mean_forecast(df, origin_idx, horizon, fit_idx, lookback=500):
    window = df["log_return"].iloc[fit_idx - lookback + 1: fit_idx  + 1] # Window up to 30 Sep 2026 max
    predicted_return = horizon * window.mean()   # average daily return × number of days
    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert return to latest predicted price

def ar1_forecast(df, origin_idx, horizon, fit_idx, lookback = 3000):
    window = df["log_return"].iloc[fit_idx - lookback + 1: fit_idx  + 1]
    fitted = AutoReg(window.values, lags=1).fit()
    c, phi = fitted.params[0], fitted.params[1]

    r = df["log_return"].iloc[origin_idx]   # today's return, the starting point
    predicted_return = 0
    for _ in range(horizon):
        r = c + phi * r # predict the next day from the previous one
        # e.g.
        # day 1: 0.06 (phi) × 0.02 (r) = 0.0012 (new r)
        # day 2: 0.06 × 0.0012 (new r) = 0.000072
        # day 3: 0.06 × 0.000072 ≈ 0.000004
        predicted_return += r
        # 5-day log return = 0.0012 + 0.000072 + 0.000004 + 0 + 0 ≈ 0.0013

    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert back return to latest predicted price

def var_forecast(df, origin_idx, horizon, fit_idx, cols, lags=3, lookback=3000):
    window = df[cols].iloc[fit_idx - lookback + 1: fit_idx + 1]
    fitted = VAR(window.values).fit(lags)
    recent = df[cols].iloc[origin_idx - lags + 1: origin_idx + 1].values  # pulls latest `lags` days of all variables (in col), used to forecast
    predicted_return = fitted.forecast(recent, steps=horizon)[:, 0].sum() #brents (log return) tomorrow in the table produced, sum the log returns
    price_now = df["brent_spot"].iloc[origin_idx] # get actual latest price
    return price_now * np.exp(predicted_return) # convert return to latest predicted price

def xgb_forecast(df, origin_idx, horizon, fit_idx, features, lookback=3000, n_estimators=200, max_depth=2, learning_rate=0.05, min_child_weight=1):
    X_all = features
    y_all = df["log_return"].rolling(horizon).sum().shift(-horizon) # sums log return over h days before it, then shifts df back by h days, so it aligns nicely as h-th days log return. 
    train_X = X_all.iloc[fit_idx - lookback: fit_idx - horizon + 1] # last row's h-day target ends exactly at fit_idx
    train_y = y_all.iloc[fit_idx - lookback: fit_idx - horizon + 1]
    keep = train_X.notna().all(axis=1) & train_y.notna() # same job as dropna(). makes Boolean flags for each row, we only train on the True rows.
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, random_state=0, n_jobs=1)
    model.fit(train_X[keep], train_y[keep])
    today = X_all.iloc[[origin_idx]]
    predicted_return = model.predict(today)[0] # it returns a list of 1 item, just specify the [0] anyways
    price_now = df["brent_spot"].iloc[origin_idx]
    return price_now * np.exp(predicted_return)
```

### `src/backtest.py`

```python
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
            break # Last target shouldn't be beyond target_end_date = 30 Sep 2026 (for val), 23 Oct 2026 (for test)
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

```

### `src/metrics.py`

```python
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats


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

def economic_value_metrics(results, transaction_cost = 0.0001, horizon = 1):
    results = results.iloc[::horizon]
    forecasted_position= np.sign(results["forecast"] - results["origin_price"])
    gross_return = forecasted_position * (results["actual"] - results["origin_price"]) / results ["origin_price"]

    trades = forecasted_position.diff().fillna(forecasted_position.iloc[0]).abs() # 0 trade: hold, 1 trade: long or short, 2 trades: flip pos
    net_return = gross_return - trades * transaction_cost # default: 0.01% per trade
    portfolio = (1 + net_return).cumprod()
    drawdown = (portfolio/portfolio.cummax()) - 1

    annualized_return = net_return.mean() * (252 / horizon)
    annualized_vol = net_return.std() * np.sqrt(252 / horizon)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan

    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.min(),
        "num_trades": trades.sum(),
    }

def summarise(results, horizon=1):
    summary = {}
    summary.update(accuracy_metrics(results))
    summary.update(directional_metrics(results))
    summary.update(economic_value_metrics(results, horizon=horizon))
    return summary

def compare_models(results_dict, horizon=1):
    rows = []
    for name, results in results_dict.items():
        row = summarise(results, horizon=horizon)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def vol_regime(results, volatility_all):
    vol_sorted = volatility_all.reindex(results["origin_date"]).values #grabs only volatility of dates of interest from the results df.
    median_vol = np.nanmedian(vol_sorted)
    is_high_vol = vol_sorted >= median_vol

    return { # 2 different accuracy metrics split by high and low vol periods
        "high_vol": accuracy_metrics(results[is_high_vol]),
        "low_vol": accuracy_metrics(results[~is_high_vol]),
    }


def compare_vol_regimes(results_dict, volatility): # one row per model, columns = each metric split by vol regime
    rows = {}
    for name, results in results_dict.items():
        regime = vol_regime(results, volatility) # nested dict, e.g. {"high_vol": {"MAE": 2.1, ...}, "low_vol": {"MAE": 0.9, ...}}
        row = {}
        for reg, metrics in regime.items(): # each regime (high_vol / low_vol)
            for metric, value in metrics.items(): # each metric inside that regime (MAE, RMSE, MAPE)
                row[f"{metric}_{reg}"] = value # flatten into one key, e.g. "MAE_high_vol"
        rows[name] = row
    return pd.DataFrame(rows).T # dict keys become columns, so transpose to put models on rows


def diebold_mariano(results_a, results_b, horizon=1, loss="ape"): # Not necessary but useful when models are super close (e.g. horizon 1 model's are ~ 1% from each other), to test if the difference is just noise or statistically significant
    a = results_a.set_index("target_date") # line both models up by the day being predicted
    b = results_b.set_index("target_date")
    if loss == "ape": # absolute % error, matches MAPE
        loss_a = (a["actual"] - a["forecast"]).abs() / a["actual"]
        loss_b = (b["actual"] - b["forecast"]).abs() / b["actual"]
    else: # squared error, matches RMSE
        loss_a = (a["actual"] - a["forecast"]) ** 2
        loss_b = (b["actual"] - b["forecast"]) ** 2
    d = (loss_a - loss_b).dropna() # daily gap, negative = A was better that day
    n = len(d)

    d_centered = d - d.mean()
    variance = (d_centered ** 2).mean()
    for k in range(1, horizon): # h>1 forecasts overlap, so neighbouring gaps are correlated: add that in
        variance += 2 * (d_centered.iloc[k:].values * d_centered.iloc[:-k].values).mean()

    dm_stat = d.mean() / np.sqrt(variance / n) # like a t-stat: average gap / its standard error
    dm_stat *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n) # small-sample correction (Harvey et al.)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n - 1)
    return {"DM_stat": dm_stat, "p_value": p_value, "n": n}

def dm_table(results_dict, benchmark="Random Walk", horizon=1, loss="ape"):
    rows = {}
    for name, results in results_dict.items():
        if name != benchmark:
            rows[name] = diebold_mariano(results, results_dict[benchmark], horizon=horizon, loss=loss)
    return pd.DataFrame(rows).T

 



```
