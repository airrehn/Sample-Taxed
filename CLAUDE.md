# Oil price forecasting project (QF5210 Part II, alternative group project)

Owner: Aaron (NUS MQF). I write the code myself and ask Claude for advice, review and debugging. Do not rewrite my files unless I ask.

## How to talk to me
- Keep explanations short. Start with a simple "ELI5" version, then more detail only if I ask.
- Include a small example when explaining a concept.
- Explain every command or line of code you give me, part by part.

## Task (from the course brief)
- Build a data-driven forecasting system for crude oil prices/returns using daily data.
- Include at least one major benchmark. Decision: **Brent** (WTI went negative in Apr 2020, which breaks log returns).
- Models should be able to update as new daily data arrives. Any language is allowed.
- Possible approaches in the brief: ARIMA/VAR/GARCH, random forest/XGBoost, LSTM/Transformer, regime switching, multi-factor models, news/sentiment, hybrids/ensembles. Must be explainable.
- Submission (Sat 31 Oct 2026): zip with editable slides, data (CSV), code, optional video. Presentation is 8 min + 2 min Q&A, every member presents.
- Must document: data sources, variable definitions, sample period, train/validation/test split, preprocessing.

## Design decisions (agreed)
1. **Horizons:** h = 1, 2, 5, 10, 15, 20, 30, 60 days ahead. Proposed unit is TRADING days (OPEN: confirm vs calendar days).
2. **Direct strategy, "design A":** a separate model per horizon. The model for horizon h predicts the value at t+h using only information available at time t. No recursion (no feeding 1-day forecasts forward), so errors do not compound. The lookback window is a tunable setting and may differ per horizon.
3. **Test window:** target dates 1-30 Oct 2026 (22 trading days). Example: for target 2 Oct, the 60-day model's information ends 60 trading days earlier; the 1-day model's ends 1 Oct.
4. **No leakage:** nothing from 1 Oct 2026 onward is used to fit or tune. October actuals may only be used as inputs, up to each forecast's origin. Fit/tune on data before 1 Oct 2026 (train + validation split, test untouched).
5. **Target:** cumulative log return over h days, converted back to a price level.
6. **Fair comparison:** compare models only against others at the SAME horizon. Report error vs horizon (expected to grow) and relative skill vs benchmark per horizon.

## Models plan (build in stages)
1. Benchmarks: random walk / no-change, historical mean, AR/ARIMA.
2. Classical: ARX / ARMA with exogenous inputs. GARCH-type models give volatility and forecast ranges (GARCH will not help the point forecast much at long horizons).
3. ML: XGBoost (explain with SHAP).
4. Sentiment from news (needs timestamped news available at the forecast origin).
5. Optional: Brent futures price as an extra benchmark (market's own forecast).
6. Extra inputs added in stages to see what each contributes: WTI, USD index, S&P 500, VIX, etc.

## Evaluation (from the brief)
- Accuracy: MAE, RMSE, MAPE where appropriate.
- Directional accuracy, plus precision/recall/F1 for up/down moves.
- Relative to benchmarks (random walk, historical mean, AR/ARIMA): % improvement in MAE/RMSE.
- Robustness: high- vs low-volatility regimes, stability across windows.
- Economic value: simple trading rule with P&L, annualised return, Sharpe, max drawdown, transaction costs.
- Only 22 test points per horizon, so metrics are noisy: also run a longer rolling backtest (e.g. 2024 to Sep 2026) and a Diebold-Mariano test.
- Long horizons (30, 60 days) are expected to be hard. Matching the random walk is an acceptable, honest result. Suspiciously good long-horizon results usually mean leakage.

## Data
- Brent: Yahoo `BZ=F` (continuous futures, has jumps at contract roll) and/or EIA daily Brent spot (FRED series `DCOILBRENTEU`). Check what is reachable.
- Today is 18 Oct 2026: actuals for 1-16 Oct exist; 19-30 Oct will arrive during the week, so the pipeline must be easy to re-run.

## Open questions
- Trading days vs calendar days for horizons.
- Which data source(s) for Brent.
- Exact train/validation split dates.
