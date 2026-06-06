# Financial Time Series

This module studies the time evolution of prices, returns, and volatility using the classical time-series and financial econometrics toolkit {cite}`box2015time,hamilton1994time,tsay2010analysis`.

## Expected outcome

By the end of this module, students should be able to:

- distinguish prices, simple returns, and log returns;
- evaluate stationarity and autocorrelation;
- interpret ACF and PACF;
- recognize volatility clustering;
- fit ARCH and GARCH models;
- compare symmetric, asymmetric, Gaussian, and heavy-tailed volatility specifications;
- use an interactive volatility dashboard to explain persistence;
- validate residuals with autocorrelation and normality diagnostics;
- explain model limitations, regime sensitivity, and forecast uncertainty.

## Included notes

- introduction to time series;
- decomposition and exploratory analysis;
- ARCH models;
- GARCH models;
- diagnostics and volatility extensions;
- ARIMA diagnostic workflow;
- GARCH volatility and risk workflow;
- Interactive Volatility and GARCH Dashboard.

## Reading sequence

1. Convert market prices into clean time-indexed series.
2. Inspect price levels and log returns.
3. Study decomposition, stationarity intuition, autocorrelation, and partial autocorrelation.
4. Introduce white noise, random walks, AR, MA, ARMA, and ARIMA.
5. Extend the discussion to volatility clustering with ARCH and GARCH {cite}`engle1982autoregressive,bollerslev1986generalized`.
6. Validate residuals with Ljung-Box and Jarque-Bera diagnostics {cite}`ljung1978measure,jarque1980efficient`.
7. Discuss asymmetric volatility and heavy-tailed risk forecasting {cite}`glosten1993relation,nelson1991conditional`.
8. Use the ARIMA diagnostic lab and the GARCH risk lab as model-building references.
9. Use the volatility dashboard to compare low- and high-persistence regimes.
