# Financial Time Series

This module studies the time evolution of prices, returns, and volatility.

## Expected outcome

By the end of this module, students should be able to:

- distinguish prices, simple returns, and log returns;
- evaluate stationarity and autocorrelation;
- interpret ACF and PACF;
- recognize volatility clustering;
- fit ARCH and GARCH models;
- use an interactive volatility dashboard to explain persistence;
- validate residuals and explain model limitations.

## Included notes

- introduction to time series;
- decomposition and exploratory analysis;
- ARCH models;
- GARCH models;
- diagnostics and volatility extensions;
- ARIMA diagnostic workflow;
- GARCH volatility and risk workflow;
- Interactive Volatility and GARCH Dashboard.

## Recommended next improvements

- clean repeated dependencies across notebooks;
- include a forecast evaluation section;
- add asymmetric volatility examples with GJR-GARCH or EGARCH;
- connect conditional volatility forecasts with dynamic VaR.

## Class sequence

1. Convert market prices into clean time-indexed series.
2. Inspect price levels and log returns.
3. Study decomposition, stationarity intuition, autocorrelation, and partial autocorrelation.
4. Introduce white noise, random walks, AR, MA, ARMA, and ARIMA.
5. Extend the discussion to volatility clustering with ARCH and GARCH.
6. Validate residuals with Ljung-Box and Jarque-Bera diagnostics.
7. Discuss asymmetric volatility and heavy-tailed risk forecasting.
8. Run the ARIMA diagnostic lab and the GARCH risk lab as model-building exercises.
9. Use the volatility dashboard to compare low- and high-persistence regimes.

## In-class practice

Students should choose one asset, plot prices and returns, interpret ACF/PACF plots, compare a simple time series model with an ARCH/GARCH volatility model, and use the dashboard to explain persistence. They should also report whether residual diagnostics support the selected model.

## Module checkpoint

The checkpoint is a diagnostic report that explains whether the selected return series shows autocorrelation, volatility clustering, asymmetric volatility, heavy tails, or a combination of these features.
