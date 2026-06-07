# Technical Glossary

This glossary defines core terms used throughout the Jupyter Book.

| Term | Definition | Main module |
| --- | --- | --- |
| Adjusted close | Price series adjusted for dividends, splits, and other corporate actions. | Markets and data |
| Alpha | Regression intercept or abnormal return not explained by the benchmark model. | Portfolio theory |
| American option | Option that can be exercised before maturity, requiring an early-exercise decision rule. | Derivatives |
| ARIMA | Autoregressive integrated moving average model for time-series dynamics. | Time series |
| Asian option | Path-dependent option whose payoff depends on an average underlying price. | Derivatives |
| Autocorrelation | Correlation between a series and its own lagged values. | Time series |
| Barrier option | Path-dependent option activated or extinguished when the underlying crosses a barrier. | Derivatives |
| Beta | Sensitivity of an asset's excess return to the market excess return. | Portfolio theory |
| Black-Scholes model | Option-pricing model for European options under lognormal price dynamics and constant volatility. | Derivatives |
| Bond convexity | Curvature of the bond price-yield relationship; improves duration-based price approximations. | Fixed income |
| Bono M | Mexican fixed-rate nominal government bond with semiannual coupons. | Fixed income |
| Bootstrapping | Procedure for deriving discount factors or spot rates from market instrument prices. | Term structure |
| Broadie-Glasserman-Kou correction | Barrier adjustment used to approximate discrete monitoring with a continuous-barrier formula. | Derivatives |
| Capital Market Line | Line connecting the risk-free asset and the tangency portfolio in mean-volatility space. | Portfolio theory |
| CETES | Mexican Treasury certificates; zero-coupon government securities quoted on money-market conventions. | Fixed income |
| Clean price | Bond quoted price excluding accrued interest. | Fixed income |
| Conditional volatility | Time-varying volatility forecast conditional on past information. | Time series |
| Convexity | Second-order sensitivity of a price to a change in yield or another risk factor. | Fixed income |
| Correlation | Standardized measure of linear co-movement between two variables. | Markets and data |
| Cost of carry | Net financing, income, storage, or convenience yield effect linking spot and forward prices. | Derivatives |
| CVaR | Conditional Value at Risk; average loss conditional on exceeding the VaR threshold. Also called Expected Shortfall. | Market risk |
| Data mode | Explicit notebook setting, such as `DATA_MODE=offline` or `DATA_MODE=live`, that determines whether a dashboard uses a versioned publication snapshot or provider-backed live data. | Markets and data |
| Data source inventory | Structured record of provider, instrument or variable, frequency, date range, field, currency, calendar, and limitations before modeling. | Markets and data |
| Dirty price | Bond settlement price including accrued interest. | Fixed income |
| Duration | First-order sensitivity of a bond price to changes in yield. | Fixed income |
| DV01 | Dollar value of one basis point; approximate currency price change for a one-basis-point yield move. | Fixed income |
| Efficient frontier | Set of portfolios with minimum variance for each expected-return target. | Portfolio theory |
| Expected Shortfall | Average tail loss beyond a selected quantile threshold. | Market risk |
| DB.NOMICS | Public macroeconomic data platform and API that aggregates series from official providers. | Markets and data |
| Forward contract | OTC agreement to buy or sell an asset at a fixed future delivery price. | Derivatives |
| Futures contract | Exchange-traded forward-like contract with margining and daily mark-to-market settlement. | Derivatives |
| GARCH | Generalized autoregressive conditional heteroskedasticity model for volatility clustering. | Time series |
| Gamma | Option Greek measuring the sensitivity of Delta to the underlying price. | Derivatives |
| Global minimum variance portfolio | Portfolio with the lowest variance among fully invested portfolios. | Portfolio theory |
| Greeks | Local option sensitivities such as Delta, Gamma, Vega, Theta, and Rho. | Derivatives |
| Historical VaR | VaR estimated directly from empirical historical returns. | Market risk |
| Heston model | Stochastic-volatility model with mean-reverting variance and correlated spot and variance shocks. | Derivatives |
| Implied volatility | Volatility value that makes an option-pricing model match the observed market option price. | Derivatives |
| Implied volatility smile | Pattern of implied volatility across strikes for a fixed maturity. | Derivatives |
| Kurtosis | Distributional measure related to tail heaviness relative to a normal distribution. | Markets and data |
| Log return | Continuously compounded return computed as the log difference of prices. | Markets and data |
| Macaulay duration | Weighted average time to receive a bond's cash flows. | Fixed income |
| Leisen-Reimer tree | Binomial tree designed to improve convergence for option pricing, especially around payoff kinks. | Derivatives |
| Local volatility | Deterministic volatility function of strike and maturity calibrated to vanilla option prices. | Derivatives |
| Live data mode | Dashboard mode that fetches provider-backed data through shared `src` helpers, local cache, and approved credentials or public endpoints. | Markets and data |
| Modified duration | Approximate percentage price change for a one-unit change in yield. | Fixed income |
| Monte Carlo simulation | Numerical method that estimates values or risks by generating many random scenarios. | Derivatives |
| Nelson-Siegel model | Parametric yield-curve model with level, slope, curvature, and decay parameters. | Term structure |
| Nelson-Siegel-Svensson model | Extension of Nelson-Siegel with an additional curvature term and decay parameter. | Term structure |
| OAS covariance | Oracle Approximating Shrinkage covariance estimator. | Portfolio theory |
| Offline snapshot | Versioned real-data extract used so notebooks and book builds run without credentials, network access, or provider-rate-limit risk. | Markets and data |
| Parametric VaR | VaR estimated from an assumed distribution, often Gaussian. | Market risk |
| Par swap rate | Fixed rate that makes a swap have zero value at inception under a given discount curve. | Derivatives |
| Principal Component Analysis | Dimension-reduction method used to summarize correlated curve or return movements with orthogonal factors. | Term structure |
| Put-call parity | No-arbitrage relationship between European call prices, put prices, spot price, strike, and discounting. | Derivatives |
| Risk parity | Allocation method that targets equal or specified risk contributions across assets. | Portfolio theory |
| Rolling window | Moving historical sample used to estimate statistics through time. | Time series |
| Sharpe ratio | Excess return per unit of volatility. | Portfolio theory |
| Skewness | Measure of distributional asymmetry. | Markets and data |
| Sortino ratio | Excess return per unit of downside deviation. | Market risk |
| Stationarity | Property that a time series has stable statistical behavior over time. | Time series |
| Stochastic volatility | Modeling framework in which volatility is random rather than fixed. | Derivatives |
| Swap | Derivative contract exchanging two streams of cash flows, commonly fixed versus floating interest payments. | Derivatives |
| Tangency portfolio | Risky portfolio with the highest Sharpe ratio relative to the selected risk-free rate. | Portfolio theory |
| Theta | Option Greek measuring sensitivity to the passage of time. | Derivatives |
| UDIBONOS | Mexican inflation-linked government bonds denominated in UDIS and settled in MXN. | Fixed income |
| Value at Risk | Quantile-based loss threshold over a chosen horizon and confidence level. | Market risk |
| VaR exception | Realized loss exceeding the forecast VaR threshold. | Market risk |
| Vega | Option Greek measuring sensitivity to volatility. | Derivatives |
| Volatility | Standard deviation of returns, commonly annualized in finance. | Markets and data |
| Yield curve | Relationship between yields and maturities for a set of fixed-income instruments. | Term structure |
