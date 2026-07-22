# Technical Glossary

This glossary defines core terms used throughout the Jupyter Book.

| Term | Definition | Main module |
| --- | --- | --- |
| Adjusted close | Price series adjusted for dividends, splits, and other corporate actions. | Markets, Instruments, and Data |
| AFFO | Adjusted funds from operations; a real-estate cash-flow measure that adjusts FFO for recurring capital expenditures and other normalized items under the stated convention. | Alternative Investments |
| Annualized log return | Mean periodic log return multiplied by the declared number of periods per year; it is not an annualized simple return unless transformed with `expm1`. | Markets, Instruments, and Data |
| Alpha | Regression intercept or abnormal return not explained by the benchmark model. | Portfolio Management, Asset Allocation, and Performance |
| American option | Option that can be exercised before maturity, requiring an early-exercise decision rule. | Derivatives and Risk Management |
| ARIMA | Autoregressive integrated moving average model for time-series dynamics. | Quantitative Methods and Financial Time Series |
| Asian option | Path-dependent option whose payoff depends on an average underlying price. | Derivatives and Risk Management |
| Autocorrelation | Correlation between a series and its own lagged values. | Quantitative Methods and Financial Time Series |
| Balance of payments | Summary of transactions between residents and the rest of the world, including current-account and financial-account flows. | Economics, Macro, and Currency |
| Barrier option | Path-dependent option activated or extinguished when the underlying crosses a barrier. | Derivatives and Risk Management |
| Benchmark | Explicit reference asset, index, or policy portfolio used to evaluate relative return, risk, or attribution. | Markets, Instruments, and Data |
| Beta | Sensitivity of an asset's excess return to the market excess return. | Portfolio Management, Asset Allocation, and Performance |
| Bid-ask spread | Difference between the best quoted ask and bid prices; a direct but incomplete measure of trading cost and liquidity. | Markets, Instruments, and Data |
| Black-Scholes model | Option-pricing model for European options under lognormal price dynamics and constant volatility. | Derivatives and Risk Management |
| Bond convexity | Curvature of the bond price-yield relationship; improves duration-based price approximations. | Fixed Income, Credit, and Term Structure |
| Bono M | Mexican fixed-rate nominal government bond with semiannual coupons. | Fixed Income, Credit, and Term Structure |
| Bootstrapping | Procedure for deriving discount factors or spot rates from market instrument prices. | Fixed Income, Credit, and Term Structure |
| Broadie-Glasserman-Kou correction | Barrier adjustment used to approximate discrete monitoring with a continuous-barrier formula. | Derivatives and Risk Management |
| Capital Market Line | Line connecting the risk-free asset and the tangency portfolio in mean-volatility space. | Portfolio Management, Asset Allocation, and Performance |
| Carry trade | Currency strategy that borrows in a lower-yielding currency and invests in a higher-yielding currency, earning carry while bearing depreciation and unwind risk. | Economics, Macro, and Currency |
| Carried interest | Performance allocation paid to a private-fund manager after the contractual waterfall, including any preferred return, catch-up, and clawback provisions. | Alternative Investments |
| CETES | Mexican Treasury certificates; zero-coupon government securities quoted on money-market conventions. | Fixed Income, Credit, and Term Structure |
| CKD | Mexican listed development-capital certificate used to finance private or real-asset projects through a trust structure. | Markets, Instruments, and Data |
| Clean price | Bond quoted price excluding accrued interest. | Fixed Income, Credit, and Term Structure |
| Conditional volatility | Time-varying volatility forecast conditional on past information. | Quantitative Methods and Financial Time Series |
| Convexity | Second-order sensitivity of a price to a change in yield or another risk factor. | Fixed Income, Credit, and Term Structure |
| Correlation | Standardized measure of linear co-movement between two variables. | Markets, Instruments, and Data |
| Cost of carry | Net financing, income, storage, or convenience yield effect linking spot and forward prices. | Derivatives and Risk Management |
| Covered interest parity | No-arbitrage relationship linking spot FX, forward FX, and domestic and foreign interest rates when currency risk is hedged. | Economics, Macro, and Currency |
| CVaR | Conditional Value at Risk; average loss conditional on exceeding the VaR threshold. Also called Expected Shortfall. | Derivatives and Risk Management |
| Data mode | Explicit notebook setting, such as `DATA_MODE=offline` or `DATA_MODE=live`, that determines whether a dashboard uses a versioned publication snapshot or provider-backed live data. | Markets, Instruments, and Data |
| Data source inventory | Structured record of provider, instrument or variable, frequency, date range, field, currency, calendar, and limitations before modeling. | Markets, Instruments, and Data |
| DIO | Days inventory outstanding; average inventory divided by cost of goods sold and multiplied by the declared day count. | Financial Statement Analysis and Financial Modeling |
| Dirty price | Bond settlement price including accrued interest. | Fixed Income, Credit, and Term Structure |
| DPO | Days payables outstanding; average operating trade payables divided by credit purchases and multiplied by the declared day count. When purchases are unavailable, a cost-of-goods-sold proxy must be labeled as such. | Financial Statement Analysis and Financial Modeling |
| DSO | Days sales outstanding; average trade receivables divided by net credit sales and multiplied by the declared day count. When credit sales are unavailable, a revenue proxy must be labeled as such. | Financial Statement Analysis and Financial Modeling |
| Drawdown | Percentage decline from the running wealth or price peak to the current level. | Markets, Instruments, and Data |
| DuPont analysis | Decomposition of return on equity into profitability, asset efficiency, and financial leverage components. | Financial Statement Analysis and Financial Modeling |
| Duration | First-order sensitivity of a bond price to changes in yield. | Fixed Income, Credit, and Term Structure |
| DV01 | Dollar value of one basis point; approximate currency price change for a one-basis-point yield move. | Fixed Income, Credit, and Term Structure |
| EBITDA | Earnings before interest, taxes, depreciation, and amortization; an intermediate performance measure whose calculation and any further adjustments must be stated because it is not a standardized IFRS line item. | Financial Statement Analysis and Financial Modeling |
| Efficient frontier | Set of portfolios with minimum variance for each expected-return target. | Portfolio Management, Asset Allocation, and Performance |
| Exchange rate | Price of one currency expressed in terms of another currency. | Economics, Macro, and Currency |
| Expected Shortfall | Average tail loss beyond a selected quantile threshold. | Derivatives and Risk Management |
| FCFE | Free cash flow to equity; cash flow available to common equity after operating needs, investment, and net debt financing under the stated forecast. | Corporate Issuers and Equity Valuation |
| FCFF | Free cash flow to the firm; after-tax operating cash flow available to debt and equity capital providers after required reinvestment. | Corporate Issuers and Equity Valuation |
| FIBRA | Mexican real-estate investment trust certificate, broadly analogous to a REIT, with exchange-traded ownership of income-producing real assets. | Markets, Instruments, and Data |
| FFO | Funds from operations; a real-estate performance measure that begins with net income and applies the declared Nareit-style adjustments. | Alternative Investments |
| Fiscal policy | Government tax, spending, deficit, and debt policy that affects demand, rates, sovereign risk, and growth expectations. | Economics, Macro, and Currency |
| Forward premium | Situation in which a forward exchange rate implies that the base currency trades at a premium relative to the spot rate under the chosen quote convention. | Economics, Macro, and Currency |
| DB.NOMICS | Public macroeconomic data platform and API that aggregates series from official providers. | Markets, Instruments, and Data |
| Forward contract | OTC agreement to buy or sell an asset at a fixed future delivery price. | Derivatives and Risk Management |
| Futures contract | Exchange-traded forward-like contract with margining and daily mark-to-market settlement. | Derivatives and Risk Management |
| GARCH | Generalized autoregressive conditional heteroskedasticity model for volatility clustering. | Quantitative Methods and Financial Time Series |
| Gamma | Option Greek measuring the sensitivity of Delta to the underlying price. | Derivatives and Risk Management |
| Global minimum variance portfolio | Portfolio with the lowest variance among fully invested portfolios. | Portfolio Management, Asset Allocation, and Performance |
| Greeks | Local option sensitivities such as Delta, Gamma, Vega, Theta, and Rho. | Derivatives and Risk Management |
| Historical VaR | VaR estimated directly from empirical historical returns. | Derivatives and Risk Management |
| Heston model | Stochastic-volatility model with mean-reverting variance and correlated spot and variance shocks. | Derivatives and Risk Management |
| Hit ratio | Share of forecasts, signals, or directional calls that meet the page's explicitly defined success condition. | Markets, Instruments, and Data |
| Implied volatility | Volatility value that makes an option-pricing model match the observed market option price. | Derivatives and Risk Management |
| Implied volatility smile | Pattern of implied volatility across strikes for a fixed maturity. | Derivatives and Risk Management |
| Inflation | Sustained increase in the general price level, reducing purchasing power and affecting nominal rates, real returns, and policy expectations. | Economics, Macro, and Currency |
| Investment Policy Statement | Governed document that records an investor's objectives, risk tolerance, horizon, liquidity needs, constraints, benchmark, responsibilities, and review rules. | Portfolio Management, Asset Allocation, and Performance |
| Interest rate parity | Relationship linking interest-rate differentials with expected or forward exchange-rate changes. | Economics, Macro, and Currency |
| Kurtosis | Distributional measure related to tail heaviness relative to a normal distribution. | Markets, Instruments, and Data |
| Log return | Continuously compounded return computed as the log difference of prices. | Markets, Instruments, and Data |
| Macaulay duration | Weighted average time to receive a bond's cash flows. | Fixed Income, Credit, and Term Structure |
| Leisen-Reimer tree | Binomial tree designed to improve convergence for option pricing, especially around payoff kinks. | Derivatives and Risk Management |
| Liquidity | Ability to trade a desired quantity promptly with limited price impact and transaction cost. | Markets, Instruments, and Data |
| Local volatility | Deterministic volatility function of strike and maturity calibrated to vanilla option prices. | Derivatives and Risk Management |
| Live data mode | Dashboard mode that fetches provider-backed data through shared `src` helpers, local cache, and approved credentials or public endpoints. | Markets, Instruments, and Data |
| Modified duration | Approximate percentage price change for a one-unit change in yield. | Fixed Income, Credit, and Term Structure |
| Monetary policy | Central bank policy affecting short rates, liquidity, inflation expectations, exchange rates, and financial conditions. | Economics, Macro, and Currency |
| Monte Carlo simulation | Numerical method that estimates values or risks by generating many random scenarios. | Derivatives and Risk Management |
| NCI | Non-controlling interest; the equity in a subsidiary not attributable, directly or indirectly, to the parent. Consolidated analysis must distinguish the group's totals from the amounts attributable to owners of the parent. | Financial Statement Analysis and Financial Modeling |
| Nelson-Siegel model | Parametric yield-curve model with level, slope, curvature, and decay parameters. | Fixed Income, Credit, and Term Structure |
| Nelson-Siegel-Svensson model | Extension of Nelson-Siegel with an additional curvature term and decay parameter. | Fixed Income, Credit, and Term Structure |
| Net present value | Present value of expected project cash inflows less the present value of cash outflows under an explicit discount-rate and scenario contract. | Corporate Issuers and Equity Valuation |
| NOPAT | Net operating profit after tax; after-tax operating profit derived from consistently defined operating earnings and taxes attributable to those earnings. It is an analytical measure rather than a standardized financial-statement line item. | Financial Statement Analysis and Financial Modeling |
| OAS covariance | Oracle Approximating Shrinkage covariance estimator. | Portfolio Management, Asset Allocation, and Performance |
| Offline snapshot | Versioned real-data extract used so notebooks and book builds run without credentials, network access, or provider-rate-limit risk. | Markets, Instruments, and Data |
| Output gap | Difference between actual output and estimated potential output, often used to interpret inflation pressure and slack. | Economics, Macro, and Currency |
| OWC | Operating working capital; operating current assets less non-interest-bearing operating current liabilities under a declared scope. Cash, debt, and other financing balances are normally excluded unless the analysis states otherwise. | Financial Statement Analysis and Financial Modeling |
| Owner earnings | Analyst-defined estimate of distributable economics based on normalized earnings, eligible non-cash charges, required capital expenditure, and additional operating working capital. It is not a standardized accounting measure, so each component and maintenance-investment assumption must be disclosed. | Financial Statement Analysis and Financial Modeling |
| Parametric VaR | VaR estimated from an assumed distribution, often Gaussian. | Derivatives and Risk Management |
| Par swap rate | Fixed rate that makes a swap have zero value at inception under a given discount curve. | Derivatives and Risk Management |
| Principal Component Analysis | Dimension-reduction method used to summarize correlated curve or return movements with orthogonal factors. | Fixed Income, Credit, and Term Structure |
| Put-call parity | No-arbitrage relationship between European call prices, put prices, spot price, strike, and discounting. | Derivatives and Risk Management |
| Purchasing power parity | Long-horizon benchmark linking exchange-rate changes with relative inflation between two currencies. | Economics, Macro, and Currency |
| Rate panel | Collection of rates from potentially different instruments or markets; unlike a yield curve, its columns need not be homogeneous tenors of one curve. | Fixed Income, Credit, and Term Structure |
| Residual income | Earnings in excess of the equity charge, commonly defined as net income minus beginning book equity times the required return on equity. | Corporate Issuers and Equity Valuation |
| ROIC | Return on invested capital; normalized after-tax operating profit divided by the consistently defined capital invested in operations. | Financial Statement Analysis and Financial Modeling |
| Risk parity | Allocation method that targets equal or specified risk contributions across assets. | Portfolio Management, Asset Allocation, and Performance |
| Rolling window | Moving historical sample used to estimate statistics through time. | Quantitative Methods and Financial Time Series |
| Sharpe ratio | Excess return per unit of volatility. | Portfolio Management, Asset Allocation, and Performance |
| Simple return | Periodic percentage change in value, computed as \(P_t/P_{t-1}-1\). | Markets, Instruments, and Data |
| Skewness | Measure of distributional asymmetry. | Markets, Instruments, and Data |
| Sortino ratio | Excess return per unit of downside deviation. | Derivatives and Risk Management |
| Stationarity | Property that a time series has stable statistical behavior over time. | Quantitative Methods and Financial Time Series |
| Stochastic volatility | Modeling framework in which volatility is random rather than fixed. | Derivatives and Risk Management |
| Swap | Derivative contract exchanging two streams of cash flows, commonly fixed versus floating interest payments. | Derivatives and Risk Management |
| Tangency portfolio | Risky portfolio with the highest Sharpe ratio relative to the selected risk-free rate. | Portfolio Management, Asset Allocation, and Performance |
| Three-statement model | Forecast model that links the income statement, balance sheet, and statement of cash flows through explicit accounting identities and schedules. | Financial Statement Analysis and Financial Modeling |
| Theta | Option Greek measuring sensitivity to the passage of time. | Derivatives and Risk Management |
| TIIE | Mexican interbank equilibrium interest-rate benchmark; its tenor and methodological vintage must be stated when used. | Markets, Instruments, and Data |
| UDI | Mexican inflation-indexed unit of account whose peso value is published by Banco de México. | Markets, Instruments, and Data |
| UDIBONOS | Mexican inflation-linked government bonds denominated in UDIS and settled in MXN. | Fixed Income, Credit, and Term Structure |
| Uncovered interest parity | Benchmark relationship linking expected exchange-rate change with an interest-rate differential when currency risk is not hedged. | Economics, Macro, and Currency |
| Value at Risk | Quantile-based loss threshold over a chosen horizon and confidence level. | Derivatives and Risk Management |
| VaR exception | Realized loss exceeding the forecast VaR threshold. | Derivatives and Risk Management |
| Vega | Option Greek measuring sensitivity to volatility. | Derivatives and Risk Management |
| Volatility | Standard deviation of returns, commonly annualized in finance. | Markets, Instruments, and Data |
| WACC | Weighted average cost of capital; market-value-weighted required return on debt and equity after the stated tax and capital-structure assumptions. | Corporate Issuers and Equity Valuation |
| Yield curve | Relationship between yields and maturities for a set of fixed-income instruments. | Fixed Income, Credit, and Term Structure |
