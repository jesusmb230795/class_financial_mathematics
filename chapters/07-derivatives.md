# Derivatives

This module introduces derivative payoffs, no-arbitrage pricing, and computational pricing methods.

## Expected outcome

By the end of this module, students should be able to:

- describe forward, futures, call option, and put option payoffs;
- use no-arbitrage replication logic in a one-period model;
- price European options with Black-Scholes assumptions;
- interpret Delta, Gamma, Vega, Theta, and Rho;
- price forwards and par swaps from carry and discount factors;
- estimate implied volatility and interpret volatility smiles;
- implement binomial trees, Monte Carlo simulations, and selected exotic-option methods.

## Included notes

- options and Black-Scholes;
- binomial trees and Monte Carlo pricing;
- Interactive Black-Scholes Greeks Dashboard;
- linear derivatives and carry;
- implied volatility and smiles;
- American and exotic option methods;
- stochastic volatility and Heston lab.

## Recommended next improvements

- add strategy payoff diagrams for spreads, straddles, collars, and covered calls;
- connect optional live option-chain data through `src/market_data.py`;
- add an implied-volatility surface fitting lab with arbitrage checks;
- extend Heston from simulation to stable characteristic-function pricing;
- add option portfolio hedging and P&L attribution exercises.

## Class sequence

1. Define derivative contracts and payoff diagrams.
2. Explain no-arbitrage pricing with replication.
3. Price forwards, futures-style exposures, and par swaps from carry and discount factors.
4. Derive binomial pricing intuition from risk-neutral probabilities.
5. Apply Black-Scholes to European calls and puts.
6. Interpret Greeks as local sensitivity measures.
7. Estimate implied volatility and explain smiles or skews.
8. Use Monte Carlo simulation to price path-independent and path-dependent options.
9. Compare American exercise, Asian control variates, and barrier continuity corrections.
10. Introduce stochastic volatility with Heston simulation.
11. Use the interactive dashboard to connect moneyness, Greeks, and payoff curvature.

## In-class practice

Students should price one European call and one European put with Black-Scholes, verify put-call parity, estimate Greeks, compare the result with a binomial tree, compute implied volatility from a synthetic option quote, and explain one exotic-option numerical method.

## Module checkpoint

The checkpoint is an option pricing notebook with assumptions, prices, Greeks, a parity check, forward or swap valuation, implied volatility, a comparison between analytical and numerical methods, and a short model-risk discussion.
