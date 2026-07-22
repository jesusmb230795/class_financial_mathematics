"""Portfolio optimization helpers used by the course notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf, OAS


def _as_array(values: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _labels_from_covariance(covariance: pd.DataFrame | np.ndarray) -> list[str]:
    if isinstance(covariance, pd.DataFrame):
        return list(covariance.columns)
    return [f"asset_{idx + 1}" for idx in range(np.asarray(covariance).shape[0])]


def validate_covariance_matrix(
    covariance: pd.DataFrame | np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Validate that a covariance matrix is finite, symmetric, and positive semidefinite."""
    sigma = _as_array(covariance)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(sigma, sigma.T, atol=tolerance, rtol=0):
        raise ValueError("covariance must be symmetric")
    if np.any(np.diag(sigma) <= 0):
        raise ValueError("covariance diagonal entries must be positive")
    minimum_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            "covariance must be positive semidefinite; "
            f"minimum eigenvalue is {minimum_eigenvalue:.6g}"
        )


def project_correlation_to_psd(
    correlation: pd.DataFrame | np.ndarray,
    *,
    minimum_eigenvalue: float = 1e-10,
) -> pd.DataFrame | np.ndarray:
    """Project a symmetric correlation scenario to a positive-semidefinite matrix.

    Eigenvalues are clipped and the result is rescaled to a unit diagonal. This
    transparent classroom guardrail is not an exact nearest-correlation algorithm
    and is not a replacement for estimating a coherent correlation model.
    """
    matrix = _as_array(correlation)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation must contain only finite values")
    if minimum_eigenvalue < 0:
        raise ValueError("minimum_eigenvalue must be non-negative")
    symmetric = (matrix + matrix.T) / 2
    if np.any(np.diag(symmetric) <= 0):
        raise ValueError("correlation diagonal entries must be positive")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.maximum(eigenvalues, minimum_eigenvalue)) @ eigenvectors.T
    scales = np.sqrt(np.diag(projected))
    projected = projected / np.outer(scales, scales)
    projected = (projected + projected.T) / 2
    np.fill_diagonal(projected, 1.0)
    if isinstance(correlation, pd.DataFrame):
        return pd.DataFrame(projected, index=correlation.index, columns=correlation.columns)
    return projected


def nearest_correlation_matrix(
    correlation: pd.DataFrame | np.ndarray,
    *,
    minimum_eigenvalue: float = 1e-10,
) -> pd.DataFrame | np.ndarray:
    """Compatibility alias for the classroom PSD projection."""
    return project_correlation_to_psd(
        correlation,
        minimum_eigenvalue=minimum_eigenvalue,
    )


def project_covariance_to_psd(
    covariance: pd.DataFrame | np.ndarray,
    *,
    minimum_eigenvalue: float = 1e-10,
) -> pd.DataFrame | np.ndarray:
    """Project a covariance matrix to PSD while preserving marginal variances."""
    sigma = _as_array(covariance)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be a square matrix")
    standard_deviations = np.sqrt(np.diag(sigma))
    if np.any(~np.isfinite(standard_deviations)) or np.any(standard_deviations <= 0):
        raise ValueError("covariance diagonal entries must be finite and positive")
    correlation = sigma / np.outer(standard_deviations, standard_deviations)
    if isinstance(covariance, pd.DataFrame):
        correlation = pd.DataFrame(
            correlation,
            index=covariance.index,
            columns=covariance.columns,
        )
    projected_correlation = project_correlation_to_psd(
        correlation,
        minimum_eigenvalue=minimum_eigenvalue,
    )
    projected = _as_array(projected_correlation) * np.outer(
        standard_deviations,
        standard_deviations,
    )
    if isinstance(covariance, pd.DataFrame):
        return pd.DataFrame(projected, index=covariance.index, columns=covariance.columns)
    return projected


def annualized_mean_returns(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """Estimate annualized arithmetic mean returns from periodic returns."""
    return returns.mean() * periods_per_year


def sample_covariance(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate an annualized sample covariance matrix."""
    return returns.cov() * periods_per_year


def portfolio_return(
    weights: pd.Series | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
) -> float:
    """Compute expected portfolio return."""
    return float(np.dot(_as_array(weights), _as_array(expected_returns)))


def portfolio_variance(
    weights: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> float:
    """Compute portfolio variance."""
    w = _as_array(weights)
    sigma = _as_array(covariance)
    validate_covariance_matrix(sigma)
    if w.ndim != 1 or len(w) != sigma.shape[0]:
        raise ValueError("weights length must match covariance dimensions")
    return float(max(w.T @ sigma @ w, 0.0))


def portfolio_volatility(
    weights: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> float:
    """Compute portfolio volatility."""
    return float(np.sqrt(portfolio_variance(weights, covariance)))


def merton_constants(
    expected_returns: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Return the analytical efficient-frontier constants A, B, C, and D."""
    mu = _as_array(expected_returns)
    sigma = _as_array(covariance)
    validate_covariance_matrix(sigma)
    if mu.ndim != 1 or len(mu) != sigma.shape[0]:
        raise ValueError("expected_returns length must match covariance dimensions")
    sigma_inv = np.linalg.pinv(sigma)
    ones = np.ones_like(mu)

    a = float(ones.T @ sigma_inv @ ones)
    b = float(ones.T @ sigma_inv @ mu)
    c = float(mu.T @ sigma_inv @ mu)
    d = float(a * c - b**2)
    if d <= 1e-14:
        raise ValueError("efficient frontier is degenerate for these inputs")
    return pd.Series({"A": a, "B": b, "C": c, "D": d})


def global_minimum_variance_weights(
    covariance: pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Return unconstrained global minimum variance portfolio weights."""
    sigma = _as_array(covariance)
    validate_covariance_matrix(sigma)
    sigma_inv = np.linalg.pinv(sigma)
    ones = np.ones(sigma_inv.shape[0])
    raw = sigma_inv @ ones
    weights = raw / (ones.T @ raw)
    return pd.Series(weights, index=_labels_from_covariance(covariance), name="gmvp")


def tangency_weights(
    expected_returns: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Return unconstrained tangency portfolio weights."""
    mu = _as_array(expected_returns)
    sigma = _as_array(covariance)
    validate_covariance_matrix(sigma)
    if mu.ndim != 1 or len(mu) != sigma.shape[0]:
        raise ValueError("expected_returns length must match covariance dimensions")
    sigma_inv = np.linalg.pinv(sigma)
    excess = mu - risk_free_rate
    raw = sigma_inv @ excess
    denominator = raw.sum()
    if np.isclose(denominator, 0):
        raise ValueError("tangency weights are undefined when excess-return exposure sums to zero")
    weights = raw / denominator
    return pd.Series(weights, index=_labels_from_covariance(covariance), name="tangency")


def efficient_frontier_variance(
    target_returns: float | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Compute analytical minimum variance for each target return."""
    constants = merton_constants(expected_returns, covariance)
    target = np.asarray(target_returns, dtype=float)
    return (constants["A"] * target**2 - 2 * constants["B"] * target + constants["C"]) / constants[
        "D"
    ]


def efficient_frontier_weights(
    target_return: float,
    expected_returns: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Return unconstrained efficient-frontier weights for a target return."""
    mu = _as_array(expected_returns)
    sigma_inv = np.linalg.pinv(_as_array(covariance))
    ones = np.ones_like(mu)
    constants = merton_constants(expected_returns, covariance)
    lambda_ = (constants["C"] - constants["B"] * target_return) / constants["D"]
    gamma = (constants["A"] * target_return - constants["B"]) / constants["D"]
    weights = sigma_inv @ (lambda_ * ones + gamma * mu)
    return pd.Series(weights, index=_labels_from_covariance(covariance), name="frontier")


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, float]:
    """Estimate annualized Ledoit-Wolf covariance and shrinkage intensity."""
    estimator = LedoitWolf().fit(returns.dropna().to_numpy())
    covariance = pd.DataFrame(
        estimator.covariance_ * periods_per_year,
        index=returns.columns,
        columns=returns.columns,
    )
    return covariance, float(estimator.shrinkage_)


def oas_covariance(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, float]:
    """Estimate annualized Oracle Approximating Shrinkage covariance."""
    estimator = OAS().fit(returns.dropna().to_numpy())
    covariance = pd.DataFrame(
        estimator.covariance_ * periods_per_year,
        index=returns.columns,
        columns=returns.columns,
    )
    return covariance, float(estimator.shrinkage_)


def risk_contributions(
    weights: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Compute absolute volatility risk contributions by asset."""
    w = _as_array(weights)
    sigma = _as_array(covariance)
    volatility = np.sqrt(w.T @ sigma @ w)
    if np.isclose(volatility, 0):
        raise ValueError("portfolio volatility must be positive")
    contributions = w * (sigma @ w) / volatility
    return pd.Series(
        contributions, index=_labels_from_covariance(covariance), name="risk_contribution"
    )


def risk_contribution_percentages(
    weights: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Compute percentage volatility risk contributions by asset."""
    contributions = risk_contributions(weights, covariance)
    return (contributions / contributions.sum()).rename("risk_contribution_pct")


def inverse_variance_weights(covariance: pd.DataFrame | np.ndarray) -> pd.Series:
    """Return long-only inverse-variance weights."""
    variances = np.diag(_as_array(covariance))
    if np.any(variances <= 0):
        raise ValueError("all variances must be positive")
    raw = 1 / variances
    weights = raw / raw.sum()
    return pd.Series(weights, index=_labels_from_covariance(covariance), name="inverse_variance")


def risk_parity_weights(
    covariance: pd.DataFrame | np.ndarray,
    target_risk_budget: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """Estimate long-only equal-risk-contribution weights."""
    labels = _labels_from_covariance(covariance)
    sigma = _as_array(covariance)
    n_assets = sigma.shape[0]
    if target_risk_budget is None:
        budget = np.ones(n_assets) / n_assets
    else:
        budget = _as_array(target_risk_budget)
        budget = budget / budget.sum()

    initial = inverse_variance_weights(covariance).to_numpy()

    def objective(weights: np.ndarray) -> float:
        contribution_pct = risk_contribution_percentages(weights, sigma).to_numpy()
        return float(np.sum((contribution_pct - budget) ** 2))

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=[(1e-8, 1.0) for _ in range(n_assets)],
        constraints=({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"risk parity optimization failed: {result.message}")
    return pd.Series(result.x, index=labels, name="risk_parity")


def capm_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    hac_lags: int | None = None,
) -> pd.Series:
    """Estimate CAPM alpha and beta against an economically valid market proxy."""
    frame = pd.concat(
        [
            asset_returns.rename("asset"),
            market_returns.rename("market"),
            pd.Series(risk_free_rate, index=asset_returns.index, name="risk_free"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("CAPM estimation requires overlapping observations")

    y = (frame["asset"] - frame["risk_free"]).rename("asset")
    market_excess = (frame["market"] - frame["risk_free"]).rename("market")
    x = sm.add_constant(market_excess)
    if hac_lags is None:
        hac_lags = int(np.floor(len(frame) ** 0.25))
    result = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return pd.Series(
        {
            "alpha": result.params["const"],
            "beta": result.params["market"],
            "alpha_p_value": result.pvalues["const"],
            "beta_p_value": result.pvalues["market"],
            "r_squared": result.rsquared,
            "nobs": result.nobs,
            "hac_lags": hac_lags,
        }
    )


def factor_sensitivity(
    asset_returns: pd.Series,
    factor_returns: pd.Series,
    hac_lags: int | None = None,
) -> pd.Series:
    """Estimate an exploratory linear factor sensitivity with HAC standard errors."""
    frame = pd.concat(
        [
            asset_returns.rename("asset"),
            factor_returns.rename("factor"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("factor sensitivity requires overlapping observations")
    x = sm.add_constant(frame["factor"])
    if hac_lags is None:
        hac_lags = int(np.floor(len(frame) ** 0.25))
    result = sm.OLS(frame["asset"], x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )
    return pd.Series(
        {
            "intercept": result.params["const"],
            "sensitivity": result.params["factor"],
            "intercept_p_value": result.pvalues["const"],
            "sensitivity_p_value": result.pvalues["factor"],
            "r_squared": result.rsquared,
            "nobs": result.nobs,
            "hac_lags": hac_lags,
        }
    )


def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Estimate rolling beta using rolling covariance divided by market variance."""
    frame = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")], axis=1
    ).dropna()
    covariance = frame["asset"].rolling(window).cov(frame["market"])
    variance = frame["market"].rolling(window).var()
    return (covariance / variance).rename("rolling_beta")


def rolling_factor_sensitivity(
    asset_returns: pd.Series,
    factor_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Estimate a rolling linear sensitivity to a documented factor."""
    if window < 2:
        raise ValueError("window must be at least 2")
    frame = pd.concat(
        [
            asset_returns.rename("asset"),
            factor_returns.rename("factor"),
        ],
        axis=1,
    ).dropna()
    covariance = frame["asset"].rolling(window).cov(frame["factor"])
    variance = frame["factor"].rolling(window).var()
    return (covariance / variance).rename("rolling_factor_sensitivity")


def correlation_distance(correlation: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix into the HRP distance matrix."""
    distance = np.sqrt(0.5 * (1 - correlation.clip(-1, 1)))
    np.fill_diagonal(distance.values, 0.0)
    return distance


def cluster_variance(covariance: pd.DataFrame, cluster_items: list[str]) -> float:
    """Compute inverse-variance cluster variance for HRP recursive bisection."""
    cluster_covariance = covariance.loc[cluster_items, cluster_items]
    weights = inverse_variance_weights(cluster_covariance)
    return portfolio_variance(weights, cluster_covariance)


def hierarchical_risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """Estimate Hierarchical Risk Parity weights from an asset return matrix."""
    clean = returns.dropna()
    if clean.shape[1] == 1:
        return pd.Series([1.0], index=clean.columns, name="hrp")

    covariance = clean.cov()
    correlation = clean.corr()
    distance = correlation_distance(correlation)
    condensed_distance = squareform(distance.to_numpy(), checks=False)
    linkage_matrix = linkage(condensed_distance, method="single")
    ordered_labels = list(clean.columns[leaves_list(linkage_matrix)])

    weights = pd.Series(1.0, index=ordered_labels)
    clusters = [ordered_labels]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            left_variance = cluster_variance(covariance, left)
            right_variance = cluster_variance(covariance, right)
            left_allocation = 1 - left_variance / (left_variance + right_variance)
            weights.loc[left] *= left_allocation
            weights.loc[right] *= 1 - left_allocation
            next_clusters.extend([left, right])
        clusters = next_clusters

    return weights.reindex(clean.columns).rename("hrp")
