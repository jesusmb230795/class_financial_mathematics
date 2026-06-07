"""Term-structure and short-rate helpers for classroom notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANXICO_DAILY_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "snapshots" / "banxico_daily.csv"


def discount_factor_from_spot_rate(rate: float, maturity: float, compounding: str = "continuous") -> float:
    """Convert a spot rate into a discount factor."""
    if compounding == "continuous":
        return float(np.exp(-rate * maturity))
    if compounding == "annual":
        return float((1 + rate) ** (-maturity))
    raise ValueError("compounding must be 'continuous' or 'annual'")


def spot_rate_from_discount_factor(discount_factor: float, maturity: float, compounding: str = "continuous") -> float:
    """Convert a discount factor into a spot rate."""
    if compounding == "continuous":
        return float(-np.log(discount_factor) / maturity)
    if compounding == "annual":
        return float(discount_factor ** (-1 / maturity) - 1)
    raise ValueError("compounding must be 'continuous' or 'annual'")


def forward_rates_from_discount_factors(discount_factors: pd.Series) -> pd.Series:
    """Compute discrete forward rates between ordered maturities."""
    maturities = discount_factors.index.to_numpy(dtype=float)
    values = discount_factors.to_numpy(dtype=float)
    forwards = []
    labels = []
    for idx in range(1, len(values)):
        delta = maturities[idx] - maturities[idx - 1]
        forward = (values[idx - 1] / values[idx] - 1) / delta
        labels.append(f"{maturities[idx - 1]:g}-{maturities[idx]:g}")
        forwards.append(forward)
    return pd.Series(forwards, index=labels, name="forward_rate")


def par_rate_from_discount_factors(discount_factors: pd.Series, frequency: int = 2) -> float:
    """Compute the par coupon rate implied by discount factors."""
    discount_sum = discount_factors.sum()
    final_discount = float(discount_factors.iloc[-1])
    return float(frequency * (1 - final_discount) / discount_sum)


def bootstrap_coupon_bond_discount_factors(instruments: pd.DataFrame, frequency: int = 2) -> pd.Series:
    """Bootstrap discount factors from par or coupon instruments.

    The input must contain `maturity`, `coupon_rate`, and `price`. Maturities are in
    years and should be ordered from shortest to longest. The function assumes each
    row adds one new maturity point.
    """
    rows = instruments.sort_values("maturity").reset_index(drop=True)
    discount_factors: dict[float, float] = {}
    for _, row in rows.iterrows():
        maturity = float(row["maturity"])
        coupon = float(row["coupon_rate"]) / frequency * 100
        price = float(row["price"])
        previous_pv = sum(coupon * df for mat, df in discount_factors.items() if mat < maturity)
        final_cash_flow = 100 + coupon
        discount_factors[maturity] = (price - previous_pv) / final_cash_flow
    return pd.Series(discount_factors, name="discount_factor").sort_index()


def nelson_siegel_yield(
    maturities: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> np.ndarray:
    """Nelson-Siegel zero-coupon yield curve."""
    t = np.asarray(maturities, dtype=float)
    x = np.maximum(t / tau, 1e-8)
    loading1 = (1 - np.exp(-x)) / x
    loading2 = loading1 - np.exp(-x)
    return beta0 + beta1 * loading1 + beta2 * loading2


def nelson_siegel_svensson_yield(
    maturities: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """Nelson-Siegel-Svensson zero-coupon yield curve."""
    t = np.asarray(maturities, dtype=float)
    x1 = np.maximum(t / tau1, 1e-8)
    x2 = np.maximum(t / tau2, 1e-8)
    loading1 = (1 - np.exp(-x1)) / x1
    loading2 = loading1 - np.exp(-x1)
    loading3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return beta0 + beta1 * loading1 + beta2 * loading2 + beta3 * loading3


def fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray) -> dict[str, float]:
    """Fit Nelson-Siegel parameters with bounded numerical optimization."""
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)

    def objective(params: np.ndarray) -> float:
        beta0, beta1, beta2, tau = params
        predicted = nelson_siegel_yield(maturities, beta0, beta1, beta2, tau)
        return float(np.sum((yields - predicted) ** 2))

    initial = np.array([yields[-1], yields[0] - yields[-1], 0.01, 2.0])
    bounds = [(0.0, 0.30), (-0.30, 0.30), (-0.30, 0.30), (0.05, 30.0)]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    beta0, beta1, beta2, tau = result.x
    return {
        "beta0": float(beta0),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "tau": float(tau),
        "sse": float(result.fun),
        "success": bool(result.success),
    }


def yield_curve_pca(yield_history: pd.DataFrame, n_components: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run PCA on standardized daily yield changes."""
    changes = yield_history.diff().dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(changes)
    model = PCA(n_components=n_components)
    model.fit(scaled)
    components = pd.DataFrame(
        model.components_,
        index=[f"PC{i + 1}" for i in range(n_components)],
        columns=yield_history.columns,
    )
    explained = pd.DataFrame(
        {
            "component": components.index,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(model.explained_variance_ratio_),
        }
    )
    return components, explained


def vasicek_ols_calibration(rates: pd.Series, dt: float = 1 / 252) -> dict[str, float]:
    """Estimate Vasicek parameters from the AR(1) bridge."""
    series = rates.dropna().astype(float)
    lagged = series.shift(1).dropna()
    current = series.loc[lagged.index]
    x = np.column_stack([np.ones(len(lagged)), lagged.to_numpy()])
    c, beta = np.linalg.lstsq(x, current.to_numpy(), rcond=None)[0]
    residuals = current.to_numpy() - (c + beta * lagged.to_numpy())
    residual_variance = float(np.var(residuals, ddof=2))
    beta = min(max(float(beta), 1e-6), 0.999999)
    kappa = -np.log(beta) / dt
    theta = c / (1 - beta)
    sigma = np.sqrt(2 * kappa * residual_variance / (1 - beta**2))
    return {
        "c": float(c),
        "beta": float(beta),
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "residual_std": float(np.sqrt(residual_variance)),
    }


def feller_condition(kappa: float, theta: float, sigma: float) -> bool:
    """Return True when the CIR Feller condition is satisfied."""
    return bool(2 * kappa * theta >= sigma**2)


def simulate_vasicek_exact(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    years: float,
    steps_per_year: int = 252,
    paths: int = 100,
    seed: int = 2026,
) -> pd.DataFrame:
    """Exact Gaussian transition simulation for the Vasicek model."""
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    steps = int(years * steps_per_year)
    rates = np.empty((steps + 1, paths))
    rates[0] = r0
    decay = np.exp(-kappa * dt)
    variance = sigma**2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    for step in range(1, steps + 1):
        mean = theta + (rates[step - 1] - theta) * decay
        rates[step] = mean + np.sqrt(variance) * rng.normal(size=paths)
    return pd.DataFrame(rates, index=np.linspace(0, years, steps + 1))


def simulate_cir_full_truncation(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    years: float,
    steps_per_year: int = 252,
    paths: int = 100,
    seed: int = 2027,
) -> pd.DataFrame:
    """Full Truncation Euler-Maruyama simulation for the CIR model."""
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    steps = int(years * steps_per_year)
    rates = np.empty((steps + 1, paths))
    rates[0] = r0
    for step in range(1, steps + 1):
        previous_positive = np.maximum(rates[step - 1], 0.0)
        shock = rng.normal(0.0, np.sqrt(dt), size=paths)
        rates[step] = (
            rates[step - 1]
            + kappa * (theta - previous_positive) * dt
            + sigma * np.sqrt(previous_positive) * shock
        )
        rates[step] = np.maximum(rates[step], 0.0)
    return pd.DataFrame(rates, index=np.linspace(0, years, steps + 1))


def official_mexican_rate_history(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load real Banxico rate history from the committed publication snapshot."""
    if not BANXICO_DAILY_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            "Banxico daily snapshot not found. Run "
            "`PYTHONPATH=$PWD uv run python scripts/generate_real_data_snapshots.py` "
            "with valid local credentials."
        )

    raw = pd.read_csv(BANXICO_DAILY_SNAPSHOT_PATH, index_col="date", parse_dates=True).sort_index()
    rates = raw[["policy_rate", "cetes_28d", "tiie_28d"]].div(100)
    rates = rates.ffill().dropna().rename_axis("date")
    if start is not None:
        rates = rates.loc[pd.to_datetime(start) :]
    if end is not None:
        rates = rates.loc[: pd.to_datetime(end)]
    rates.attrs["data_mode"] = "snapshot"
    rates.attrs["sources"] = "Banxico SIE official snapshot"
    rates.attrs["series"] = {
        "policy_rate": "SF61745",
        "cetes_28d": "SF60633",
        "tiie_28d": "SF60648",
    }
    return rates
