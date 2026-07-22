"""Term-structure and short-rate helpers for classroom notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANXICO_DAILY_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "snapshots" / "banxico_daily.csv"
SNAPSHOT_METADATA_PATH = PROJECT_ROOT / "data" / "snapshots" / "metadata.json"


def discount_factor_from_spot_rate(
    rate: float, maturity: float, compounding: str = "continuous"
) -> float:
    """Convert an annual spot rate into a discount factor under a named convention."""
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if compounding == "continuous":
        return float(np.exp(-rate * maturity))
    if compounding == "annual":
        if rate <= -1:
            raise ValueError("an annually compounded rate must be greater than -1")
        return float((1 + rate) ** (-maturity))
    raise ValueError("compounding must be 'continuous' or 'annual'")


def spot_rate_from_discount_factor(
    discount_factor: float, maturity: float, compounding: str = "continuous"
) -> float:
    """Invert :func:`discount_factor_from_spot_rate` under the same convention."""
    if discount_factor <= 0 or maturity <= 0:
        raise ValueError("discount_factor and maturity must be positive")
    if compounding == "continuous":
        return float(-np.log(discount_factor) / maturity)
    if compounding == "annual":
        return float(discount_factor ** (-1 / maturity) - 1)
    raise ValueError("compounding must be 'continuous' or 'annual'")


def forward_rates_from_discount_factors(discount_factors: pd.Series) -> pd.Series:
    """Compute annually compounded forward rates between ordered maturities."""
    if len(discount_factors) < 2:
        raise ValueError("at least two discount factors are required")
    discount_factors = discount_factors.sort_index()
    maturities = discount_factors.index.to_numpy(dtype=float)
    values = discount_factors.to_numpy(dtype=float)
    if np.any(maturities <= 0) or np.any(np.diff(maturities) <= 0):
        raise ValueError("maturities must be unique, positive, and strictly increasing")
    if np.any(values <= 0):
        raise ValueError("discount factors must be positive")
    forwards = []
    labels = []
    for idx in range(1, len(values)):
        delta = maturities[idx] - maturities[idx - 1]
        forward = (values[idx - 1] / values[idx]) ** (1 / delta) - 1
        labels.append(f"{maturities[idx - 1]:g}-{maturities[idx]:g}")
        forwards.append(forward)
    return pd.Series(forwards, index=labels, name="forward_rate")


def par_rate_from_discount_factors(discount_factors: pd.Series, frequency: int = 2) -> float:
    """Compute a par coupon rate from a complete contractual discount grid.

    The index must be the ordered payment grid ``j / frequency`` in years for
    ``j = 1, ..., n``. Discount factors must be finite and positive; the
    function does not reorder or silently fill missing contractual nodes.
    """
    if isinstance(frequency, (bool, np.bool_)) or not isinstance(frequency, (int, np.integer)):
        raise ValueError("frequency must be a positive integer")
    if frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    if not isinstance(discount_factors, pd.Series) or discount_factors.empty:
        raise ValueError("discount_factors must be a non-empty pandas Series")
    if not pd.api.types.is_numeric_dtype(discount_factors.index.dtype):
        raise ValueError("discount-factor maturities must use a numeric index")
    maturities = discount_factors.index.to_numpy(dtype=float)
    try:
        values = discount_factors.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("discount factors must be numeric") from exc
    if not np.all(np.isfinite(maturities)):
        raise ValueError("discount-factor maturities must be finite")
    if np.any(maturities <= 0) or np.any(np.diff(maturities) <= 0):
        raise ValueError("discount-factor maturities must be positive, unique, and ordered")
    expected_grid = np.arange(1, len(maturities) + 1, dtype=float) / frequency
    if not np.allclose(maturities, expected_grid, rtol=0.0, atol=1e-12):
        raise ValueError("discount factors must cover every contractual j/frequency payment date")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("discount factors must be finite and positive")
    discount_sum = float(values.sum())
    final_discount = float(values[-1])
    return float(frequency * (1 - final_discount) / discount_sum)


def bootstrap_coupon_bond_discount_factors(
    instruments: pd.DataFrame, frequency: int = 2
) -> pd.Series:
    """Bootstrap discount factors from par or coupon instruments.

    The input must contain `maturity`, `coupon_rate`, and `price`. Maturities are in
    years and each coupon-bearing row must have earlier discount factors at every
    contractual coupon date. Missing nodes raise ``ValueError`` rather than
    silently dropping cash flows.
    """
    required_columns = {"maturity", "coupon_rate", "price"}
    if missing := required_columns.difference(instruments.columns):
        raise ValueError(f"instruments is missing required columns: {sorted(missing)}")
    if isinstance(frequency, (bool, np.bool_)) or not isinstance(frequency, (int, np.integer)):
        raise ValueError("frequency must be a positive integer")
    if frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    if instruments.empty:
        raise ValueError("instruments must contain at least one row")
    rows = instruments.copy()
    for column in required_columns:
        try:
            rows[column] = rows[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"instrument {column} values must be numeric") from exc
    numeric_values = rows[["maturity", "coupon_rate", "price"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("instrument maturities, coupon rates, and prices must be finite")
    if np.any(rows["coupon_rate"].to_numpy(dtype=float) < 0):
        raise ValueError("instrument coupon rates must be non-negative")
    if np.any(rows["price"].to_numpy(dtype=float) <= 0):
        raise ValueError("instrument prices must be positive")
    rows = rows.sort_values("maturity").reset_index(drop=True)
    maturities = rows["maturity"].to_numpy(dtype=float)
    if np.any(maturities <= 0) or np.any(np.diff(maturities) <= 0):
        raise ValueError("instrument maturities must be unique, positive, and increasing")

    discount_factors: dict[float, float] = {}
    for _, row in rows.iterrows():
        maturity = float(row["maturity"])
        periods_float = maturity * frequency
        periods = int(round(periods_float))
        if not np.isclose(periods_float, periods):
            raise ValueError(f"maturity {maturity:g} is not aligned with frequency {frequency}")
        coupon_rate = float(row["coupon_rate"])
        coupon = coupon_rate / frequency * 100
        price = float(row["price"])

        previous_pv = 0.0
        for period in range(1, periods):
            payment_time = period / frequency
            if np.isclose(coupon, 0.0):
                continue
            if payment_time not in discount_factors:
                raise ValueError(
                    "cannot bootstrap maturity "
                    f"{maturity:g}: missing discount factor for coupon at "
                    f"{payment_time:g} years"
                )
            previous_pv += coupon * discount_factors[payment_time]

        final_cash_flow = 100 + coupon
        discount_factor = (price - previous_pv) / final_cash_flow
        if discount_factor <= 0:
            raise ValueError(
                f"instrument at maturity {maturity:g} implies a non-positive discount factor"
            )
        discount_factors[maturity] = float(discount_factor)
    return pd.Series(discount_factors, name="discount_factor").sort_index()


def nelson_siegel_yield(
    maturities: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> np.ndarray:
    """Nelson-Siegel zero-coupon yield curve."""
    t = _validate_curve_evaluation_inputs(maturities, (beta0, beta1, beta2), (tau,))
    x = t / tau
    loading1 = _nelson_siegel_loading(x)
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
    t = _validate_curve_evaluation_inputs(
        maturities,
        (beta0, beta1, beta2, beta3),
        (tau1, tau2),
    )
    x1 = t / tau1
    x2 = t / tau2
    loading1 = _nelson_siegel_loading(x1)
    loading2 = loading1 - np.exp(-x1)
    loading3 = _nelson_siegel_loading(x2) - np.exp(-x2)
    return beta0 + beta1 * loading1 + beta2 * loading2 + beta3 * loading3


def _validate_curve_evaluation_inputs(
    maturities: np.ndarray,
    betas: tuple[float, ...],
    taus: tuple[float, ...],
) -> np.ndarray:
    """Validate Nelson-Siegel-family inputs without hiding invalid domains."""
    t = np.asarray(maturities, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("maturities must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("maturities must be finite and non-negative")
    if not np.all(np.isfinite(np.asarray(betas, dtype=float))):
        raise ValueError("beta parameters must be finite")
    if not np.all(np.isfinite(np.asarray(taus, dtype=float))) or any(tau <= 0 for tau in taus):
        raise ValueError("tau parameters must be finite and positive")
    return t


def _nelson_siegel_loading(x: np.ndarray) -> np.ndarray:
    """Return ``(1-exp(-x))/x`` with its analytical limit at zero."""
    loading = np.ones_like(x, dtype=float)
    positive = x > 0
    loading[positive] = -np.expm1(-x[positive]) / x[positive]
    return loading


def fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray) -> dict[str, float]:
    """Fit Nelson-Siegel parameters with bounded numerical optimization."""
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    if maturities.ndim != 1 or yields.ndim != 1 or maturities.shape != yields.shape:
        raise ValueError("maturities and yields must be one-dimensional arrays of equal length")
    if len(maturities) < 4 or np.any(maturities <= 0):
        raise ValueError("at least four positive maturities are required")
    if not np.all(np.isfinite(maturities)) or not np.all(np.isfinite(yields)):
        raise ValueError("maturities and yields must be finite")

    def objective(params: np.ndarray) -> float:
        beta0, beta1, beta2, tau = params
        predicted = nelson_siegel_yield(maturities, beta0, beta1, beta2, tau)
        return float(np.sum((yields - predicted) ** 2))

    initial = np.array([yields[-1], yields[0] - yields[-1], 0.01, 2.0])
    bounds = [(0.0, 0.30), (-0.30, 0.30), (-0.30, 0.30), (0.05, 30.0)]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Nelson-Siegel calibration failed: {result.message}")
    beta0, beta1, beta2, tau = result.x
    return {
        "beta0": float(beta0),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "tau": float(tau),
        "sse": float(result.fun),
        "success": bool(result.success),
    }


def rate_panel_pca(
    rate_history: pd.DataFrame, n_components: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run PCA on standardized changes in a documented panel of rates.

    The columns need not be homogeneous yield-curve tenors. Accordingly, the
    returned components are statistical co-movements; callers may use
    level/slope/curvature labels only when the input columns are ordered,
    comparable zero-rate tenors and the loading shapes justify those labels.
    """
    if rate_history.shape[1] < 2:
        raise ValueError("rate_history must contain at least two rate series")
    changes = rate_history.diff().dropna()
    if len(changes) < 2:
        raise ValueError("rate_history must contain at least three complete observations")
    maximum = min(changes.shape)
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(changes)
    model = PCA(n_components=n_components)
    model.fit(scaled)
    components = pd.DataFrame(
        model.components_,
        index=[f"PC{i + 1}" for i in range(n_components)],
        columns=rate_history.columns,
    )
    explained = pd.DataFrame(
        {
            "component": components.index,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(model.explained_variance_ratio_),
        }
    )
    return components, explained


def yield_curve_pca(
    yield_history: pd.DataFrame,
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper for PCA of homogeneous yield-curve tenor histories."""
    return rate_panel_pca(yield_history, n_components=n_components)


def vasicek_ols_calibration(rates: pd.Series, dt: float = 1 / 252) -> dict[str, float]:
    """Estimate Vasicek parameters from the exact AR(1) transition bridge.

    ``dt`` is the annualized interval between adjacent observations and must
    match the caller's sampling grid. The estimator rejects an AR coefficient
    outside ``(0, 1)`` instead of clipping it into a mean-reverting model.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(rates, pd.Series):
        raise ValueError("rates must be a pandas Series")
    try:
        series = rates.dropna().astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("rates must contain numeric observations") from exc
    if len(series) < 4:
        raise ValueError("rates must contain at least four finite observations")
    if not np.all(np.isfinite(series.to_numpy())):
        raise ValueError(
            "rates must contain only finite observations after removing missing values"
        )
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError("a datetime rate index must be unique and increasing")
        elapsed_days = np.diff(series.index.asi8) / (86_400 * 10**9)
        if not np.allclose(elapsed_days, elapsed_days[0], rtol=0.0, atol=1e-9):
            raise ValueError("datetime rate observations must lie on a regular interval")
        observed_dt = float(elapsed_days[0] / 365.25)
        if not np.isclose(dt, observed_dt, rtol=0.02, atol=1e-8):
            raise ValueError("dt is inconsistent with the datetime observation interval")
    lagged = series.shift(1).dropna()
    current = series.loc[lagged.index]
    x = np.column_stack([np.ones(len(lagged)), lagged.to_numpy()])
    if np.linalg.matrix_rank(x) < 2:
        raise ValueError("rates must vary enough to estimate an AR(1) coefficient")
    c, beta = np.linalg.lstsq(x, current.to_numpy(), rcond=None)[0]
    residuals = current.to_numpy() - (c + beta * lagged.to_numpy())
    residual_variance = float(np.var(residuals, ddof=2))
    beta = float(beta)
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("estimated AR(1) beta must lie strictly between 0 and 1")
    if not np.isfinite(residual_variance) or residual_variance < 0:
        raise ValueError("residual variance must be finite and non-negative")
    kappa = -np.log(beta) / dt
    theta = c / (1 - beta)
    sigma = np.sqrt(2 * kappa * residual_variance / (1 - beta**2))
    if not np.all(np.isfinite([c, kappa, theta, sigma])):
        raise ValueError("calibration produced non-finite Vasicek parameters")
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
    if not np.all(np.isfinite([kappa, theta, sigma])):
        raise ValueError("kappa, theta, and sigma must be finite")
    if kappa <= 0 or theta < 0 or sigma < 0:
        raise ValueError("kappa must be positive; theta and sigma must be non-negative")
    return bool(2 * kappa * theta >= sigma**2)


def _validate_simulation_inputs(
    *,
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    years: float,
    steps_per_year: int,
    paths: int,
    seed: int,
    require_non_negative_rates: bool,
) -> int:
    """Validate shared short-rate simulation inputs and return the step count."""
    if not np.all(np.isfinite([r0, kappa, theta, sigma, years])):
        raise ValueError("short-rate parameters and horizon must be finite")
    if kappa <= 0 or sigma < 0 or years <= 0:
        raise ValueError("kappa and years must be positive; sigma must be non-negative")
    if require_non_negative_rates and (r0 < 0 or theta < 0):
        raise ValueError("CIR requires non-negative r0 and theta")
    for value, name in ((steps_per_year, "steps_per_year"), (paths, "paths")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    steps_float = years * steps_per_year
    steps = int(round(steps_float))
    if steps < 1 or not np.isclose(steps_float, steps):
        raise ValueError("years must align with an integer number of simulation steps")
    return steps


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
    steps = _validate_simulation_inputs(
        r0=r0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        years=years,
        steps_per_year=steps_per_year,
        paths=paths,
        seed=seed,
        require_non_negative_rates=False,
    )
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
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
    """Full Truncation Euler-Maruyama simulation for the CIR model.

    The Euler state is allowed to become negative. Its positive part enters
    the drift and diffusion, and only the reported rate is truncated to zero.
    Keeping this auxiliary state distinguishes full truncation from clipping
    the state after every step.
    """
    steps = _validate_simulation_inputs(
        r0=r0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        years=years,
        steps_per_year=steps_per_year,
        paths=paths,
        seed=seed,
        require_non_negative_rates=True,
    )
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    rates = np.empty((steps + 1, paths))
    rates[0] = r0
    state = np.full(paths, r0, dtype=float)
    for step in range(1, steps + 1):
        previous_positive = np.maximum(state, 0.0)
        shock = rng.normal(0.0, np.sqrt(dt), size=paths)
        state = (
            state
            + kappa * (theta - previous_positive) * dt
            + sigma * np.sqrt(previous_positive) * shock
        )
        rates[step] = np.maximum(state, 0.0)
    return pd.DataFrame(rates, index=np.linspace(0, years, steps + 1))


def official_mexican_rate_history(
    start: str | None = None,
    end: str | None = None,
    *,
    frequency: str,
) -> pd.DataFrame:
    """Load the committed Banxico rate panel on an explicit observation grid.

    ``frequency='provider'`` preserves the snapshot's provider-dated rows and
    missing values. ``frequency='weekly'`` selects the last available provider
    observation for each series *within* each ``W-FRI`` week and retains only
    complete weeks; it never carries a value across an empty week. Rates are
    returned as decimals and no network call or synthetic fallback is used.
    """
    if frequency not in {"provider", "weekly"}:
        raise ValueError("frequency must be 'provider' or 'weekly'")
    if not BANXICO_DAILY_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            "Banxico daily snapshot not found. Run "
            "`PYTHONPATH=$PWD uv run python scripts/generate_real_data_snapshots.py` "
            "with valid local credentials."
        )
    if not SNAPSHOT_METADATA_PATH.exists():
        raise FileNotFoundError("snapshot metadata is required for rate-panel provenance")

    start_date = pd.to_datetime(start) if start is not None else None
    end_date = pd.to_datetime(end) if end is not None else None
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start must be on or before end")

    raw = pd.read_csv(BANXICO_DAILY_SNAPSHOT_PATH, index_col="date", parse_dates=True).sort_index()
    required_columns = ["policy_rate", "cetes_28d", "tiie_28d"]
    if missing := set(required_columns).difference(raw.columns):
        raise ValueError(f"Banxico snapshot is missing required columns: {sorted(missing)}")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Banxico snapshot dates must be unique and increasing")
    rates = raw[required_columns].div(100)
    if start_date is not None:
        rates = rates.loc[start_date:]
    if end_date is not None:
        rates = rates.loc[:end_date]
    if frequency == "weekly":
        rates = rates.resample("W-FRI").last().dropna(how="any")
        if start_date is not None:
            rates = rates.loc[start_date:]
        if end_date is not None:
            rates = rates.loc[:end_date]
        alignment = (
            "last provider observation for each series within each W-FRI week; "
            "complete weeks only; no carry across empty weeks"
        )
    else:
        rates = rates.dropna(how="all")
        alignment = "provider-dated rows; missing values preserved; no calendar filling"
    rates = rates.rename_axis("date")
    if rates.empty:
        raise ValueError("the requested date range contains no rate observations")

    with SNAPSHOT_METADATA_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    series_ids = {
        column: metadata.get("banxico_series", {}).get(column) for column in required_columns
    }
    if any(series_id is None for series_id in series_ids.values()):
        raise ValueError("snapshot metadata is missing one or more Banxico series IDs")
    source_vintage = metadata.get("generated_at")
    methodology_breaks = metadata.get("methodology_breaks", {})
    rates.attrs["data_mode"] = "snapshot"
    rates.attrs["offline"] = True
    rates.attrs["provider"] = "Banco de México SIE"
    rates.attrs["source"] = "Banco de México SIE official snapshot"
    rates.attrs["sources"] = rates.attrs["source"]
    rates.attrs["series_ids"] = series_ids
    rates.attrs["series"] = series_ids
    rates.attrs["unit"] = "decimal annual rates"
    rates.attrs["frequency"] = frequency
    rates.attrs["calendar"] = "provider dates" if frequency == "provider" else "W-FRI"
    rates.attrs["alignment"] = alignment
    rates.attrs["observation_policy"] = metadata.get("observation_policies", {}).get(
        "banxico_daily"
    )
    rates.attrs["requested_start"] = (
        start_date.date().isoformat() if start_date is not None else None
    )
    rates.attrs["requested_end"] = end_date.date().isoformat() if end_date is not None else None
    rates.attrs["sample_start"] = rates.index.min().date().isoformat()
    rates.attrs["sample_end"] = rates.index.max().date().isoformat()
    rates.attrs["source_vintage"] = source_vintage
    rates.attrs["retrieved_at"] = source_vintage
    rates.attrs["methodology_breaks"] = methodology_breaks
    rates.attrs["methodology_break"] = methodology_breaks.get("tiie_28d")
    rates.attrs["snapshot_path"] = str(BANXICO_DAILY_SNAPSHOT_PATH.relative_to(PROJECT_ROOT))
    rates.attrs["missing_observations"] = rates.isna().sum().astype(int).to_dict()
    return rates
