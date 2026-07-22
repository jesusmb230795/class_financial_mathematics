"""Fixed-income valuation helpers for classroom notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def _validate_time_and_compounding(time: float, compounding: int | None) -> None:
    if time < 0:
        raise ValueError("time must be non-negative")
    if compounding is not None and (
        not isinstance(compounding, (int, np.integer)) or compounding <= 0
    ):
        raise ValueError("compounding must be a positive integer or None for continuous")


def discount_factor(rate: float, time: float, compounding: int | None = 1) -> float:
    """Return a discount factor under nominal or continuous compounding.

    ``compounding=m`` treats ``rate`` as a nominal annual rate convertible ``m``
    times per year. ``compounding=None`` treats it as a continuously compounded
    annual rate.
    """
    _validate_time_and_compounding(time, compounding)
    if compounding is None:
        return float(np.exp(-rate * time))
    if 1 + rate / compounding <= 0:
        raise ValueError("rate is outside the domain of discrete compounding")
    return float((1 + rate / compounding) ** (-compounding * time))


def present_value(
    cash_flows: np.ndarray,
    times: np.ndarray,
    rate: float,
    compounding: int | None = 1,
) -> float:
    """Present value of dated cash flows under a flat rate."""
    cash_flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(times, dtype=float)
    if cash_flows.ndim != 1 or times.ndim != 1 or cash_flows.shape != times.shape:
        raise ValueError("cash_flows and times must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(cash_flows)) or not np.all(np.isfinite(times)):
        raise ValueError("cash_flows and times must contain finite values")
    if np.any(times < 0):
        raise ValueError("cash-flow times must be non-negative")
    factors = np.array([discount_factor(rate, t, compounding) for t in times])
    return float(np.sum(cash_flows * factors))


def future_value(present: float, rate: float, time: float, compounding: int | None = 1) -> float:
    """Future value under the same convention used by :func:`discount_factor`."""
    _validate_time_and_compounding(time, compounding)
    if compounding is None:
        return float(present * np.exp(rate * time))
    if 1 + rate / compounding <= 0:
        raise ValueError("rate is outside the domain of discrete compounding")
    return float(present * (1 + rate / compounding) ** (compounding * time))


def simple_future_value(present: float, rate: float, time: float) -> float:
    """Future value under simple interest, ``FV = PV(1 + rT)``."""
    if time < 0:
        raise ValueError("time must be non-negative")
    if 1 + rate * time <= 0:
        raise ValueError("rate and time imply a non-positive accumulation factor")
    return float(present * (1 + rate * time))


def simple_present_value(future: float, rate: float, time: float) -> float:
    """Present value under simple interest."""
    if time < 0:
        raise ValueError("time must be non-negative")
    accumulation = 1 + rate * time
    if accumulation <= 0:
        raise ValueError("rate and time imply a non-positive accumulation factor")
    return float(future / accumulation)


def effective_annual_rate(nominal_rate: float, compounding: int) -> float:
    """Convert a nominal annual rate convertible ``m`` times to an effective rate."""
    _validate_time_and_compounding(1.0, compounding)
    if 1 + nominal_rate / compounding <= 0:
        raise ValueError("nominal_rate is outside the compounding domain")
    return float((1 + nominal_rate / compounding) ** compounding - 1)


def nominal_rate_from_effective(effective_rate: float, compounding: int) -> float:
    """Convert an effective annual rate to a nominal annual rate convertible ``m`` times."""
    _validate_time_and_compounding(1.0, compounding)
    if effective_rate <= -1:
        raise ValueError("effective_rate must be greater than -1")
    return float(compounding * ((1 + effective_rate) ** (1 / compounding) - 1))


def continuous_rate_from_effective(effective_rate: float) -> float:
    """Convert an effective annual rate to a continuously compounded annual rate."""
    if effective_rate <= -1:
        raise ValueError("effective_rate must be greater than -1")
    return float(np.log1p(effective_rate))


def effective_rate_from_continuous(continuous_rate: float) -> float:
    """Convert a continuously compounded annual rate to an effective annual rate."""
    return float(np.expm1(continuous_rate))


def annuity_payment(
    principal: float, annual_rate: float, years: float, payments_per_year: int = 12
) -> float:
    """Level payment for an amortizing loan."""
    if principal <= 0 or years <= 0 or payments_per_year <= 0:
        raise ValueError("principal, years, and payments_per_year must be positive")
    periods = int(round(years * payments_per_year))
    period_rate = annual_rate / payments_per_year
    if 1 + period_rate <= 0:
        raise ValueError("annual_rate is outside the payment-frequency domain")
    if period_rate == 0:
        return principal / periods
    return float(principal * period_rate / (1 - (1 + period_rate) ** (-periods)))


def amortization_schedule(
    principal: float,
    annual_rate: float,
    years: float,
    payments_per_year: int = 12,
) -> pd.DataFrame:
    """Build a deterministic amortization table."""
    payment = annuity_payment(principal, annual_rate, years, payments_per_year)
    period_rate = annual_rate / payments_per_year
    balance = principal
    rows = []
    for period in range(1, int(round(years * payments_per_year)) + 1):
        interest = balance * period_rate
        principal_paid = min(payment - interest, balance)
        balance = max(balance - principal_paid, 0.0)
        rows.append(
            {
                "period": period,
                "payment": payment,
                "interest": interest,
                "principal": principal_paid,
                "ending_balance": balance,
            }
        )
    return pd.DataFrame(rows)


def coupon_bond_cash_flows(
    face_value: float,
    coupon_rate: float,
    maturity: float,
    frequency: int,
) -> pd.DataFrame:
    """Build a regular coupon-bond schedule with no implicit stub period.

    ``maturity`` is measured in years and must lie exactly on the contractual
    grid ``n / frequency``. Irregular first or final coupons require an
    explicit schedule and are intentionally outside this classroom helper.
    """
    values = np.asarray([face_value, coupon_rate, maturity], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("face_value, coupon_rate, and maturity must be finite")
    if face_value <= 0 or maturity <= 0:
        raise ValueError("face_value and maturity must be positive")
    if coupon_rate < 0:
        raise ValueError("coupon_rate must be non-negative")
    if isinstance(frequency, (bool, np.bool_)) or not isinstance(frequency, (int, np.integer)):
        raise ValueError("frequency must be a positive integer")
    if frequency <= 0:
        raise ValueError("frequency must be a positive integer")

    periods_float = maturity * frequency
    periods = int(round(periods_float))
    if not np.isclose(periods_float, periods, rtol=0.0, atol=1e-12):
        raise ValueError("maturity must align with the coupon frequency; stubs are not implicit")

    coupon = face_value * coupon_rate / frequency
    times = np.arange(1, periods + 1, dtype=float) / frequency
    times[-1] = maturity
    cash_flows = np.full(periods, coupon, dtype=float)
    cash_flows[-1] += face_value
    return pd.DataFrame(
        {
            "period": np.arange(1, periods + 1),
            "time": times,
            "cash_flow": cash_flows,
        }
    )


@dataclass(frozen=True)
class Cetes:
    """Mexican Treasury certificate priced with ACT/360 money-market yield."""

    days_to_maturity: int
    annual_yield: float
    face_value: float = 10.0
    day_count: int = 360

    def __post_init__(self) -> None:
        if self.days_to_maturity <= 0:
            raise ValueError("days_to_maturity must be positive")
        if self.face_value <= 0 or self.day_count <= 0:
            raise ValueError("face_value and day_count must be positive")
        if 1 + self.annual_yield * self.days_to_maturity / self.day_count <= 0:
            raise ValueError("annual_yield implies a non-positive price denominator")

    @property
    def price(self) -> float:
        denominator = 1 + self.annual_yield * self.days_to_maturity / self.day_count
        return float(self.face_value / denominator)

    @property
    def discount_rate(self) -> float:
        return cetes_discount_rate_from_yield(
            self.annual_yield, self.days_to_maturity, self.day_count
        )


def cetes_price(
    annual_yield: float,
    days_to_maturity: int,
    face_value: float = 10.0,
    day_count: int = 360,
) -> float:
    """Price a CETES instrument from its annualized ACT/360 yield."""
    return Cetes(days_to_maturity, annual_yield, face_value, day_count).price


def cetes_discount_rate_from_yield(
    annual_yield: float, days_to_maturity: int, day_count: int = 360
) -> float:
    """Convert CETES return yield into the quoted discount-rate basis."""
    if days_to_maturity <= 0 or day_count <= 0:
        raise ValueError("days_to_maturity and day_count must be positive")
    if 1 + annual_yield * days_to_maturity / day_count <= 0:
        raise ValueError("annual_yield is outside the quotation domain")
    return float(annual_yield / (1 + annual_yield * days_to_maturity / day_count))


def cetes_yield_from_discount_rate(
    discount_rate: float, days_to_maturity: int, day_count: int = 360
) -> float:
    """Convert CETES discount-rate quote into return-yield basis."""
    if days_to_maturity <= 0 or day_count <= 0:
        raise ValueError("days_to_maturity and day_count must be positive")
    if 1 - discount_rate * days_to_maturity / day_count <= 0:
        raise ValueError("discount_rate is outside the quotation domain")
    return float(discount_rate / (1 - discount_rate * days_to_maturity / day_count))


@dataclass(frozen=True)
class BonoM:
    """Simplified Bono M valuation with 182-day coupons and a declared basis.

    ``day_count`` is used consistently for the coupon cash amount, the yield per
    coupon period, and accrued interest. Changing it therefore changes the whole
    simplified quotation convention, not accrued interest alone.
    """

    coupon_rate: float
    annual_yield: float
    remaining_coupons: int
    days_since_last_coupon: int = 0
    face_value: float = 100.0
    coupon_days: int = 182
    day_count: int = 360

    def __post_init__(self) -> None:
        if self.remaining_coupons <= 0:
            raise ValueError("remaining_coupons must be positive")
        if not 0 <= self.days_since_last_coupon < self.coupon_days:
            raise ValueError("days_since_last_coupon must be in [0, coupon_days)")
        if self.face_value <= 0 or self.coupon_days <= 0 or self.day_count <= 0:
            raise ValueError("face_value, coupon_days, and day_count must be positive")
        period_yield = self.annual_yield * self.coupon_days / self.day_count
        if 1 + period_yield <= 0:
            raise ValueError("annual_yield is outside the coupon-period domain")

    @property
    def coupon_payment(self) -> float:
        return self.face_value * self.coupon_rate * self.coupon_days / self.day_count

    @property
    def fractional_periods(self) -> np.ndarray:
        periods = np.arange(1, self.remaining_coupons + 1, dtype=float)
        fraction_to_next_coupon = (
            self.coupon_days - self.days_since_last_coupon
        ) / self.coupon_days
        return periods - 1 + fraction_to_next_coupon

    @property
    def cash_flows(self) -> np.ndarray:
        flows = np.full(self.remaining_coupons, self.coupon_payment, dtype=float)
        flows[-1] += self.face_value
        return flows

    @property
    def dirty_price(self) -> float:
        period_yield = self.annual_yield * self.coupon_days / self.day_count
        factors = (1 + period_yield) ** (-self.fractional_periods)
        return float(np.sum(self.cash_flows * factors))

    @property
    def accrued_interest(self) -> float:
        return float(
            self.face_value * self.coupon_rate * self.days_since_last_coupon / self.day_count
        )

    @property
    def clean_price(self) -> float:
        return self.dirty_price - self.accrued_interest


def bono_m_price_table(bond: BonoM) -> pd.DataFrame:
    """Return the cash-flow and present-value table for a Bono M."""
    period_yield = bond.annual_yield * bond.coupon_days / bond.day_count
    factors = (1 + period_yield) ** (-bond.fractional_periods)
    return pd.DataFrame(
        {
            "period": np.arange(1, bond.remaining_coupons + 1),
            "fractional_period": bond.fractional_periods,
            "cash_flow": bond.cash_flows,
            "discount_factor": factors,
            "present_value": bond.cash_flows * factors,
        }
    )


def udibono_settlement_mxn(
    clean_price_udis: float, accrued_interest_udis: float, udi_value: float
) -> float:
    """Convert an UDIBONO clean price and accrued interest in UDIS into MXN settlement."""
    return float((clean_price_udis + accrued_interest_udis) * udi_value)


def yield_to_maturity_from_cash_flows(
    price: float,
    cash_flows: np.ndarray,
    times: np.ndarray,
    compounding: int | None = 1,
    lower: float = -0.95,
    upper: float = 1.50,
) -> float:
    """Solve the unique flat annual YTM of conventional promised cash flows.

    Cash flows must be non-negative and include at least one strictly positive
    payment after settlement. This domain makes present value strictly
    decreasing in the rate and excludes investment streams that can have
    multiple internal rates of return.
    """
    cash_flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(times, dtype=float)
    if not np.isfinite(price) or price <= 0:
        raise ValueError("price must be finite and positive")
    if cash_flows.ndim != 1 or times.ndim != 1 or cash_flows.shape != times.shape:
        raise ValueError("cash_flows and times must be one-dimensional arrays of equal length")
    if cash_flows.size == 0:
        raise ValueError("cash_flows and times must not be empty")
    if not np.all(np.isfinite(cash_flows)) or not np.all(np.isfinite(times)):
        raise ValueError("cash_flows and times must contain finite values")
    if np.any(cash_flows < 0):
        raise ValueError("cash_flows must be non-negative for a unique conventional-bond YTM")
    if np.any(times < 0):
        raise ValueError("cash-flow times must be non-negative")
    if not np.any((cash_flows > 0) & (times > 0)):
        raise ValueError("at least one positive cash flow must occur after settlement")
    if not np.all(np.isfinite([lower, upper])) or lower >= upper:
        raise ValueError("lower and upper must be finite with lower less than upper")
    present_value(cash_flows, times, 0.0, compounding)

    def error(rate: float) -> float:
        return present_value(cash_flows, times, rate, compounding=compounding) - price

    return float(brentq(error, lower, upper))


def risk_measures_from_cash_flows(
    cash_flows: np.ndarray,
    times: np.ndarray,
    annual_yield: float,
    frequency: int = 2,
) -> dict[str, float]:
    """Compute price and interest-rate risk under periodic compounding.

    This helper covers conventional bonds with non-negative promised cash
    flows. Rates are decimal nominal annual rates convertible ``frequency``
    times per year, and ``times`` are measured in years. ``dv01`` is returned
    as a non-negative price-point magnitude: the first-order price change for
    a parallel ``+1`` basis-point yield shock is approximately ``-dv01``.
    """
    cash_flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(times, dtype=float)
    if cash_flows.ndim != 1 or times.ndim != 1 or cash_flows.shape != times.shape:
        raise ValueError("cash_flows and times must be one-dimensional arrays of equal length")
    if cash_flows.size == 0:
        raise ValueError("cash_flows and times must not be empty")
    if not np.all(np.isfinite(cash_flows)) or not np.all(np.isfinite(times)):
        raise ValueError("cash_flows and times must contain finite values")
    if np.any(cash_flows < 0) or not np.any(cash_flows > 0):
        raise ValueError("cash_flows must be non-negative with at least one positive payment")
    if np.any(times < 0):
        raise ValueError("cash-flow times must be non-negative")
    if isinstance(frequency, (bool, np.bool_)) or not isinstance(frequency, (int, np.integer)):
        raise ValueError("frequency must be a positive integer")
    if frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    if not np.isfinite(annual_yield):
        raise ValueError("annual_yield must be finite")
    period_accumulation = 1 + annual_yield / frequency
    if period_accumulation <= 0:
        raise ValueError("annual_yield is outside the periodic-compounding domain")

    periods = times * frequency
    discount = period_accumulation ** (-periods)
    present_values = cash_flows * discount
    price = float(present_values.sum())
    if not np.isfinite(price) or price <= 0:
        raise ValueError("discounted cash flows must imply a positive finite price")
    weights = present_values / price
    macaulay = float(np.sum(times * weights))
    modified = macaulay / period_accumulation
    convexity = float(
        np.sum(present_values * periods * (periods + 1))
        / (price * period_accumulation**2 * frequency**2)
    )
    if not np.all(np.isfinite([macaulay, modified, convexity])):
        raise ValueError("cash flows imply non-finite risk measures")
    return {
        "price": price,
        "macaulay_duration": macaulay,
        "modified_duration": modified,
        "convexity": convexity,
        "dv01": dv01(price, modified),
    }


def dv01(price: float, modified_duration: float) -> float:
    """Return the positive price-point magnitude of a one-basis-point move.

    ``price`` and the result use the same currency or quotation unit. For a
    standard positive-duration position, the signed first-order price change
    caused by a parallel ``+1`` basis-point yield shock is ``-dv01``.
    """
    if not np.all(np.isfinite([price, modified_duration])):
        raise ValueError("price and modified_duration must be finite")
    if price <= 0:
        raise ValueError("price must be positive")
    if modified_duration < 0:
        raise ValueError("modified_duration must be non-negative")
    return float(price * modified_duration * 0.0001)


def tranche_loss(
    collateral_loss: float | np.ndarray,
    attachment: float | np.ndarray,
    detachment: float | np.ndarray,
) -> float | np.ndarray:
    """Allocate collateral loss to a tranche with validated boundaries.

    All three arguments must use the same unit, either currency amounts or
    fractions of a common collateral notional. For attachment ``A`` and
    detachment ``D``, the allocated loss is
    ``min(max(collateral_loss - A, 0), D - A)``. NumPy broadcasting is
    supported, so one loss grid can be evaluated across several tranches.
    """
    try:
        losses, attachments, detachments = np.broadcast_arrays(
            np.asarray(collateral_loss, dtype=float),
            np.asarray(attachment, dtype=float),
            np.asarray(detachment, dtype=float),
        )
    except ValueError as exc:
        raise ValueError(
            "collateral_loss, attachment, and detachment must be broadcast-compatible"
        ) from exc
    if not (
        np.all(np.isfinite(losses))
        and np.all(np.isfinite(attachments))
        and np.all(np.isfinite(detachments))
    ):
        raise ValueError("tranche-loss inputs must contain finite values")
    if np.any(losses < 0):
        raise ValueError("collateral_loss must be non-negative")
    if np.any(attachments < 0):
        raise ValueError("attachment must be non-negative")
    if np.any(detachments <= attachments):
        raise ValueError("detachment must be strictly greater than attachment")

    allocated = np.minimum(
        np.maximum(losses - attachments, 0.0),
        detachments - attachments,
    )
    if allocated.ndim == 0:
        return float(allocated)
    return allocated


def duration_convexity_price(
    price: float,
    modified_duration: float,
    convexity: float,
    yield_shock: float,
) -> float:
    """Second-order Taylor approximation to the shocked bond price."""
    relative_change = -modified_duration * yield_shock + 0.5 * convexity * yield_shock**2
    return float(price * (1 + relative_change))


def redington_immunization_check(
    asset_pv: float,
    liability_pv: float,
    asset_duration: float,
    liability_duration: float,
    asset_convexity: float,
    liability_convexity: float,
    tolerance: float = 1e-4,
    *,
    pv_rtol: float = 1e-6,
    pv_atol: float = 0.0,
    duration_atol: float | None = None,
    convexity_margin: float = 0.0,
) -> dict[str, bool]:
    """Evaluate Redington conditions with dimension-aware tolerances.

    ``tolerance`` is retained as the legacy duration tolerance, in years.
    Present-value matching is relative by default, which preserves the result
    under a change of currency unit. A caller may set ``pv_atol`` explicitly in
    the chosen currency unit, accepting that the absolute materiality floor is
    then unit-dependent. Convexity must exceed the liability value by more than
    the non-negative ``convexity_margin``; equality does not pass.
    """
    values = np.asarray(
        [
            asset_pv,
            liability_pv,
            asset_duration,
            liability_duration,
            asset_convexity,
            liability_convexity,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("present values, durations, and convexities must be finite")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    duration_absolute_tolerance = tolerance if duration_atol is None else duration_atol
    tolerances = np.asarray(
        [pv_rtol, pv_atol, duration_absolute_tolerance, convexity_margin],
        dtype=float,
    )
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0):
        raise ValueError("all Redington tolerances and margins must be finite and non-negative")

    pv_scale = max(abs(asset_pv), abs(liability_pv))
    pv_gap = abs(asset_pv - liability_pv)
    return {
        "present_value_matched": bool(pv_gap <= pv_atol + pv_rtol * pv_scale),
        "duration_matched": bool(
            abs(asset_duration - liability_duration) <= duration_absolute_tolerance
        ),
        "convexity_excess": bool(asset_convexity > liability_convexity + convexity_margin),
    }
