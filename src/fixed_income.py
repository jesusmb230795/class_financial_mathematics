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
    """Solve a flat annual YTM under the declared compounding convention."""
    cash_flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(times, dtype=float)
    if price <= 0:
        raise ValueError("price must be positive")
    present_value(cash_flows, times, 0.0, compounding)
    if lower >= upper:
        raise ValueError("lower must be less than upper")

    def error(rate: float) -> float:
        return present_value(cash_flows, times, rate, compounding=compounding) - price

    return float(brentq(error, lower, upper))


def risk_measures_from_cash_flows(
    cash_flows: np.ndarray,
    times: np.ndarray,
    annual_yield: float,
    frequency: int = 2,
) -> dict[str, float]:
    """Compute price, Macaulay duration, modified duration, convexity, and DV01."""
    cash_flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(times, dtype=float)
    periods = times * frequency
    discount = (1 + annual_yield / frequency) ** (-periods)
    present_values = cash_flows * discount
    price = float(present_values.sum())
    weights = present_values / price
    macaulay = float(np.sum(times * weights))
    modified = macaulay / (1 + annual_yield / frequency)
    convexity = float(
        np.sum(present_values * periods * (periods + 1))
        / (price * (1 + annual_yield / frequency) ** 2 * frequency**2)
    )
    return {
        "price": price,
        "macaulay_duration": macaulay,
        "modified_duration": modified,
        "convexity": convexity,
        "dv01": dv01(price, modified),
    }


def dv01(price: float, modified_duration: float) -> float:
    """Dollar value of a one-basis-point parallel yield move."""
    return float(price * modified_duration * 0.0001)


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
) -> dict[str, bool]:
    """Evaluate the three classical Redington immunization conditions."""
    return {
        "present_value_matched": abs(asset_pv - liability_pv) <= tolerance,
        "duration_matched": abs(asset_duration - liability_duration) <= tolerance,
        "convexity_excess": asset_convexity > liability_convexity,
    }
