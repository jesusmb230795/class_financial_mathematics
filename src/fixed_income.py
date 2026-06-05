"""Fixed-income valuation helpers for classroom notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def discount_factor(rate: float, time: float, compounding: int | None = 1) -> float:
    """Return the discount factor for a rate and maturity in years."""
    if compounding is None:
        return float(np.exp(-rate * time))
    return float((1 + rate / compounding) ** (-compounding * time))


def present_value(cash_flows: np.ndarray, times: np.ndarray, rate: float, compounding: int | None = 1) -> float:
    """Present value of dated cash flows under a flat rate."""
    cash_flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(times, dtype=float)
    factors = np.array([discount_factor(rate, t, compounding) for t in times])
    return float(np.sum(cash_flows * factors))


def future_value(present: float, rate: float, time: float, compounding: int | None = 1) -> float:
    """Future value under discrete or continuous compounding."""
    if compounding is None:
        return float(present * np.exp(rate * time))
    return float(present * (1 + rate / compounding) ** (compounding * time))


def annuity_payment(principal: float, annual_rate: float, years: float, payments_per_year: int = 12) -> float:
    """Level payment for an amortizing loan."""
    periods = int(round(years * payments_per_year))
    period_rate = annual_rate / payments_per_year
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

    @property
    def price(self) -> float:
        denominator = 1 + self.annual_yield * self.days_to_maturity / self.day_count
        return float(self.face_value / denominator)

    @property
    def discount_rate(self) -> float:
        return cetes_discount_rate_from_yield(self.annual_yield, self.days_to_maturity, self.day_count)


def cetes_price(
    annual_yield: float,
    days_to_maturity: int,
    face_value: float = 10.0,
    day_count: int = 360,
) -> float:
    """Price a CETES instrument from its annualized ACT/360 yield."""
    return Cetes(days_to_maturity, annual_yield, face_value, day_count).price


def cetes_discount_rate_from_yield(annual_yield: float, days_to_maturity: int, day_count: int = 360) -> float:
    """Convert CETES return yield into the quoted discount-rate basis."""
    return float(annual_yield / (1 + annual_yield * days_to_maturity / day_count))


def cetes_yield_from_discount_rate(discount_rate: float, days_to_maturity: int, day_count: int = 360) -> float:
    """Convert CETES discount-rate quote into return-yield basis."""
    return float(discount_rate / (1 - discount_rate * days_to_maturity / day_count))


@dataclass(frozen=True)
class BonoM:
    """Simplified Bono M valuation with semiannual 182-day coupons."""

    coupon_rate: float
    annual_yield: float
    remaining_coupons: int
    days_since_last_coupon: int = 0
    face_value: float = 100.0
    coupon_days: int = 182
    day_count: int = 360

    @property
    def coupon_payment(self) -> float:
        return self.face_value * self.coupon_rate * self.coupon_days / self.day_count

    @property
    def fractional_periods(self) -> np.ndarray:
        periods = np.arange(1, self.remaining_coupons + 1, dtype=float)
        fraction_to_next_coupon = (self.coupon_days - self.days_since_last_coupon) / self.coupon_days
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
        return float(self.face_value * self.coupon_rate * self.days_since_last_coupon / self.day_count)

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


def udibono_settlement_mxn(clean_price_udis: float, accrued_interest_udis: float, udi_value: float) -> float:
    """Convert an UDIBONO clean price and accrued interest in UDIS into MXN settlement."""
    return float((clean_price_udis + accrued_interest_udis) * udi_value)


def yield_to_maturity_from_cash_flows(
    price: float,
    cash_flows: np.ndarray,
    times: np.ndarray,
    lower: float = -0.95,
    upper: float = 1.50,
) -> float:
    """Solve yield to maturity from a market price and dated cash flows."""
    cash_flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(times, dtype=float)

    def error(rate: float) -> float:
        return present_value(cash_flows, times, rate, compounding=1) - price

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
