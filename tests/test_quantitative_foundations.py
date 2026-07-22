from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.fixed_income import (
    BonoM,
    continuous_rate_from_effective,
    discount_factor,
    effective_annual_rate,
    effective_rate_from_continuous,
    future_value,
    nominal_rate_from_effective,
    simple_future_value,
    simple_present_value,
)
from src.portfolio_optimization import (
    factor_sensitivity,
    project_correlation_to_psd,
    project_covariance_to_psd,
    tangency_weights,
    validate_covariance_matrix,
)
from src.term_structure import (
    bootstrap_coupon_bond_discount_factors,
    discount_factor_from_spot_rate,
    forward_rates_from_discount_factors,
    rate_panel_pca,
    spot_rate_from_discount_factor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("compounding", ["annual", "continuous"])
def test_spot_discount_factor_roundtrip_declares_compounding(compounding: str) -> None:
    rate = 0.0875
    maturity = 7.25

    discount = discount_factor_from_spot_rate(rate, maturity, compounding=compounding)
    recovered = spot_rate_from_discount_factor(
        discount,
        maturity,
        compounding=compounding,
    )

    assert recovered == pytest.approx(rate)


def test_forward_rate_uses_compounded_multi_year_interval() -> None:
    discounts = pd.Series(
        [1 / 1.05, 1 / 1.05**3],
        index=[1.0, 3.0],
    )

    forwards = forward_rates_from_discount_factors(discounts)

    assert forwards.iloc[0] == pytest.approx(0.05)


def test_bootstrap_rejects_missing_coupon_discount_factor() -> None:
    sparse = pd.DataFrame(
        {
            "maturity": [0.5, 1.0, 2.0],
            "coupon_rate": [0.0, 0.0, 0.06],
            "price": [98.0, 95.0, 100.0],
        }
    )

    with pytest.raises(ValueError, match="missing discount factor.*1.5"):
        bootstrap_coupon_bond_discount_factors(sparse, frequency=2)


def test_bootstrap_accounts_for_every_coupon_cash_flow() -> None:
    frequency = 2
    flat_spot = 0.08
    maturities = np.arange(0.5, 3.5, 0.5)
    true_discounts = pd.Series(
        [(1 + flat_spot) ** -maturity for maturity in maturities],
        index=maturities,
    )
    coupon_rates = np.array([0.0, 0.0, 0.05, 0.052, 0.054, 0.056])
    prices = []
    for maturity, coupon_rate in zip(maturities, coupon_rates):
        coupon = 100 * coupon_rate / frequency
        payment_times = np.arange(0.5, maturity + 0.25, 0.5)
        cash_flows = np.full(len(payment_times), coupon)
        cash_flows[-1] += 100
        prices.append(float(np.dot(cash_flows, true_discounts.loc[payment_times])))
    instruments = pd.DataFrame(
        {
            "maturity": maturities,
            "coupon_rate": coupon_rates,
            "price": prices,
        }
    )

    bootstrapped = bootstrap_coupon_bond_discount_factors(
        instruments,
        frequency=frequency,
    )

    np.testing.assert_allclose(bootstrapped, true_discounts, atol=1e-12)


def test_tvm_conventions_roundtrip() -> None:
    principal = 1_000.0
    rate = 0.08
    years = 2.5
    for compounding in [1, 4, None]:
        accumulated = future_value(principal, rate, years, compounding)
        assert accumulated * discount_factor(rate, years, compounding) == pytest.approx(principal)

    simple_accumulated = simple_future_value(principal, rate, years)
    assert simple_present_value(simple_accumulated, rate, years) == pytest.approx(principal)


def test_rate_quote_conversions_roundtrip() -> None:
    nominal = 0.12
    effective = effective_annual_rate(nominal, compounding=12)
    assert nominal_rate_from_effective(effective, compounding=12) == pytest.approx(nominal)

    continuous = continuous_rate_from_effective(effective)
    assert effective_rate_from_continuous(continuous) == pytest.approx(effective)


def test_bono_day_count_changes_full_quotation_convention() -> None:
    bond_360 = BonoM(
        coupon_rate=0.075,
        annual_yield=0.088,
        remaining_coupons=10,
        days_since_last_coupon=73,
        day_count=360,
    )
    bond_365 = BonoM(
        coupon_rate=0.075,
        annual_yield=0.088,
        remaining_coupons=10,
        days_since_last_coupon=73,
        day_count=365,
    )

    assert bond_360.coupon_payment != pytest.approx(bond_365.coupon_payment)
    assert bond_360.accrued_interest != pytest.approx(bond_365.accrued_interest)
    assert bond_360.dirty_price != pytest.approx(bond_365.dirty_price)


def test_invalid_correlation_is_projected_before_portfolio_use() -> None:
    labels = ["a", "b", "c"]
    invalid = pd.DataFrame(
        [
            [1.0, 0.95, -0.95],
            [0.95, 1.0, 0.95],
            [-0.95, 0.95, 1.0],
        ],
        index=labels,
        columns=labels,
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance_matrix(invalid)

    projected = project_correlation_to_psd(invalid)
    validate_covariance_matrix(projected)

    assert np.diag(projected) == pytest.approx(np.ones(3))
    assert np.linalg.eigvalsh(projected).min() >= -1e-10


def test_covariance_projection_preserves_variances() -> None:
    covariance = pd.DataFrame(
        [
            [0.04, 0.055, -0.0285],
            [0.055, 0.09, 0.057],
            [-0.0285, 0.057, 0.0225],
        ]
    )

    projected = project_covariance_to_psd(covariance)

    validate_covariance_matrix(projected)
    np.testing.assert_allclose(np.diag(projected), np.diag(covariance))


def test_tangency_weights_are_fully_invested() -> None:
    expected = pd.Series([0.08, 0.12, 0.10], index=["a", "b", "c"])
    covariance = pd.DataFrame(
        np.diag([0.01, 0.04, 0.0225]),
        index=expected.index,
        columns=expected.index,
    )

    weights = tangency_weights(expected, covariance, risk_free_rate=0.04)

    assert weights.sum() == pytest.approx(1.0)


def test_factor_sensitivity_recovers_known_loading() -> None:
    rng = np.random.default_rng(2026)
    factor = pd.Series(rng.normal(0, 0.01, 500))
    asset = 0.0002 + 1.4 * factor + pd.Series(rng.normal(0, 0.001, 500))

    estimate = factor_sensitivity(asset, factor, hac_lags=3)

    assert estimate["sensitivity"] == pytest.approx(1.4, abs=0.02)


def test_rate_panel_pca_uses_statistical_component_names() -> None:
    history = pd.DataFrame(
        {
            "policy_rate": [0.08, 0.081, 0.079, 0.082, 0.083],
            "cetes_28d": [0.079, 0.080, 0.078, 0.081, 0.082],
            "tiie_28d": [0.085, 0.086, 0.084, 0.087, 0.088],
        }
    )

    components, explained = rate_panel_pca(history, n_components=2)

    assert list(components.index) == ["PC1", "PC2"]
    assert set(components.columns) == set(history.columns)
    assert explained["explained_variance_ratio"].sum() <= 1 + 1e-12


def test_promoted_quantitative_lessons_have_specific_checkpoints_and_citations() -> None:
    sources = [
        *sorted((PROJECT_ROOT / "notebooks" / "course").glob("6.*.py")),
        *sorted((PROJECT_ROOT / "notebooks" / "course").glob("7.*.py")),
        *sorted((PROJECT_ROOT / "notebooks" / "course").glob("9.*.py")),
    ]

    assert len(sources) == 26
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "Reproduce one result that demonstrates this objective" not in text
        assert "Change one economically meaningful input" not in text
        assert "{cite}`" in text


def _legacy_functions(*names: str) -> dict[str, object]:
    source = (PROJECT_ROOT / "notebooks" / "legacy" / "modern_portfolio_theory_1.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    namespace: dict[str, object] = {"np": np, "norm": norm}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<legacy>", "exec"), namespace)
    return namespace


def test_legacy_tangency_portfolio_is_fully_invested() -> None:
    function = _legacy_functions("get_tangency_portfolio")["get_tangency_portfolio"]
    covariance = np.array([[0.04, 0.006], [0.006, 0.01]])
    expected_returns = np.array([0.12, 0.08])

    weights = function(covariance, expected_returns, 0.04)

    assert weights.sum() == pytest.approx(1.0)


def test_legacy_portfolio_var_uses_daily_positive_loss_scale() -> None:
    function = _legacy_functions("calculate_portfolio_var")["calculate_portfolio_var"]
    weights = np.array([0.60, 0.40])
    annual_means = np.array([0.10, 0.06])
    annual_covariance = np.array([[0.04, 0.004], [0.004, 0.01]])
    value = 1_000_000

    result = function(
        value,
        weights,
        annual_means,
        annual_covariance,
        0.95,
    )
    annual_mean = weights @ annual_means
    annual_std = np.sqrt(weights @ annual_covariance @ weights)
    expected = max(
        -value * (annual_mean / 252 + norm.ppf(0.05) * annual_std / np.sqrt(252)),
        0.0,
    )

    assert result == pytest.approx(expected)
