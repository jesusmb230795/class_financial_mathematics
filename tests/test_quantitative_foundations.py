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
    coupon_bond_cash_flows,
    discount_factor,
    dv01,
    effective_annual_rate,
    effective_rate_from_continuous,
    future_value,
    nominal_rate_from_effective,
    redington_immunization_check,
    risk_measures_from_cash_flows,
    simple_future_value,
    simple_present_value,
    tranche_loss,
    yield_to_maturity_from_cash_flows,
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
    nelson_siegel_svensson_yield,
    nelson_siegel_yield,
    official_mexican_rate_history,
    par_rate_from_discount_factors,
    rate_panel_pca,
    simulate_cir_full_truncation,
    simulate_vasicek_exact,
    spot_rate_from_discount_factor,
    vasicek_ols_calibration,
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


def test_par_rate_uses_complete_ordered_contractual_grid() -> None:
    discount_factors = pd.Series([0.97, 0.94], index=[0.5, 1.0])

    par_rate = par_rate_from_discount_factors(discount_factors, frequency=2)

    assert par_rate == pytest.approx(0.06282722513089005)
    with pytest.raises(ValueError, match="positive, unique, and ordered"):
        par_rate_from_discount_factors(discount_factors.iloc[::-1], frequency=2)


@pytest.mark.parametrize(
    ("discount_factors", "frequency", "message"),
    [
        (pd.Series([0.97, 0.94], index=[0.5, 1.0]), 2.0, "positive integer"),
        (pd.Series([0.97, 0.94], index=["0.5", "1.0"]), 2, "numeric index"),
        (pd.Series([0.97, 0.94], index=[0.5, 1.5]), 2, "every contractual"),
        (pd.Series([0.97, np.nan], index=[0.5, 1.0]), 2, "finite and positive"),
    ],
)
def test_par_rate_rejects_invalid_contractual_inputs(
    discount_factors: pd.Series,
    frequency: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        par_rate_from_discount_factors(discount_factors, frequency=frequency)


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


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("maturity", np.inf, "must be finite"),
        ("coupon_rate", np.nan, "must be finite"),
        ("price", np.inf, "must be finite"),
        ("coupon_rate", -0.01, "coupon rates must be non-negative"),
    ],
)
def test_bootstrap_rejects_non_finite_or_negative_instrument_terms(
    column: str,
    invalid_value: float,
    message: str,
) -> None:
    instruments = pd.DataFrame(
        {
            "maturity": [0.5, 1.0],
            "coupon_rate": [0.0, 0.0],
            "price": [0.97, 0.94],
        }
    )
    instruments.loc[1, column] = invalid_value

    with pytest.raises(ValueError, match=message):
        bootstrap_coupon_bond_discount_factors(instruments, frequency=2)


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


def test_cash_flow_risk_measures_report_positive_dv01_magnitude() -> None:
    risk = risk_measures_from_cash_flows(
        cash_flows=np.array([100.0]),
        times=np.array([2.0]),
        annual_yield=0.05,
        frequency=1,
    )

    assert risk["price"] == pytest.approx(100 / 1.05**2)
    assert risk["macaulay_duration"] == pytest.approx(2.0)
    assert risk["modified_duration"] == pytest.approx(2 / 1.05)
    assert risk["dv01"] > 0
    assert -risk["dv01"] == pytest.approx(-risk["price"] * risk["modified_duration"] * 0.0001)


def test_coupon_bond_schedule_rejects_an_implicit_stub() -> None:
    with pytest.raises(ValueError, match="maturity must align.*stubs"):
        coupon_bond_cash_flows(
            face_value=100.0,
            coupon_rate=0.08,
            maturity=5.25,
            frequency=2,
        )


def test_coupon_bond_schedule_ends_at_the_declared_maturity() -> None:
    schedule = coupon_bond_cash_flows(
        face_value=100.0,
        coupon_rate=0.08,
        maturity=5.0,
        frequency=2,
    )

    assert len(schedule) == 10
    assert schedule.iloc[-1]["time"] == 5.0
    assert schedule.iloc[-1]["cash_flow"] == pytest.approx(104.0)


def test_ytm_rejects_mixed_sign_cash_flows_with_multiple_roots() -> None:
    with pytest.raises(ValueError, match="non-negative.*unique"):
        yield_to_maturity_from_cash_flows(
            price=100.0,
            cash_flows=np.array([230.0, -132.0]),
            times=np.array([1.0, 2.0]),
            compounding=1,
        )


def test_ytm_requires_a_positive_payment_after_settlement() -> None:
    with pytest.raises(ValueError, match="after settlement"):
        yield_to_maturity_from_cash_flows(
            price=100.0,
            cash_flows=np.array([100.0]),
            times=np.array([0.0]),
            compounding=1,
        )


@pytest.mark.parametrize(
    ("cash_flows", "times", "annual_yield", "frequency", "message"),
    [
        (np.array([1.0, 2.0]), np.array([1.0]), 0.05, 1, "equal length"),
        (np.array([1.0]), np.array([np.inf]), 0.05, 1, "finite"),
        (np.array([1.0]), np.array([-1.0]), 0.05, 1, "non-negative"),
        (np.array([1.0]), np.array([1.0]), 0.05, 0, "positive integer"),
        (np.array([1.0]), np.array([1.0]), -1.0, 1, "compounding domain"),
        (np.array([-1.0]), np.array([1.0]), 0.05, 1, "cash_flows must be non-negative"),
        (np.array([0.0]), np.array([1.0]), 0.05, 1, "at least one positive"),
    ],
)
def test_cash_flow_risk_measures_reject_invalid_inputs(
    cash_flows: np.ndarray,
    times: np.ndarray,
    annual_yield: float,
    frequency: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        risk_measures_from_cash_flows(cash_flows, times, annual_yield, frequency)


def test_dv01_rejects_non_positive_price_or_negative_duration() -> None:
    with pytest.raises(ValueError, match="price must be positive"):
        dv01(0.0, 4.0)
    with pytest.raises(ValueError, match="non-negative"):
        dv01(100.0, -4.0)
    with pytest.raises(ValueError, match="finite"):
        dv01(100.0, np.nan)


def test_tranche_loss_is_capped_between_attachment_and_detachment() -> None:
    collateral_losses = np.array([[0.02], [0.07], [0.15]])
    attachments = np.array([0.0, 0.04, 0.10])
    detachments = np.array([0.04, 0.10, 1.00])

    allocated = tranche_loss(collateral_losses, attachments, detachments)

    expected = np.array(
        [
            [0.02, 0.00, 0.00],
            [0.04, 0.03, 0.00],
            [0.04, 0.06, 0.05],
        ]
    )
    np.testing.assert_allclose(allocated, expected)
    assert tranche_loss(0.07, 0.04, 0.10) == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("loss", "attachment", "detachment", "message"),
    [
        (-0.01, 0.0, 0.1, "collateral_loss"),
        (0.01, -0.1, 0.1, "attachment"),
        (0.01, 0.1, 0.1, "strictly greater"),
        (np.nan, 0.0, 0.1, "finite"),
    ],
)
def test_tranche_loss_rejects_invalid_boundaries(
    loss: float,
    attachment: float,
    detachment: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tranche_loss(loss, attachment, detachment)


def test_redington_present_value_match_is_invariant_to_currency_scale() -> None:
    common = {
        "asset_duration": 5.00005,
        "liability_duration": 5.0,
        "asset_convexity": 28.0,
        "liability_convexity": 24.0,
    }
    unscaled = redington_immunization_check(
        asset_pv=100.00005,
        liability_pv=100.0,
        **common,
    )
    scaled = redington_immunization_check(
        asset_pv=100.00005 * 1_000_000,
        liability_pv=100.0 * 1_000_000,
        **common,
    )

    assert (
        unscaled
        == scaled
        == {
            "present_value_matched": True,
            "duration_matched": True,
            "convexity_excess": True,
        }
    )


def test_redington_default_pv_threshold_is_scale_invariant() -> None:
    common = {
        "asset_duration": 5.0,
        "liability_duration": 5.0,
        "asset_convexity": 28.0,
        "liability_convexity": 24.0,
    }
    unscaled = redington_immunization_check(
        asset_pv=100.00015,
        liability_pv=100.0,
        **common,
    )
    scaled = redington_immunization_check(
        asset_pv=100.00015 * 1_000_000,
        liability_pv=100.0 * 1_000_000,
        **common,
    )

    assert unscaled["present_value_matched"] is False
    assert scaled["present_value_matched"] is False


def test_redington_convexity_margin_remains_strict() -> None:
    result = redington_immunization_check(
        asset_pv=100.0,
        liability_pv=100.0,
        asset_duration=5.0,
        liability_duration=5.0,
        asset_convexity=25.0,
        liability_convexity=24.0,
        convexity_margin=1.0,
    )

    assert result["convexity_excess"] is False


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


def test_official_rate_history_preserves_provider_grid_without_implicit_fill() -> None:
    provider = official_mexican_rate_history(
        start="2018-01-01",
        end="2018-01-19",
        frequency="provider",
    )
    weekly = official_mexican_rate_history(
        start="2018-01-01",
        end="2018-01-19",
        frequency="weekly",
    )

    assert pd.isna(provider.loc["2018-01-03", "cetes_28d"])
    assert provider.attrs["alignment"].endswith("no calendar filling")
    assert provider.attrs["series_ids"] == {
        "policy_rate": "SF61745",
        "cetes_28d": "SF60633",
        "tiie_28d": "SF60648",
    }
    assert weekly.notna().all().all()
    assert set(weekly.index.dayofweek) == {4}
    assert weekly.attrs["frequency"] == "weekly"
    assert weekly.attrs["calendar"] == "W-FRI"
    assert weekly.attrs["sample_start"] == "2018-01-05"
    assert weekly.attrs["sample_end"] == "2018-01-19"
    assert weekly.attrs["source_vintage"] == weekly.attrs["retrieved_at"]
    assert weekly.attrs["methodology_break"]["effective_date"] == "2025-01-01"


def test_official_rate_history_requires_a_supported_explicit_frequency() -> None:
    with pytest.raises(ValueError, match="provider.*weekly"):
        official_mexican_rate_history(frequency="daily")


def test_vasicek_calibration_rejects_non_mean_reverting_ar_fit() -> None:
    explosive = pd.Series([0.01, 0.02, 0.04, 0.08, 0.16])

    with pytest.raises(ValueError, match="beta.*strictly between 0 and 1"):
        vasicek_ols_calibration(explosive, dt=1 / 52)


def test_vasicek_calibration_validates_interval_and_sample() -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        vasicek_ols_calibration(pd.Series([0.01, 0.02, 0.03, 0.04]), dt=0)
    with pytest.raises(ValueError, match="at least four"):
        vasicek_ols_calibration(pd.Series([0.01, 0.02, 0.03]), dt=1 / 52)
    with pytest.raises(ValueError, match="finite"):
        vasicek_ols_calibration(pd.Series([0.01, 0.02, np.inf, 0.03]), dt=1 / 52)


def test_vasicek_calibration_checks_datetime_grid_against_dt() -> None:
    weekly_dates = pd.date_range("2026-01-02", periods=5, freq="W-FRI")
    rates = pd.Series([0.0800, 0.0790, 0.0785, 0.0782, 0.0780], index=weekly_dates)

    with pytest.raises(ValueError, match="dt is inconsistent"):
        vasicek_ols_calibration(rates, dt=1 / 252)
    with pytest.raises(ValueError, match="regular interval"):
        vasicek_ols_calibration(rates.drop(weekly_dates[2]), dt=1 / 52)


def test_short_rate_simulations_validate_domains_and_are_deterministic() -> None:
    first = simulate_vasicek_exact(
        r0=0.08,
        kappa=0.6,
        theta=0.06,
        sigma=0.01,
        years=1.0,
        steps_per_year=12,
        paths=3,
        seed=42,
    )
    second = simulate_vasicek_exact(
        r0=0.08,
        kappa=0.6,
        theta=0.06,
        sigma=0.01,
        years=1.0,
        steps_per_year=12,
        paths=3,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match="kappa and years must be positive"):
        simulate_vasicek_exact(0.08, 0.0, 0.06, 0.01, 1.0)
    with pytest.raises(ValueError, match="CIR requires non-negative"):
        simulate_cir_full_truncation(-0.01, 0.6, 0.06, 0.01, 1.0)


def test_cir_full_truncation_preserves_negative_auxiliary_state() -> None:
    r0 = 0.08
    kappa = 0.6
    theta = 0.06
    sigma = 0.2
    years = 3.0
    steps_per_year = 12
    paths = 100
    seed = 7
    dt = 1 / steps_per_year
    steps = int(years * steps_per_year)
    shocks = np.random.default_rng(seed).normal(
        0.0,
        np.sqrt(dt),
        size=(steps, paths),
    )

    reference = np.empty((steps + 1, paths))
    reference[0] = r0
    state = np.full(paths, r0)
    auxiliary_became_negative = False
    clipped_reference = np.empty_like(reference)
    clipped_reference[0] = r0
    for step, shock in enumerate(shocks, start=1):
        state_positive = np.maximum(state, 0.0)
        state = (
            state + kappa * (theta - state_positive) * dt + sigma * np.sqrt(state_positive) * shock
        )
        auxiliary_became_negative |= bool(np.any(state < 0))
        reference[step] = np.maximum(state, 0.0)

        clipped_positive = np.maximum(clipped_reference[step - 1], 0.0)
        clipped_next = (
            clipped_reference[step - 1]
            + kappa * (theta - clipped_positive) * dt
            + sigma * np.sqrt(clipped_positive) * shock
        )
        clipped_reference[step] = np.maximum(clipped_next, 0.0)

    simulated = simulate_cir_full_truncation(
        r0=r0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        years=years,
        steps_per_year=steps_per_year,
        paths=paths,
        seed=seed,
    )

    assert auxiliary_became_negative
    np.testing.assert_allclose(simulated.to_numpy(), reference)
    assert not np.allclose(simulated.to_numpy(), clipped_reference)


def test_nelson_siegel_evaluators_validate_decay_domains() -> None:
    maturities = np.array([0.0, 1.0, 5.0])
    curve = nelson_siegel_yield(maturities, 0.08, -0.02, 0.01, 2.0)

    assert curve[0] == pytest.approx(0.06)
    with pytest.raises(ValueError, match="tau parameters.*positive"):
        nelson_siegel_yield(maturities, 0.08, -0.02, 0.01, 0.0)
    with pytest.raises(ValueError, match="maturities.*non-negative"):
        nelson_siegel_svensson_yield(
            np.array([-1.0, 2.0]),
            0.08,
            -0.02,
            0.01,
            0.005,
            2.0,
            5.0,
        )


def test_promoted_quantitative_lessons_have_citations() -> None:
    sources = [
        *sorted((PROJECT_ROOT / "notebooks" / "course").glob("6.*.py")),
        *sorted((PROJECT_ROOT / "notebooks" / "course").glob("7.*.py")),
        *sorted((PROJECT_ROOT / "notebooks" / "course").glob("9.*.py")),
    ]

    assert len(sources) == 26
    for source in sources:
        text = source.read_text(encoding="utf-8")
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
