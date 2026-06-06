UV_CACHE_DIR ?= .uv-cache
PYTHONPATH ?= $(CURDIR)
JUPYTER_CONFIG_DIR ?= .jupyter-config
JUPYTER_DATA_DIR ?= .jupyter-config/data
JUPYTER_RUNTIME_DIR ?= .jupyter-config/runtime
MPLCONFIGDIR ?= $(CURDIR)/.matplotlib-cache
NOTEBOOK_CHECK_OUTPUT ?= /private/tmp/class_financial_mathematics_notebooks
PRE_COMMIT_HOME ?= .pre-commit-cache
BOOK_CONFIG ?= _config.outputs.yml
BOOK_STATIC_CONFIG ?= _config.yml
BOOK_OUTPUTS_CONFIG ?= $(BOOK_CONFIG)
NOTEBOOK_SOURCE_FILES = $(shell find notebooks/class \( -name '*.ipynb' -o -name '*.md' \) | sort)
PUBLISHED_EXECUTABLE_NOTEBOOKS = \
	notebooks/class/0.2.environment_validation_lab.ipynb \
	notebooks/class/0.3.initial_repository_setup.ipynb \
	notebooks/class/0.4.classroom_environment_setup.ipynb \
	notebooks/class/1.3.data_extraction.ipynb \
	notebooks/class/1.4.EDA_stock_data.ipynb \
	notebooks/class/1.5.EDA_macroeconomic_data.ipynb \
	notebooks/class/1.6.market_data_quality_framework.ipynb \
	notebooks/class/1.7.mexican_market_data_pipeline.ipynb \
	notebooks/class/1.8.macro_dashboard_banxico_fred.ipynb \
	notebooks/class/1.9.return_explorer_dashboard.ipynb \
	notebooks/class/1.10.market_analysis_communication_case.ipynb \
	notebooks/class/2.1.time_series_1.ipynb \
	notebooks/class/2.2.time_series_2.ipynb \
	notebooks/class/2.3.time_series_diagnostics_and_volatility_extensions.ipynb \
	notebooks/class/2.4.arima_diagnostic_workflow.ipynb \
	notebooks/class/2.5.garch_volatility_risk_workflow.ipynb \
	notebooks/class/2.6.interactive_volatility_garch_dashboard.ipynb
CURATED_EXECUTABLE_NOTEBOOKS = $(PUBLISHED_EXECUTABLE_NOTEBOOKS) \
	notebooks/class/3.2.downside_risk_var_methods.ipynb \
	notebooks/class/3.3.var_backtesting_and_stress_testing.ipynb \
	notebooks/class/3.4.interactive_var_cvar_simulator.ipynb \
	notebooks/class/4.2.analytical_efficient_frontier.ipynb \
	notebooks/class/4.3.robust_portfolio_construction.ipynb \
	notebooks/class/4.4.interactive_efficient_frontier_dashboard.ipynb \
	notebooks/class/5.1.time_value_of_money.ipynb \
	notebooks/class/5.2.bond_pricing_duration_convexity.ipynb \
	notebooks/class/5.3.interactive_bond_sensitivity.ipynb \
	notebooks/class/5.4.mexican_government_bond_valuation.ipynb \
	notebooks/class/5.5.ytm_dv01_and_immunization_lab.ipynb \
	notebooks/class/6.1.yield_curve_bootstrapping.ipynb \
	notebooks/class/6.2.short_rate_models.ipynb \
	notebooks/class/6.3.nelson_siegel_curve_fitting.ipynb \
	notebooks/class/6.4.yield_curve_pca_and_scenarios.ipynb \
	notebooks/class/6.5.short_rate_calibration_lab.ipynb \
	notebooks/class/7.1.options_black_scholes.ipynb \
	notebooks/class/7.2.binomial_and_monte_carlo.ipynb \
	notebooks/class/7.3.interactive_black_scholes_dashboard.ipynb \
	notebooks/class/7.4.linear_derivatives_and_carry.ipynb \
	notebooks/class/7.5.implied_volatility_and_smiles.ipynb \
	notebooks/class/7.6.american_and_exotic_options.ipynb \
	notebooks/class/7.7.stochastic_volatility_heston_lab.ipynb

JUPYTER_ENV = UV_CACHE_DIR=$(UV_CACHE_DIR) PYTHONPATH=$(PYTHONPATH) JUPYTER_CONFIG_DIR=$(JUPYTER_CONFIG_DIR) JUPYTER_DATA_DIR=$(JUPYTER_DATA_DIR) JUPYTER_RUNTIME_DIR=$(JUPYTER_RUNTIME_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)
BOOK_ENV = $(JUPYTER_ENV) RUN_INTERACTIVE_WIDGETS=0
NOTEBOOK_CHECK_ENV = $(JUPYTER_ENV) RUN_INTERACTIVE_WIDGETS=0
PRE_COMMIT_ENV = $(JUPYTER_ENV) PRE_COMMIT_HOME=$(PRE_COMMIT_HOME)

.PHONY: book book-static book-with-outputs check-book-links check-curated-notebooks check-notebook-sources check-published-notebooks clean-book clean-book-all install-pre-commit pre-commit publish-check sync lab visual-assets visual-assets-exchange-chart visual-assets-sync visual-assets-validate

sync:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync

lab:
	$(JUPYTER_ENV) uv run jupyter lab

book:
	$(BOOK_ENV) uv run jupyter-book build . --config $(BOOK_CONFIG)

book-static:
	$(BOOK_ENV) uv run jupyter-book build . --config $(BOOK_STATIC_CONFIG)

book-with-outputs:
	$(BOOK_ENV) uv run jupyter-book build . --config $(BOOK_OUTPUTS_CONFIG)

install-pre-commit:
	$(PRE_COMMIT_ENV) uv run pre-commit install

pre-commit:
	$(PRE_COMMIT_ENV) uv run pre-commit run --all-files

check-notebook-sources:
	$(NOTEBOOK_CHECK_ENV) uv run python scripts/validate_notebooks.py $(NOTEBOOK_SOURCE_FILES)

visual-assets:
	python3 scripts/visual_assets.py status

visual-assets-exchange-chart:
	$(JUPYTER_ENV) uv run python scripts/generate_exchange_market_cap_chart.py

visual-assets-sync:
	python3 scripts/visual_assets.py sync

visual-assets-validate:
	python3 scripts/visual_assets.py validate

check-published-notebooks:
	mkdir -p $(NOTEBOOK_CHECK_OUTPUT)
	$(NOTEBOOK_CHECK_ENV) uv run jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=python3 --to notebook --output-dir $(NOTEBOOK_CHECK_OUTPUT) $(PUBLISHED_EXECUTABLE_NOTEBOOKS)

check-curated-notebooks:
	mkdir -p $(NOTEBOOK_CHECK_OUTPUT)
	$(NOTEBOOK_CHECK_ENV) uv run jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=python3 --to notebook --output-dir $(NOTEBOOK_CHECK_OUTPUT) $(CURATED_EXECUTABLE_NOTEBOOKS)

check-book-links:
	$(JUPYTER_ENV) uv run python scripts/check_book_links.py _build/html

publish-check:
	git diff --check
	$(MAKE) visual-assets-validate
	$(MAKE) check-notebook-sources
	$(MAKE) book
	$(MAKE) check-book-links

clean-book:
	$(JUPYTER_ENV) uv run jupyter-book clean .

clean-book-all:
	$(JUPYTER_ENV) uv run jupyter-book clean . --all
