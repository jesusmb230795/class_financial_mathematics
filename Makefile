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

JUPYTER_ENV = UV_CACHE_DIR=$(UV_CACHE_DIR) PYTHONPATH=$(PYTHONPATH) JUPYTER_CONFIG_DIR=$(JUPYTER_CONFIG_DIR) JUPYTER_DATA_DIR=$(JUPYTER_DATA_DIR) JUPYTER_RUNTIME_DIR=$(JUPYTER_RUNTIME_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR)
BOOK_ENV = $(JUPYTER_ENV) RUN_INTERACTIVE_WIDGETS=0
NOTEBOOK_CHECK_ENV = $(JUPYTER_ENV) RUN_INTERACTIVE_WIDGETS=0
PRE_COMMIT_ENV = UV_CACHE_DIR=$(UV_CACHE_DIR) PRE_COMMIT_HOME=$(PRE_COMMIT_HOME)

.PHONY: book book-static book-with-outputs check-curated-notebooks clean-book install-pre-commit pre-commit sync lab visual-assets visual-assets-exchange-chart visual-assets-sync visual-assets-validate

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

visual-assets:
	python3 scripts/visual_assets.py status

visual-assets-exchange-chart:
	$(JUPYTER_ENV) uv run python scripts/generate_exchange_market_cap_chart.py

visual-assets-sync:
	python3 scripts/visual_assets.py sync

visual-assets-validate:
	python3 scripts/visual_assets.py validate

check-curated-notebooks:
	mkdir -p $(NOTEBOOK_CHECK_OUTPUT)
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/1.3.data_extraction.ipynb notebooks/class/1.3.data_extraction.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/1.4.EDA_stock_data.ipynb notebooks/class/1.4.EDA_stock_data.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/1.5.EDA_macroeconomic_data.ipynb notebooks/class/1.5.EDA_macroeconomic_data.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/1.7.mexican_market_data_pipeline.ipynb notebooks/class/1.7.mexican_market_data_pipeline.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/1.8.macro_dashboard_banxico_fred.ipynb notebooks/class/1.8.macro_dashboard_banxico_fred.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/1.9.return_explorer_dashboard.ipynb notebooks/class/1.9.return_explorer_dashboard.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/2.6.interactive_volatility_garch_dashboard.ipynb notebooks/class/2.6.interactive_volatility_garch_dashboard.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/3.2.downside_risk_var_methods.ipynb notebooks/class/3.2.downside_risk_var_methods.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/3.3.var_backtesting_and_stress_testing.ipynb notebooks/class/3.3.var_backtesting_and_stress_testing.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/3.4.interactive_var_cvar_simulator.ipynb notebooks/class/3.4.interactive_var_cvar_simulator.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/4.2.analytical_efficient_frontier.ipynb notebooks/class/4.2.analytical_efficient_frontier.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/4.3.robust_portfolio_construction.ipynb notebooks/class/4.3.robust_portfolio_construction.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/4.4.interactive_efficient_frontier_dashboard.ipynb notebooks/class/4.4.interactive_efficient_frontier_dashboard.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/5.3.interactive_bond_sensitivity.ipynb notebooks/class/5.3.interactive_bond_sensitivity.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/5.4.mexican_government_bond_valuation.ipynb notebooks/class/5.4.mexican_government_bond_valuation.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/5.5.ytm_dv01_and_immunization_lab.ipynb notebooks/class/5.5.ytm_dv01_and_immunization_lab.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/6.3.nelson_siegel_curve_fitting.ipynb notebooks/class/6.3.nelson_siegel_curve_fitting.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/6.4.yield_curve_pca_and_scenarios.ipynb notebooks/class/6.4.yield_curve_pca_and_scenarios.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/6.5.short_rate_calibration_lab.ipynb notebooks/class/6.5.short_rate_calibration_lab.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/7.3.interactive_black_scholes_dashboard.ipynb notebooks/class/7.3.interactive_black_scholes_dashboard.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/7.4.linear_derivatives_and_carry.ipynb notebooks/class/7.4.linear_derivatives_and_carry.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/7.5.implied_volatility_and_smiles.ipynb notebooks/class/7.5.implied_volatility_and_smiles.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/7.6.american_and_exotic_options.ipynb notebooks/class/7.6.american_and_exotic_options.md
	$(NOTEBOOK_CHECK_ENV) uv run jupytext --execute --to ipynb --output $(NOTEBOOK_CHECK_OUTPUT)/7.7.stochastic_volatility_heston_lab.ipynb notebooks/class/7.7.stochastic_volatility_heston_lab.md

clean-book:
	$(JUPYTER_ENV) uv run jupyter-book clean .
