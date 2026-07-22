UV_CACHE_DIR ?= .uv-cache
UV_TOOL_DIR ?= .uv-tools
UV_TOOL_BIN_DIR ?= .uv-tools/bin
PYTHONPATH ?= $(CURDIR)
JUPYTER_CONFIG_DIR ?= .jupyter-config
JUPYTER_DATA_DIR ?= .jupyter-config/data
JUPYTER_RUNTIME_DIR ?= .jupyter-config/runtime
JUPYTERLAB_SETTINGS_DIR ?= .jupyter-config/lab/user-settings
JUPYTERLAB_WORKSPACES_DIR ?= .jupyter-config/lab/workspaces
IPYTHONDIR ?= .jupyter-config/ipython
MPLCONFIGDIR ?= $(CURDIR)/.matplotlib-cache
NOTEBOOK_CHECK_OUTPUT ?= /private/tmp/class_financial_mathematics_notebooks
NOTEBOOK_EXECUTION_REPORT ?= _build/reports/notebook-execution.json
PRE_COMMIT_HOME ?= .pre-commit-cache
BOOK_CONFIG ?= _config.outputs.yml
BOOK_STATIC_CONFIG ?= _config.yml
BOOK_OUTPUTS_CONFIG ?= $(BOOK_CONFIG)
NOTEBOOK_SOURCE_FILES = $(shell python3 scripts/notebook_manifest.py paths)
PUBLISHED_EXECUTABLE_NOTEBOOKS = $(shell python3 scripts/notebook_manifest.py paths --status published --executable)
CURATED_EXECUTABLE_NOTEBOOKS = $(shell python3 scripts/notebook_manifest.py paths --status published,labs --executable)
MODULE ?=
SLUG ?=
TITLE ?=
TYPE ?= lesson
LANGUAGE ?= en
TARGET ?=
BODY_FILE ?=

JUPYTER_ENV = UV_CACHE_DIR=$(UV_CACHE_DIR) PYTHONPATH=$(PYTHONPATH) JUPYTER_CONFIG_DIR=$(JUPYTER_CONFIG_DIR) JUPYTER_DATA_DIR=$(JUPYTER_DATA_DIR) JUPYTER_RUNTIME_DIR=$(JUPYTER_RUNTIME_DIR) JUPYTERLAB_SETTINGS_DIR=$(JUPYTERLAB_SETTINGS_DIR) JUPYTERLAB_WORKSPACES_DIR=$(JUPYTERLAB_WORKSPACES_DIR) IPYTHONDIR=$(IPYTHONDIR) MPLCONFIGDIR=$(MPLCONFIGDIR)
BOOK_ENV = $(JUPYTER_ENV) RUN_INTERACTIVE_WIDGETS=0
NOTEBOOK_CHECK_ENV = $(JUPYTER_ENV) RUN_INTERACTIVE_WIDGETS=0
PRE_COMMIT_ENV = $(JUPYTER_ENV) PRE_COMMIT_HOME=$(PRE_COMMIT_HOME)

.PHONY: bootstrap-jupyterlab book book-static book-strict book-v2-evaluate book-with-outputs check-book-content check-book-links check-curated-notebooks check-jupytext check-lock check-notebook-manifest check-notebook-sources check-published-notebooks clean-book clean-book-all content-status install-pre-commit lab lab-author lab-reset lab-student lint new-content pre-commit prepare-notebooks publish-check sync sync-jupytext test visual-assets visual-assets-concept-maps visual-assets-exchange-chart visual-assets-sync visual-assets-validate

sync:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync

lab: lab-student

bootstrap-jupyterlab:
	$(JUPYTER_ENV) uv run python scripts/bootstrap_jupyterlab.py

lab-student: bootstrap-jupyterlab
	$(JUPYTER_ENV) uv run jupyter lab --ServerApp.root_dir=$(CURDIR) --LabApp.default_url="/lab/workspaces/student/tree/START_HERE.md"

lab-author: bootstrap-jupyterlab
	$(JUPYTER_ENV) uv run jupyter lab --ServerApp.root_dir=$(CURDIR) --LabApp.default_url="/lab/workspaces/author/tree/content/README.md"

lab-reset:
	$(JUPYTER_ENV) uv run python scripts/bootstrap_jupyterlab.py --force

book:
	$(BOOK_ENV) uv run jupyter-book build . --config $(BOOK_CONFIG)

book-static:
	$(BOOK_ENV) uv run jupyter-book build . --config $(BOOK_STATIC_CONFIG)

book-strict:
	$(BOOK_ENV) uv run jupyter-book build . --config $(BOOK_CONFIG) --all -W --keep-going

book-v2-evaluate:
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) UV_TOOL_BIN_DIR=$(UV_TOOL_BIN_DIR) uvx --from jupyter-book==2.1.6 jupyter book build --html --strict --ci

book-with-outputs:
	$(BOOK_ENV) uv run jupyter-book build . --config $(BOOK_OUTPUTS_CONFIG)

install-pre-commit:
	$(PRE_COMMIT_ENV) uv run pre-commit install

pre-commit:
	$(PRE_COMMIT_ENV) uv run pre-commit run --all-files

content-status:
	python3 scripts/book_content.py status

new-content:
	python3 scripts/book_content.py new --module "$(MODULE)" --slug "$(SLUG)" --title "$(TITLE)" --type "$(TYPE)" --language "$(LANGUAGE)" $(if $(strip $(TARGET)),--target "$(TARGET)") $(if $(strip $(BODY_FILE)),--body-file "$(BODY_FILE)")

check-book-content:
	python3 scripts/book_content.py validate

prepare-notebooks: sync-jupytext

sync-jupytext:
	$(JUPYTER_ENV) uv run python scripts/sync_notebook_pairs.py

check-notebook-manifest:
	python3 scripts/notebook_manifest.py validate

check-notebook-sources: check-notebook-manifest
	$(NOTEBOOK_CHECK_ENV) uv run python scripts/validate_notebooks.py $(NOTEBOOK_SOURCE_FILES)

check-jupytext:
	$(NOTEBOOK_CHECK_ENV) uv run python scripts/validate_notebooks.py $(filter %.ipynb,$(NOTEBOOK_SOURCE_FILES))

check-lock:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv lock --check

lint:
	$(JUPYTER_ENV) uv run ruff check src scripts tests

test:
	$(JUPYTER_ENV) uv run pytest

visual-assets:
	python3 scripts/visual_assets.py status

visual-assets-concept-maps:
	$(JUPYTER_ENV) uv run python scripts/generate_module_concept_maps.py

visual-assets-exchange-chart:
	$(JUPYTER_ENV) uv run python scripts/generate_exchange_market_cap_chart.py

visual-assets-sync:
	python3 scripts/visual_assets.py sync

visual-assets-validate:
	python3 scripts/visual_assets.py validate

check-published-notebooks:
	$(NOTEBOOK_CHECK_ENV) uv run python scripts/execute_notebooks.py --status published --output-dir $(NOTEBOOK_CHECK_OUTPUT) --report $(NOTEBOOK_EXECUTION_REPORT)

check-curated-notebooks:
	$(NOTEBOOK_CHECK_ENV) uv run python scripts/execute_notebooks.py --status published,labs --output-dir $(NOTEBOOK_CHECK_OUTPUT) --report $(NOTEBOOK_EXECUTION_REPORT)

check-book-links:
	$(JUPYTER_ENV) uv run python scripts/check_book_links.py _build/html

publish-check:
	git diff --check
	$(MAKE) check-lock
	$(MAKE) check-book-content
	$(MAKE) visual-assets-validate
	$(MAKE) check-notebook-manifest
	$(MAKE) check-notebook-sources
	$(MAKE) check-jupytext
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) clean-book-all
	$(MAKE) check-published-notebooks
	$(MAKE) book-strict
	touch _build/html/.nojekyll
	$(MAKE) check-book-links

clean-book:
	$(JUPYTER_ENV) uv run jupyter-book clean .

clean-book-all:
	$(JUPYTER_ENV) uv run jupyter-book clean . --all
