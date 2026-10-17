PYTHON ?= python3
PORT ?= 8000
VENV_PYTHON = .venv/bin/python

.PHONY: help setup setup-vision models serve demo test lint check readiness

help:
	@echo 'make setup         Install a lightweight development environment'
	@echo 'make setup-vision  Install YOLO/torch dependencies too'
	@echo 'make models        Prepare YOLO weights (online preparation)'
	@echo 'make demo          Add a synthetic example to the local UI history'
	@echo 'make serve         Start this checkout on localhost (PORT=8001 to change)'
	@echo 'make check         Run synthetic tests and Python lint'
	@echo 'make readiness     Check all components of a running local service'

$(VENV_PYTHON):
	$(PYTHON) -m venv .venv

setup: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install -e '.[dev]'
	$(VENV_PYTHON) scripts/setup.py

setup-vision: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install -e '.[vision,dev]'
	$(VENV_PYTHON) scripts/setup.py

models:
	$(VENV_PYTHON) scripts/prepare_models.py

serve:
	$(VENV_PYTHON) -m forte_guard.cli serve --port $(PORT)

demo:
	$(VENV_PYTHON) scripts/create_demo_job.py

test:
	$(VENV_PYTHON) -m pytest -q

lint:
	$(VENV_PYTHON) -m ruff check src scripts examples tests

check: test lint

readiness:
	$(VENV_PYTHON) scripts/check_service.py --url http://127.0.0.1:$(PORT)
