PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install dev-install run lint typecheck test format alembic-upgrade

install:
	$(PIP) install -r requirements.txt

dev-install:
	$(PIP) install -r requirements-dev.txt

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	$(PYTHON) -m ruff check app tests

typecheck:
	$(PYTHON) -m mypy app tests

test:
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m ruff check --fix app tests

alembic-upgrade:
	$(PYTHON) -m alembic upgrade head
