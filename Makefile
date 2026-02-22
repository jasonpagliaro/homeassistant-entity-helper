NPM ?= npm

.PHONY: install dev-install build run lint typecheck test format alembic-upgrade clean

install:
	$(NPM) run bootstrap

dev-install:
	$(NPM) run bootstrap

build:
	$(NPM) run build

run:
	$(NPM) run run

lint:
	$(NPM) run lint

typecheck:
	$(NPM) run typecheck

test:
	$(NPM) run test

format:
	$(NPM) run format

alembic-upgrade:
	$(NPM) run db:migrate

clean:
	$(NPM) run clean
