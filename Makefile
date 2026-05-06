.PHONY: demo build up down logs test lint typecheck clean simulate

demo:
	docker compose up --build

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down -v

logs:
	docker compose logs -f api

test:
	pytest -q

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

simulate:
	python scripts/simulate.py --days 60 --seed 42

smoke:
	bash scripts/smoke.sh
