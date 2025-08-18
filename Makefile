.PHONY: install dev test check format clean security lint

install:
	uv sync

dev:
	uv sync --all-extras --dev
	uv run pre-commit install

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=src/openesm --cov-report=html --cov-report=term-missing -v

test-fast:
	uv run pytest --no-cov -x

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy src/openesm 

security:
	uv run safety scan
	uv run bandit -r src/openesm/

check: lint type-check test-cov security
	@echo "All checks passed!"

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	uv build

publish-test:
	uv build
	uv run twine upload --repository testpypi dist/*

publish:
	uv build
	uv run twine upload dist/*

ci-local: clean dev check
	@echo "Local CI simulation complete!"