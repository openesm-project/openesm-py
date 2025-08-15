.PHONY: install dev test check format clean

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"
	pre-commit install

test:
	uv run pytest

test-cov:
	uv run pytest --cov=openesm --cov-report=html
	open htmlcov/index.html

check:  
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/openesm
# 	uv run pytest

format:  
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -rm -rf {} +

build:  
	uv build

publish-test:  
	uv publish --test-pypi

publish:  
	uv publish