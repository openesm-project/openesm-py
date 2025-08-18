# CI/CD Setup for OpenESM Python

This document describes the CI/CD pipeline setup for the openESM Python package.

## Overview

The CI/CD pipeline provides automated testing, quality checks, and publishing for the openESM package. It's designed to be robust yet simple, focusing on the most essential features.

## Components

### 🔄 Continuous Integration (`.github/workflows/ci.yml`)

**Triggers:** Push to main, Pull Requests to main

**Matrix Testing:**
- **Operating Systems:** Ubuntu, Windows, macOS
- **Python Versions:** 3.9, 3.10, 3.11, 3.12, 3.13

**Quality Checks:**
- **Linting:** Ruff for code style and quality
- **Type Checking:** MyPy for type safety (non-blocking)
- **Testing:** Pytest with 98% coverage requirement
- **Security:** Safety and Bandit for dependency and code security

**Coverage Reporting:**
- Uploads coverage to Codecov (Ubuntu + Python 3.11 only)
- Generates HTML coverage reports

###  Release Pipeline (`.github/workflows/release.yml`)

**Triggers:** GitHub release publication

**Features:**
- Builds wheel and source distributions
- Validates packages with twine
- Publishes to both TestPyPI and PyPI
- Uses trusted publishing (no API tokens needed)

###  Dependency Management (`.github/dependabot.yml`)

**Features:**
- Weekly dependency updates (Mondays 9 AM)
- Python packages and GitHub Actions
- Auto-assigns to `@bsiepe`
- Structured commit messages

###  Pre-commit Setup (`.pre-commit-config.yaml`)

**Local Development Hooks:**
- Ruff formatting and linting
- MyPy type checking
- Basic file hygiene (trailing whitespace, etc.)

## Usage

### Development Workflow

```bash
# Setup development environment
make dev

# Run all quality checks locally
make check

# Run tests with coverage
make test-cov

# Format code
make format

# Simulate CI pipeline locally
make ci-local
```

### Release Process

1. **Prepare Release:**
   ```bash
   # Update version in pyproject.toml
   # Ensure all tests pass
   make check
   ```

2. **Create GitHub Release:**
   - Go to GitHub → Releases → New Release
   - Create a new tag (e.g., `v0.1.0`)
   - Add release notes
   - Publish release

3. **Automated Publishing:**
   - CI automatically builds and publishes to PyPI
   - Package becomes available via `pip install openesm`

### Makefile Commands

```bash
make install    # Install package in development mode
make dev        # Setup full development environment
make test       # Run tests
make test-cov   # Run tests with coverage report
make test-fast  # Run tests without coverage
make lint       # Check code style
make format     # Format code
make type-check # Run type checking
make security   # Run security checks
make check      # Run all quality checks
make clean      # Clean build artifacts
make build      # Build package
make ci-local   # Simulate CI pipeline locally
```

## Configuration

### Testing
- **Coverage Target:** 98% (currently achieved)
- **Test Framework:** pytest with requests-mock for HTTP mocking
- **Test Organization:** Class-based structure with comprehensive fixtures

### Code Quality
- **Formatter:** Ruff (88 character line limit)
- **Linter:** Ruff with comprehensive rule set
- **Type Checker:** MyPy (relaxed configuration for initial release)

### Security
- **Dependency Scanning:** Safety for known vulnerabilities
- **Code Security:** Bandit for common security issues
- **Dependency Updates:** Dependabot for automated updates

## GitHub Setup Requirements

### Secrets (Optional)
- `CODECOV_TOKEN`: For coverage reporting (recommended)

### Environments
The release workflow uses GitHub environments for security:
- `pypi`: Production PyPI publishing
- `testpypi`: TestPyPI publishing

### Branch Protection (Recommended)
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Include administrators in restrictions

## Monitoring

### Coverage Reports
- View coverage at: https://codecov.io/gh/bsiepe/openesm-py
- HTML reports generated in `htmlcov/` directory

### Dependencies
- Dependabot creates PRs for updates
- Review and merge dependency updates regularly

### Release Status
- Monitor PyPI releases: https://pypi.org/project/openesm/
- Check GitHub Actions for CI/CD status

## Troubleshooting

### Common Issues
1. **Type Checking Failures:** Currently non-blocking, will be improved in future releases
2. **Coverage Drops:** Add tests to maintain 98% coverage
3. **Dependency Conflicts:** Review Dependabot PRs carefully

### Getting Help
- Check GitHub Actions logs for detailed error information
- Review test output for specific failure details
- Ensure local `make check` passes before pushing
