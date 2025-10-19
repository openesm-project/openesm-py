# openESM Python Package

[![CI](https://github.com/bsiepe/openesm-py/workflows/CI/badge.svg)](https://github.com/bsiepe/openesm-py/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bsiepe/openesm-py/branch/main/graph/badge.svg)](https://codecov.io/gh/bsiepe/openesm-py)
[![Python versions](https://img.shields.io/pypi/pyversions/openesm.svg)](https://pypi.org/project/openesm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Python interface for accessing openly available Experience Sampling Method (ESM) datasets**

This package provides a Python equivalent to the [openESM R package](https://github.com/bsiepe/openesm), allowing researchers to programmatically access and download ESM datasets from the openESM repository. All datasets are hosted on Zenodo with proper DOIs for citation.

## Features

- **Simple API** - Download datasets with a single function call
- **Automatic Caching** - Datasets cached locally to avoid re-downloads
- **Citation Helper** - Built-in methods for proper dataset citation


## Installation

```bash
pip install openesm
```

## Quick Start

```python
import openesm

# List available datasets
datasets = openesm.list_datasets()
print(datasets)

# Download a specific dataset
dataset = openesm.get_dataset("0001")
print(dataset.data)  # Access the polars DataFrame
print(dataset.cite())  # Get citation information

# Download multiple datasets
datasets = openesm.get_dataset(["0001", "0002"])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
