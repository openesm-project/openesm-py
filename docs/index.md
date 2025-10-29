# openESM Python

Welcome! This is a Python package for accessing openly available Experience Sampling Method (ESM) datasets in the [openESM](https://openesmdata.org) database. 

## What is openESM?

openESM is a collection of openly available ESM datasets, all archived on Zenodo with DOIs for citation. This Python package is the companion to the [openESM R package](https://github.com/bsiepe/openesm), giving you the same functionality in Python.

## Why use this package?

- **Simple**: Download datasets with a single line of code
- **Automatic caching**: Downloaded datasets are cached locally, so you don't waste time re-downloading
- **Citation helpers**: Built-in methods make it easy to properly cite the datasets you use

## Quick example

```python
import openesm

# See what's available
datasets = openesm.list_datasets()
print(datasets)

# Download a dataset
dataset = openesm.get_dataset("0001")

# Access the data (as a polars DataFrame)
print(dataset.data)

# Get citation information
print(dataset.cite())
```

## Installation

Install from PyPI using pip:

```bash
pip install openesm
```


## What's included?

The package provides two main functions:

- **`list_datasets()`** - Browse all available datasets with their metadata
- **`get_dataset()`** - Download one or more datasets

When you download a dataset, you get an `OpenESMDataset` object that includes:

- The actual data (in a polars DataFrame)
- Metadata about the study
- Helper methods for citations, license info, and notes

## Need help?

- Check out the [Getting Started](getting-started.md) guide for a walkthrough
- Browse the [API Reference](api.md) for detailed documentation
- Report issues on [GitHub](https://github.com/openesm-project/openesm-py/issues)

## License

This package is licensed under the MIT License. Individual datasets have their own licenses - check `dataset.license()` for details.
