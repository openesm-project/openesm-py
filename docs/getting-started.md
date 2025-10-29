# Getting Started

This guide will walk you through the basics of using the openESM Python package.

## Installation

First, install the package:

```bash
pip install openesm
```

## Your first dataset

Let's download your first ESM dataset:

```python
import openesm

# Download a dataset by its ID
dataset = openesm.get_dataset("0001")
```

That's it! The dataset is now downloaded and ready to use.

## Exploring your dataset

The `dataset` object contains everything you need:

```python
# Look at the data (polars DataFrame)
print(dataset.data)

# See the first few rows
print(dataset.data.head())

# Check the shape of the data
print(dataset.data.shape)
```

### Getting metadata

Every dataset comes with rich metadata about the study. You can find more about the dataset on our homepage, but here's how to access some key info programmatically:

```python
# Who collected this data when?
print(dataset.metadata["first_author"])
print(dataset.metadata["year"])

# How many participants?
print(dataset.metadata["n_participants"])

# What topics does it cover?
print(dataset.metadata["topics"])
```

### Citation and licensing

Always cite the datasets you use! The package makes this easy:

```python
# Get formatted citation
print(dataset.cite())

# Check the license
print(dataset.license())

# Read any additional notes
print(dataset.notes())
```

## Downloading multiple datasets

Need several datasets at once? Just pass a list:

```python
# Download multiple datasets
datasets = openesm.get_dataset(["0001", "0002", "0003"])

# Access individual datasets
dataset_1 = datasets["0001"]
dataset_2 = datasets["0002"]
```

## Working with the data

The data comes as a polars DataFrame. If you're more comfortable with pandas:

```python
# Convert to pandas
import pandas as pd
df_pandas = dataset.data.to_pandas()

# Now use familiar pandas methods
df_pandas.head()
df_pandas.describe()
```


## Caching

The package automatically caches downloaded datasets. This means:

- First download might take a moment
- Subsequent calls are instant (data loaded from cache)
- Cache works across Python sessions

Want to force a fresh download?

```python
dataset = openesm.get_dataset("0001", force_download=True)
```

## Advanced options

### Specify dataset version

Datasets may have multiple versions on Zenodo:

```python
# Get a specific version
dataset = openesm.get_dataset("0001", version="1.0.0")

# Or always get the latest (default)
dataset = openesm.get_dataset("0001", version="latest")
```

## Next steps

Now you know the basics! Check out the [API Reference](api.md) for complete documentation of all functions and classes.

