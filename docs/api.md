# API Reference

Complete reference for all functions and classes in the openESM package.

## Functions

### list_datasets

::: openesm.list_datasets
    options:
      show_root_heading: true
      show_source: false

**Example:**

```python
import openesm

# Get all available datasets
datasets = openesm.list_datasets()
print(datasets)

# Force fresh download of metadata
datasets = openesm.list_datasets(force_download=True)

# Use older metadata version
datasets = openesm.list_datasets(metadata_version="1.0.0")
```

---

### get_dataset

::: openesm.get_dataset
    options:
      show_root_heading: true
      show_source: false

**Examples:**

```python
import openesm

# Download a single dataset
dataset = openesm.get_dataset("0001")

# Download multiple datasets
datasets = openesm.get_dataset(["0001", "0002", "0003"])

# Download specific version
dataset = openesm.get_dataset("0001", version="1.0.0")

# Force re-download (ignore cache)
dataset = openesm.get_dataset("0001", force_download=True)

# Disable caching
dataset = openesm.get_dataset("0001", cache=False)

# Quiet mode (no progress messages)
dataset = openesm.get_dataset("0001", quiet=True)
```

---

## Classes

### OpenESMDataset

::: openesm.OpenESMDataset
    options:
      show_root_heading: true
      show_source: false
      members:
        - cite
        - license
        - notes

**Example:**

```python
import openesm

# Get a dataset
dataset = openesm.get_dataset("0001")

# Access the data (polars DataFrame)
print(dataset.data)
print(dataset.data.head())

# Access metadata
print(dataset.metadata["first_author"])
print(dataset.metadata["n_participants"])

# Get version information
print(dataset.dataset_id)
print(dataset.dataset_version)
print(dataset.metadata_version)

# Use helper methods
print(dataset.cite())
print(dataset.license())
print(dataset.notes())
```

---

### OpenESMDatasetList

::: openesm.OpenESMDatasetList
    options:
      show_root_heading: true
      show_source: false
      members:
        - keys
        - values
        - items

**Example:**

```python
import openesm

# Download multiple datasets
datasets = openesm.get_dataset(["0001", "0002", "0003"])

# Access individual datasets
dataset_1 = datasets["0001"]

# Iterate over dataset IDs
for dataset_id in datasets:
    print(dataset_id)

# Iterate over datasets
for dataset in datasets.values():
    print(dataset.data.shape)

# Iterate over IDs and datasets together
for dataset_id, dataset in datasets.items():
    print(f"{dataset_id}: {len(dataset.data)} rows")

# Get list of IDs
dataset_ids = list(datasets.keys())

# Check how many datasets
print(len(datasets))
```

---

## Utility Functions

The package includes several utility functions for cache management. These are exposed through the `openesm.utils` module but most users won't need them directly.

### Cache management

```python
from openesm.utils import cache_info, clear_cache

# Get information about cached files
info = cache_info()
print(info)

# Clear the cache (interactive prompt)
clear_cache()

# Clear cache without prompt
clear_cache(force=True)
```

---

## Data Structures

### Dataset metadata dictionary

When you access `dataset.metadata`, you get a dictionary with these common fields:

| Field | Type | Description |
|-------|------|-------------|
| `dataset_id` | str | Dataset identifier (e.g., "0001") |
| `first_author` | str | Last name of first author |
| `year` | int | Publication year |
| `reference_a` | str | Primary citation |
| `reference_b` | str | Secondary citation (if any) |
| `paper_doi` | str | DOI of associated paper |
| `zenodo_doi` | str | DOI of dataset on Zenodo |
| `license` | str | Dataset license (e.g., "CC BY 4.0") |
| `n_participants` | int | Number of participants |
| `n_time_points` | int | Number of time points |
| `sampling_scheme` | str | ESM sampling method used |
| `topics` | str | Research topics covered |
| `additional_comments` | str | Extra notes about the dataset |
| `features` | dict | Variable descriptions (if available) |

Not all fields are present in all datasets. Use `dataset.metadata.keys()` to see what's available for a specific dataset.

---

## Type Hints

The package provides full type hints for type checkers like mypy:

```python
from openesm import OpenESMDataset, OpenESMDatasetList
import polars as pl

# Single dataset
dataset: OpenESMDataset = openesm.get_dataset("0001")
data: pl.DataFrame = dataset.data
metadata: dict[str, Any] = dataset.metadata

# Multiple datasets
datasets: OpenESMDatasetList = openesm.get_dataset(["0001", "0002"])

# List of datasets
all_datasets: pl.DataFrame = openesm.list_datasets()
```
