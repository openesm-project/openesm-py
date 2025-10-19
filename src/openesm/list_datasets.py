"""List available ESM datasets from openESM repository."""

from typing import Any

import polars as pl

from .utils import (
    download_metadata_from_zenodo,
    msg_warn,
    process_specific_metadata,
    read_json_safe,
)


def list_datasets(
    metadata_version: str = "latest", cache_hours: float = 24
) -> pl.DataFrame:
    """List available ESM datasets from openESM repository.

    Retrieves a list of available Experience Sampling Method (ESM) datasets from
    the openESM metadata repository hosted on Zenodo. Returns a polars DataFrame with
    dataset information and metadata that can be used with get_dataset().

    Args:
        metadata_version: Version of metadata catalog to use. Default is "latest".
            Use specific versions like "1.0.0" for reproducible results.
        cache_hours: Number of hours to consider the cached dataset
            index valid. Default is 24. Set to 0 to force fresh download.

    Returns:
        A polars DataFrame with one row per dataset containing:
        - dataset_id: Unique dataset identifier
        - first_author: First author's surname
        - year: Year of publication
        - reference_a: Primary reference
        - reference_b: Secondary reference (if available)
        - paper_doi: Publication DOI
        - zenodo_doi: Zenodo dataset DOI
        - license: Dataset license
        - link_to_data: Direct data link
        - link_to_codebook: Codebook link
        - link_to_code: Analysis code link
        - n_participants: Number of participants
        - n_time_points: Number of time points
        - n_beeps_per_day: Beeps per day information
        - passive_data_available: Passive data availability
        - cross_sectional_available: Cross-sectional data availability
        - topics: Study topics
        - implicit_missingness: Missingness information
        - raw_time_stamp: Timestamp format information
        - sampling_scheme: Sampling scheme details
        - participants: Participant information
        - coding_file: Coding file information
        - additional_comments: Additional notes
        - features: Feature information for each dataset

    Examples:
        >>> # Get list of all available datasets (latest metadata)
        >>> datasets = list_datasets()
        >>>
        >>> # Use specific metadata version for reproducibility
        >>> datasets = list_datasets(metadata_version="1.0.0")
        >>>
        >>> # Force fresh download of index
        >>> fresh_list = list_datasets(cache_hours=0)
        >>>
        >>> # Use dataset IDs with get_dataset()
        >>> # dataset = get_dataset(datasets['dataset_id'][0])
    """
    # download metadata from Zenodo (handles caching internally)
    datasets_json_path = download_metadata_from_zenodo(
        metadata_version=metadata_version, cache_hours=cache_hours
    )

    # read file and process it
    raw_list = read_json_safe(datasets_json_path)
    return _process_raw_datasets_list(raw_list)


def _process_raw_datasets_list(raw_list: dict[str, Any]) -> pl.DataFrame:
    """Process raw datasets list into a polars DataFrame.

    Helper function to process the raw dict from JSON into a DataFrame.

    Args:
        raw_list: The raw dict parsed from the datasets.json file

    Returns:
        A polars DataFrame with processed dataset metadata
    """
    datasets_list = raw_list.get("datasets", [])

    # process each dataset using the same function used by get_dataset()
    processed_datasets = []
    for dataset in datasets_list:
        processed_metadata = process_specific_metadata(dataset)
        # convert features to string representation to avoid polars issues
        features = processed_metadata.get("features")
        if features is not None:
            if hasattr(features, "shape"):  # polars DataFrame
                processed_metadata["features"] = (
                    f"DataFrame({features.shape[0]} rows, {features.shape[1]} cols)"
                )
            elif isinstance(features, list):
                processed_metadata["features"] = f"List({len(features)} items)"
            else:
                processed_metadata["features"] = str(features)
        else:
            processed_metadata["features"] = None
        processed_datasets.append(processed_metadata)

    # convert to polars DataFrame
    # handle the case where some datasets might have missing fields
    if not processed_datasets:
        # return empty DataFrame with expected schema
        return pl.DataFrame()

    # create DataFrame from list of dicts with explicit schema handling
    try:
        df = pl.DataFrame(processed_datasets)
    except Exception as e:
        # fallback: create DataFrame step by step to handle schema issues
        msg_warn(f"Warning: Schema issue creating DataFrame: {e}")
        # get all unique keys from all datasets
        all_keys: set[str] = set()
        for dataset in processed_datasets:
            all_keys.update(dataset.keys())

        # create a standardized list where all dicts have the same keys
        standardized_datasets = []
        for dataset in processed_datasets:
            standardized = {}
            for key in all_keys:
                standardized[key] = dataset.get(key, None)
            standardized_datasets.append(standardized)

        df = pl.DataFrame(standardized_datasets)

    return df
