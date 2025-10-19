"""Tests for openesm.list_datasets module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

# Add src to path to import openesm
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openesm.list_datasets import _process_raw_datasets_list, list_datasets


@pytest.fixture
def sample_raw_datasets_list():
    """Sample raw datasets list from JSON for testing."""
    return {
        "datasets": [
            {
                "dataset_id": "0001",
                "first_author": "Smith",
                "year": 2023,
                "reference_a": "Smith, J. et al. (2023). Test study.",
                "paper_doi": "10.1000/test",
                "zenodo_doi": "10.5072/zenodo.123456",
                "license": "CC BY 4.0",
                "n_participants": 50,
                "n_time_points": 30,
                "topics": "mood, stress",
                "features": [
                    {"name": "mood", "type": "numeric"},
                    {"name": "stress", "type": "numeric"},
                ],
            },
            {
                "dataset_id": "0002",
                "first_author": "Jones",
                "year": 2022,
                "reference_a": "Jones, A. et al. (2022). Another study.",
                "paper_doi": "10.1000/test2",
                "zenodo_doi": "10.5072/zenodo.789012",
                "license": "CC0",
                "n_participants": 100,
                "n_time_points": 60,
                "topics": "anxiety",
                "features": [],  # Empty features
            },
        ]
    }


@pytest.fixture
def empty_raw_datasets_list():
    """Empty raw datasets list for testing."""
    return {"datasets": []}


def test_process_raw_datasets_list_success(sample_raw_datasets_list):
    """Test _process_raw_datasets_list with valid data."""
    result = _process_raw_datasets_list(sample_raw_datasets_list)

    # Should be a polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Should have 2 rows (2 datasets)
    assert result.shape[0] == 2

    # Check key columns exist
    expected_columns = ["dataset_id", "first_author", "year", "n_participants"]
    for col in expected_columns:
        assert col in result.columns

    # Check data content
    assert result["dataset_id"].to_list() == ["0001", "0002"]
    assert result["first_author"].to_list() == ["Smith", "Jones"]
    assert result["year"].to_list() == [2023, 2022]

    # Check features processing - should be string representation
    features = result["features"].to_list()
    assert "DataFrame(2 rows, 2 cols)" in features[0]  # First dataset has 2 features
    assert features[1] is None  # Second dataset has empty features


def test_process_raw_datasets_list_empty(empty_raw_datasets_list):
    """Test _process_raw_datasets_list with empty datasets list."""
    result = _process_raw_datasets_list(empty_raw_datasets_list)

    # Should return empty DataFrame
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 0


@patch("openesm.list_datasets.msg_warn")
def test_process_raw_datasets_list_schema_error(
    mock_msg_warn, sample_raw_datasets_list
):
    """Test _process_raw_datasets_list handles schema errors gracefully."""
    # Add problematic data that might cause schema issues
    sample_raw_datasets_list["datasets"][0]["problematic_field"] = {"complex": "object"}

    # Should still return a DataFrame without errors
    result = _process_raw_datasets_list(sample_raw_datasets_list)

    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


@patch("openesm.list_datasets.download_metadata_from_zenodo")
def test_list_datasets_use_cache(
    mock_download_metadata,
    temp_cache_dir,
    sample_raw_datasets_list,
):
    """Test list_datasets works with caching (simplified test)."""
    # Create a test JSON file
    index_path = temp_cache_dir / "datasets.json"
    with open(index_path, "w") as f:
        json.dump(sample_raw_datasets_list, f)

    # Mock download_metadata_from_zenodo to return our test file
    mock_download_metadata.return_value = index_path

    result = list_datasets()

    # Should call download_metadata_from_zenodo once
    mock_download_metadata.assert_called_once()

    # Should return processed DataFrame
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


@patch("openesm.list_datasets.download_metadata_from_zenodo")
def test_list_datasets_download_fresh(
    mock_download_metadata,
    temp_cache_dir,
    sample_raw_datasets_list,
):
    """Test list_datasets downloads fresh data with cache_hours=0."""
    # Create a test JSON file
    index_path = temp_cache_dir / "datasets.json"
    with open(index_path, "w") as f:
        json.dump(sample_raw_datasets_list, f)

    # Mock download_metadata_from_zenodo to return our test file
    mock_download_metadata.return_value = index_path

    result = list_datasets(cache_hours=0)

    # Should call download_metadata_from_zenodo with cache_hours=0
    mock_download_metadata.assert_called_once_with(
        metadata_version="latest", cache_hours=0
    )

    # Should return processed DataFrame
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


@patch("openesm.list_datasets.download_metadata_from_zenodo")
def test_list_datasets_no_cache_file(
    mock_download_metadata,
    temp_cache_dir,
    sample_raw_datasets_list,
):
    """Test list_datasets downloads when called without existing cache."""
    # Create a test JSON file
    index_path = temp_cache_dir / "datasets.json"
    with open(index_path, "w") as f:
        json.dump(sample_raw_datasets_list, f)

    # Mock download_metadata_from_zenodo to return our test file
    mock_download_metadata.return_value = index_path

    result = list_datasets()

    # Should call download_metadata_from_zenodo once
    mock_download_metadata.assert_called_once()

    # Should return processed DataFrame
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


@patch("openesm.list_datasets.download_metadata_from_zenodo")
def test_list_datasets_force_download(
    mock_download_metadata,
    temp_cache_dir,
    sample_raw_datasets_list,
):
    """Test list_datasets with cache_hours=0 forces download."""
    # Create a test JSON file
    index_path = temp_cache_dir / "datasets.json"
    with open(index_path, "w") as f:
        json.dump(sample_raw_datasets_list, f)

    # Mock download_metadata_from_zenodo to return our test file
    mock_download_metadata.return_value = index_path

    # Force download with cache_hours=0
    result = list_datasets(cache_hours=0)

    # Should call download_metadata_from_zenodo with cache_hours=0
    mock_download_metadata.assert_called_once_with(
        metadata_version="latest", cache_hours=0
    )

    # Should return processed DataFrame
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


def test_process_features_string_conversion():
    """Test features are properly converted to string representations."""
    # Test with different feature types - avoid polars DataFrame due to boolean issues
    test_cases = [
        # List of features
        {"dataset_id": "test1", "features": [{"name": "mood"}, {"name": "stress"}]},
        # Single feature as list
        {"dataset_id": "test2", "features": [{"name": "mood", "type": "numeric"}]},
        # None/empty features
        {"dataset_id": "test3", "features": None},
        # Empty list
        {"dataset_id": "test4", "features": []},
    ]

    raw_data = {"datasets": test_cases}
    result = _process_raw_datasets_list(raw_data)

    # Check feature string representations
    features = result["features"].to_list()

    # Features get converted to DataFrames by process_specific_metadata
    # So we expect DataFrame representations
    assert "DataFrame(2 rows," in features[0]  # 2 features
    assert "DataFrame(1 rows," in features[1]  # 1 feature

    # Third and fourth should be None (empty)
    assert features[2] is None
    assert features[3] is None


@patch("openesm.list_datasets.download_metadata_from_zenodo")
def test_list_datasets_integration_with_custom_cache_hours(
    mock_download_metadata, temp_cache_dir, sample_raw_datasets_list
):
    """Integration test with custom cache_hours parameter."""
    # Create a test JSON file
    index_path = temp_cache_dir / "datasets.json"
    with open(index_path, "w") as f:
        json.dump(sample_raw_datasets_list, f)

    # Mock download_metadata_from_zenodo to return our test file
    mock_download_metadata.return_value = index_path

    # Test with cache_hours=1
    result1 = list_datasets(cache_hours=1)
    mock_download_metadata.assert_called_with(metadata_version="latest", cache_hours=1)

    # Reset mock
    mock_download_metadata.reset_mock()

    # Test with cache_hours=3
    result2 = list_datasets(cache_hours=3)
    mock_download_metadata.assert_called_with(metadata_version="latest", cache_hours=3)

    # Both should return valid DataFrames
    assert isinstance(result1, pl.DataFrame)
    assert isinstance(result2, pl.DataFrame)


def test_process_features_with_string_fallback():
    """Test features processing with string fallback."""
    # This test verifies the str(features) fallback path
    # but the process_specific_metadata function converts it to DataFrame
    # so let's test what actually happens
    test_cases = [
        {
            "dataset_id": "test1",
            # This gets processed by process_specific_metadata
            "features": "simple string",
        }
    ]

    raw_data = {"datasets": test_cases}
    result = _process_raw_datasets_list(raw_data)

    # The string gets converted to a DataFrame by process_specific_metadata
    # so we expect a DataFrame representation
    features = result["features"].to_list()
    assert "DataFrame(" in features[0]


@patch("openesm.list_datasets.msg_warn")
@patch("openesm.list_datasets.pl.DataFrame")
def test_process_raw_datasets_list_polars_error_fallback(mock_dataframe, mock_msg_warn):
    """Test fallback when polars DataFrame creation fails."""
    # Mock polars.DataFrame to raise an exception on first call
    mock_dataframe.side_effect = [
        Exception("Schema error"),  # First call fails
        pl.DataFrame(
            [{"dataset_id": "test", "features": None}]
        ),  # Second call succeeds
    ]

    test_data = {"datasets": [{"dataset_id": "test", "features": None}]}

    result = _process_raw_datasets_list(test_data)

    # Should have warned about schema issue
    mock_msg_warn.assert_called_once()
    assert "Schema issue creating DataFrame" in str(mock_msg_warn.call_args)

    # Should still return a DataFrame (the second mock call)
    assert result is not None


def test_process_datasets_with_inconsistent_schemas():
    """Test processing datasets with different field sets."""
    # Create datasets with different fields to test schema standardization
    test_data = {
        "datasets": [
            {"dataset_id": "0001", "first_author": "Smith", "unique_field_1": "value1"},
            {"dataset_id": "0002", "first_author": "Jones", "unique_field_2": "value2"},
        ]
    }

    result = _process_raw_datasets_list(test_data)

    # Should handle different schemas gracefully
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2

    # All standardized fields should exist
    columns = result.columns
    assert "dataset_id" in columns
    assert "first_author" in columns


class MockDataFrameWithShape:
    """Mock object with shape attribute to test DataFrame path."""

    def __init__(self, rows, cols):
        self.shape = (rows, cols)


@patch("openesm.list_datasets.process_specific_metadata")
def test_features_shape_handling(mock_process_metadata):
    """Test the shape attribute handling for features."""
    # Mock process_specific_metadata to return a mock DataFrame-like object
    mock_df = MockDataFrameWithShape(3, 4)
    mock_process_metadata.return_value = {"dataset_id": "test", "features": mock_df}

    test_data = {"datasets": [{"dataset_id": "test"}]}
    result = _process_raw_datasets_list(test_data)

    # Should format the mock DataFrame shape
    features = result["features"].to_list()
    assert features[0] == "DataFrame(3 rows, 4 cols)"
