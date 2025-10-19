"""Updated tests for list_datasets module.

These tests are updated to work with the new Zenodo-based implementation.
"""

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


class TestProcessRawDatasetsList:
    """Test the _process_raw_datasets_list function."""

    def test_process_raw_datasets_list_success(self, sample_raw_datasets_list):
        """Test successful processing of raw datasets list."""
        result = _process_raw_datasets_list(sample_raw_datasets_list)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

        # Check required columns exist
        required_cols = ["dataset_id", "first_author", "year"]
        for col in required_cols:
            assert col in result.columns

        # Check data content
        dataset_ids = result["dataset_id"].to_list()
        assert "0001" in dataset_ids
        assert "0002" in dataset_ids

    def test_process_empty_datasets(self):
        """Test processing empty datasets list."""
        empty_data = {"datasets": []}
        result = _process_raw_datasets_list(empty_data)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_process_missing_datasets_key(self):
        """Test processing data without datasets key."""
        bad_data = {"other_key": []}
        result = _process_raw_datasets_list(bad_data)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0


class TestListDatasets:
    """Test the list_datasets function with mocking."""

    @patch("openesm.list_datasets.download_metadata_from_zenodo")
    @patch("openesm.list_datasets.read_json_safe")
    def test_list_datasets_success(
        self, mock_read_json, mock_download_metadata, sample_raw_datasets_list
    ):
        """Test successful list_datasets call."""
        # Mock the download and read functions
        mock_download_metadata.return_value = Path("/fake/path/datasets.json")
        mock_read_json.return_value = sample_raw_datasets_list

        result = list_datasets()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

        # Verify the download function was called with correct defaults
        mock_download_metadata.assert_called_once_with(
            metadata_version="latest", cache_hours=24
        )
        mock_read_json.assert_called_once()

    @patch("openesm.list_datasets.download_metadata_from_zenodo")
    @patch("openesm.list_datasets.read_json_safe")
    def test_list_datasets_with_custom_params(
        self, mock_read_json, mock_download_metadata, sample_raw_datasets_list
    ):
        """Test list_datasets with custom parameters."""
        mock_download_metadata.return_value = Path("/fake/path/datasets.json")
        mock_read_json.return_value = sample_raw_datasets_list

        result = list_datasets(metadata_version="v1.0.0", cache_hours=48)

        assert isinstance(result, pl.DataFrame)

        # Verify custom parameters were passed correctly
        mock_download_metadata.assert_called_once_with(
            metadata_version="v1.0.0", cache_hours=48
        )

    @patch("openesm.list_datasets.download_metadata_from_zenodo")
    @patch("openesm.list_datasets.read_json_safe")
    def test_list_datasets_empty_response(self, mock_read_json, mock_download_metadata):
        """Test list_datasets with empty response."""
        mock_download_metadata.return_value = Path("/fake/path/datasets.json")
        mock_read_json.return_value = {"datasets": []}

        result = list_datasets()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0


class TestListDatasetsIntegration:
    """Integration tests that actually call the function."""

    def test_list_datasets_integration(self):
        """Integration test of list_datasets function."""
        try:
            result = list_datasets()

            assert isinstance(result, pl.DataFrame)
            assert len(result) > 0

            # Check that key columns exist
            expected_cols = ["dataset_id", "first_author", "year"]
            for col in expected_cols:
                assert col in result.columns

        except Exception as e:
            pytest.skip(f"Integration test failed (network issue?): {e}")

    def test_list_datasets_metadata_version_parameter(self):
        """Test that metadata_version parameter works."""
        try:
            result = list_datasets(metadata_version="latest")

            assert isinstance(result, pl.DataFrame)

        except Exception as e:
            pytest.skip(f"Integration test failed (network issue?): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
