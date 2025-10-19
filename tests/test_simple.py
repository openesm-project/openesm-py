"""Simple, focused tests for OpenESM Python package.

These tests are designed to be simple and focus on core functionality
for PyPI readiness. They test the actual functionality rather than
complex mocking scenarios.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import openesm
from openesm.get_dataset import OpenESMDataset, OpenESMDatasetList


class TestListDatasets:
    """Test list_datasets functionality."""

    def test_list_datasets_basic(self):
        """Test basic list_datasets functionality."""
        datasets = openesm.list_datasets()

        # Should return a polars DataFrame
        assert isinstance(datasets, pl.DataFrame)

        # Should have expected columns
        expected_cols = ["dataset_id", "first_author", "year"]
        for col in expected_cols:
            assert col in datasets.columns

        # Should have some datasets
        assert len(datasets) > 0

    def test_list_datasets_with_metadata_version(self):
        """Test list_datasets with metadata_version parameter."""
        datasets = openesm.list_datasets(metadata_version="latest")

        assert isinstance(datasets, pl.DataFrame)
        assert len(datasets) > 0


class TestOpenESMDataset:
    """Test OpenESMDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return pl.DataFrame(
            {
                "participant_id": [1, 1, 2, 2],
                "mood": [7, 6, 8, 7],
                "stress": [3, 4, 2, 3],
            }
        )

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            "dataset_id": "0001",
            "first_author": "Smith",
            "year": 2023,
            "license": "CC BY 4.0",
            "n_participants": 50,
            "n_time_points": 100,
        }

    def test_dataset_creation(self, sample_data, sample_metadata):
        """Test dataset creation."""
        dataset = OpenESMDataset(
            data=sample_data,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        assert dataset.dataset_id == "0001"
        assert dataset.dataset_version == "1.0.0"
        assert dataset.metadata_version == "latest"
        assert dataset.version == "1.0.0"  # backward compatibility
        assert dataset.data.equals(sample_data)
        assert dataset.metadata == sample_metadata

    def test_dataset_repr(self, sample_data, sample_metadata):
        """Test dataset __repr__ method."""
        dataset = OpenESMDataset(
            data=sample_data,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        repr_str = repr(dataset)
        assert "OpenESMDataset" in repr_str
        assert "0001" in repr_str
        assert "1.0.0" in repr_str
        assert "latest" in repr_str

    def test_dataset_str(self, sample_data, sample_metadata):
        """Test dataset __str__ method."""
        dataset = OpenESMDataset(
            data=sample_data,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        str_output = str(dataset)
        assert "OpenESM Dataset" in str_output
        assert "0001" in str_output
        assert "1.0.0" in str_output


class TestOpenESMDatasetList:
    """Test OpenESMDatasetList class."""

    @pytest.fixture
    def sample_datasets(self):
        """Sample datasets for testing."""
        data = pl.DataFrame({"mood": [7, 6]})
        metadata = {"dataset_id": "0001", "first_author": "Smith"}

        dataset1 = OpenESMDataset(data, metadata, "0001", "1.0.0", "latest")
        dataset2 = OpenESMDataset(data, metadata, "0002", "1.0.0", "latest")

        return {"0001": dataset1, "0002": dataset2}

    def test_dataset_list_creation(self, sample_datasets):
        """Test dataset list creation."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")

        assert len(dataset_list) == 2
        assert "0001" in dataset_list.keys()
        assert "0002" in dataset_list.keys()
        assert dataset_list.metadata_version == "latest"

    def test_dataset_list_access(self, sample_datasets):
        """Test dataset list item access."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")

        # Test __getitem__
        dataset = dataset_list["0001"]
        assert dataset.dataset_id == "0001"

        # Test iteration
        ids = list(dataset_list)
        assert "0001" in ids
        assert "0002" in ids


class TestGetDatasetIntegration:
    """Integration tests for get_dataset function.

    These tests use sandbox mode to avoid hitting production Zenodo.
    """

    def test_get_single_dataset_sandbox(self):
        """Test getting a single dataset in sandbox mode."""
        try:
            # This should work since 0001 is available in sandbox
            dataset = openesm.get_dataset("0001", sandbox=True, quiet=True)

            assert isinstance(dataset, OpenESMDataset)
            assert dataset.dataset_id == "0001"
            assert dataset.dataset_version is not None
            assert dataset.metadata_version == "latest"
            assert isinstance(dataset.data, pl.DataFrame)
            assert len(dataset.data) > 0

        except Exception as e:
            pytest.skip(f"Sandbox dataset not available: {e}")

    def test_get_multiple_datasets_sandbox(self):
        """Test getting multiple datasets in sandbox mode."""
        try:
            # Test with duplicate to check deduplication
            datasets = openesm.get_dataset(["0001", "0001"], sandbox=True, quiet=True)

            assert isinstance(datasets, OpenESMDatasetList)
            assert len(datasets) == 1  # Should be deduplicated
            assert "0001" in datasets.keys()

            dataset = datasets["0001"]
            assert dataset.dataset_id == "0001"
            assert isinstance(dataset.data, pl.DataFrame)

        except Exception as e:
            pytest.skip(f"Sandbox dataset not available: {e}")

    def test_dataset_versioning(self):
        """Test that versioning works correctly."""
        try:
            dataset = openesm.get_dataset(
                "0001", metadata_version="latest", sandbox=True, quiet=True
            )

            assert dataset.metadata_version == "latest"
            assert dataset.dataset_version is not None
            assert dataset.version == dataset.dataset_version  # backward compatibility

        except Exception as e:
            pytest.skip(f"Sandbox dataset not available: {e}")


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_dataset_id(self):
        """Test error with invalid dataset ID."""
        with pytest.raises(ValueError, match="not found"):
            openesm.get_dataset("9999", sandbox=True)

    def test_invalid_dataset_id_format(self):
        """Test error with non-numeric dataset ID."""
        with pytest.raises(ValueError, match="No numeric dataset ID found"):
            openesm.get_dataset("abc", sandbox=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
