"""Tests for openesm.get_dataset module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# Add src to path to import openesm
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openesm.get_dataset import OpenESMDataset, OpenESMDatasetList, get_dataset


@pytest.fixture
def sample_dataframe():
    """Sample polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "participant_id": [1, 1, 2, 2],
            "timestamp": [
                "2023-01-01T10:00:00",
                "2023-01-01T11:00:00",
                "2023-01-01T10:00:00",
                "2023-01-01T11:00:00",
            ],
            "mood": [7, 6, 8, 7],
            "stress": [3, 4, 2, 3],
        }
    )


@pytest.fixture
def sample_metadata():
    """Sample dataset metadata for testing."""
    return {
        "dataset_id": "0001",
        "first_author": "Smith",
        "year": 2023,
        "reference_a": "Smith, J., et al. (2023). Test study. "
        "Journal of Testing, 1(1), 1-10.",
        "reference_b": "Smith, J., et al. (2023). Additional reference. "
        "Another Journal, 2(1), 11-20.",
        "paper_doi": "10.1000/test",
        "zenodo_doi": "10.5072/zenodo.123456",
        "license": "CC BY 4.0",
        "n_participants": 50,
        "n_time_points": 100,
        "additional_comments": "This is a test dataset; Used for validation purposes",
        "topics": "mood, stress",
    }


@pytest.fixture
def minimal_metadata():
    """Minimal dataset metadata for testing."""
    return {
        "dataset_id": "0002",
        "first_author": "Jones",
        "year": 2022,
    }


class TestOpenESMDataset:
    """Tests for OpenESMDataset class."""

    def test_initialization(self, sample_dataframe, sample_metadata):
        """Test OpenESMDataset initialization."""
        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        assert dataset.data.equals(sample_dataframe)
        assert dataset.metadata == sample_metadata
        assert dataset.dataset_id == "0001"
        assert dataset.dataset_version == "1.0.0"
        assert dataset.metadata_version == "latest"
        assert dataset.version == "1.0.0"  # backward compatibility

    def test_repr(self, sample_dataframe, sample_metadata):
        """Test __repr__ method."""
        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        repr_str = repr(dataset)
        assert "OpenESMDataset" in repr_str
        assert "id='0001'" in repr_str
        assert "dataset_version='1.0.0'" in repr_str
        assert "metadata_version='latest'" in repr_str
        assert f"shape={sample_dataframe.shape}" in repr_str

    @patch("openesm.get_dataset.Console")
    def test_str_method(self, mock_console_class, sample_dataframe, sample_metadata):
        """Test __str__ method with console capture."""
        # Mock console instance and capture
        mock_console = MagicMock()
        mock_capture = MagicMock()
        mock_capture.get.return_value = "Dataset output"
        mock_console.capture.return_value.__enter__.return_value = mock_capture
        mock_console_class.return_value = mock_console

        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = str(dataset)
        assert result == "Dataset output"

        # Verify console was used for formatting
        mock_console.print.assert_called()

    @patch("openesm.get_dataset.Console")
    def test_cite_with_references(
        self, mock_console_class, sample_dataframe, sample_metadata
    ):
        """Test cite() method with references available."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = dataset.cite()

        # Should return both references joined by newlines
        expected = (
            "Smith, J., et al. (2023). Test study. Journal of Testing, 1(1), 1-10.\n\n"
            "Smith, J., et al. (2023). Additional reference. "
            "Another Journal, 2(1), 11-20."
        )
        assert result == expected

        # Should have printed formatted output
        mock_console.print.assert_called()

    @patch("openesm.get_dataset.Console")
    def test_cite_no_references(self, mock_console_class, sample_dataframe):
        """Test cite() method with no references available."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        metadata_no_refs = {"dataset_id": "0001"}
        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=metadata_no_refs,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = dataset.cite()

        assert result == ""
        # Should print "no citation information" message
        mock_console.print.assert_called()

    @patch("openesm.get_dataset.Console")
    def test_cite_empty_references(self, mock_console_class, sample_dataframe):
        """Test cite() method with empty/whitespace references."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        metadata_empty_refs = {
            "dataset_id": "0001",
            "reference_a": "   ",  # whitespace only
            "reference_b": None,  # None value
        }
        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=metadata_empty_refs,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = dataset.cite()

        assert result == ""

    @patch("openesm.get_dataset.Console")
    def test_license_available(
        self, mock_console_class, sample_dataframe, sample_metadata
    ):
        """Test license() method with license information."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = dataset.license()

        assert result == "CC BY 4.0"
        mock_console.print.assert_called()

    @patch("openesm.get_dataset.Console")
    def test_license_not_available(self, mock_console_class, sample_dataframe):
        """Test license() method with no license information."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        metadata_no_license = {"dataset_id": "0001"}
        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=metadata_no_license,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = dataset.license()

        expected = (
            "License information not available. Please check the original publication."
        )
        assert result == expected
        mock_console.print.assert_called()

    @patch("openesm.get_dataset.Console")
    def test_notes_with_data(
        self, mock_console_class, sample_dataframe, sample_metadata
    ):
        """Test notes() method with available data."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = dataset.notes()

        # Should include parsed comments and participant/timepoint info
        expected_lines = [
            "This is a test dataset",
            "Used for validation purposes",
            "Participants: 50",
            "Time points: 100",
        ]

        for line in expected_lines:
            assert line in result

        mock_console.print.assert_called()

    @patch("openesm.get_dataset.Console")
    def test_notes_no_data(self, mock_console_class, sample_dataframe):
        """Test notes() method with no additional data."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        metadata_minimal = {"dataset_id": "0001"}
        dataset = OpenESMDataset(
            data=sample_dataframe,
            metadata=metadata_minimal,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        result = dataset.notes()

        assert result == ""
        mock_console.print.assert_called()


class TestOpenESMDatasetList:
    """Tests for OpenESMDatasetList class."""

    @pytest.fixture
    def sample_datasets(self, sample_dataframe, sample_metadata, minimal_metadata):
        """Create sample datasets for testing."""
        dataset1 = OpenESMDataset(
            data=sample_dataframe,
            metadata=sample_metadata,
            dataset_id="0001",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        dataset2 = OpenESMDataset(
            data=sample_dataframe,
            metadata=minimal_metadata,
            dataset_id="0002",
            dataset_version="1.0.0",
            metadata_version="latest",
        )

        return {"0001": dataset1, "0002": dataset2}

    def test_initialization(self, sample_datasets):
        """Test OpenESMDatasetList initialization."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        assert dataset_list.datasets == sample_datasets

    def test_getitem(self, sample_datasets):
        """Test __getitem__ method."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        assert dataset_list["0001"] == sample_datasets["0001"]

    def test_iter(self, sample_datasets):
        """Test __iter__ method."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        keys = list(dataset_list)
        assert keys == ["0001", "0002"]

    def test_len(self, sample_datasets):
        """Test __len__ method."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        assert len(dataset_list) == 2

    def test_keys(self, sample_datasets):
        """Test keys() method."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        assert list(dataset_list.keys()) == ["0001", "0002"]

    def test_values(self, sample_datasets):
        """Test values() method."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        values = list(dataset_list.values())
        assert len(values) == 2
        assert values[0] == sample_datasets["0001"]

    def test_items(self, sample_datasets):
        """Test items() method."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        items = list(dataset_list.items())
        assert len(items) == 2
        assert items[0] == ("0001", sample_datasets["0001"])

    def test_repr(self, sample_datasets):
        """Test __repr__ method."""
        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        repr_str = repr(dataset_list)
        assert "OpenESMDatasetList" in repr_str
        assert "0001" in repr_str
        assert "0002" in repr_str

    @patch("openesm.get_dataset.Console")
    def test_str_method(self, mock_console_class, sample_datasets):
        """Test __str__ method."""
        mock_console = MagicMock()
        mock_capture = MagicMock()
        mock_capture.get.return_value = "Collection output"
        mock_console.capture.return_value.__enter__.return_value = mock_capture
        mock_console_class.return_value = mock_console

        dataset_list = OpenESMDatasetList(sample_datasets, "latest")
        result = str(dataset_list)

        assert result == "Collection output"
        mock_console.print.assert_called()

    def test_empty_list(self):
        """Test empty dataset list."""
        dataset_list = OpenESMDatasetList({}, "latest")
        assert len(dataset_list) == 0
        assert list(dataset_list.keys()) == []

    @patch("openesm.get_dataset.Console")
    def test_str_method_many_datasets(
        self, mock_console_class, sample_dataframe, sample_metadata
    ):
        """Test __str__ method with more than 5 datasets (truncation)."""
        mock_console = MagicMock()
        mock_capture = MagicMock()
        mock_capture.get.return_value = "Collection output"
        mock_console.capture.return_value.__enter__.return_value = mock_capture
        mock_console_class.return_value = mock_console

        # Create 7 datasets to trigger truncation
        datasets = {}
        for i in range(7):
            dataset_id = f"000{i + 1}"
            metadata = {**sample_metadata, "dataset_id": dataset_id}
            dataset = OpenESMDataset(
                sample_dataframe, metadata, dataset_id, "1.0.0", "latest"
            )
            datasets[dataset_id] = dataset

        dataset_list = OpenESMDatasetList(datasets, "latest")
        result = str(dataset_list)

        assert result == "Collection output"
        # Should show truncation message "... and X more."
        mock_console.print.assert_called()


class TestGetDataset:
    """Tests for get_dataset function and related helpers."""

    @patch("openesm.get_dataset.list_datasets")
    @patch("openesm.get_dataset.download_metadata_from_zenodo")
    @patch("openesm.get_dataset.read_json_safe")
    @patch("openesm.get_dataset.resolve_zenodo_version")
    @patch("openesm.get_dataset.download_from_zenodo")
    @patch("openesm.get_dataset.pl.read_csv")
    @patch("openesm.get_dataset.get_cache_path")
    def test_get_single_dataset_success(
        self,
        mock_get_cache_path,
        mock_read_csv,
        mock_download_zenodo,
        mock_resolve_version,
        mock_read_json,
        mock_download_metadata,
        mock_list_datasets,
        sample_dataframe,
        sample_metadata,
        temp_cache_dir,
    ):
        """Test successful single dataset download."""
        # Mock list_datasets response
        datasets_df = pl.DataFrame(
            {
                "dataset_id": ["0001"],
                "first_author": ["Smith"],
                "zenodo_doi": ["10.5072/zenodo.123456"],
            }
        )
        mock_list_datasets.return_value = datasets_df

        # Mock cache paths
        metadata_path = temp_cache_dir / "metadata.json"
        data_path = temp_cache_dir / "data.tsv"
        mock_get_cache_path.side_effect = [metadata_path, data_path]

        # Mock file existence (both don't exist, so will download)
        metadata_path.touch()  # Create empty file
        data_path.touch()

        # Mock other dependencies
        mock_download_metadata.return_value = metadata_path
        mock_read_json.return_value = {"datasets": [sample_metadata]}
        mock_resolve_version.return_value = "1.0.0"
        mock_read_csv.return_value = sample_dataframe

        # Call function
        result = get_dataset("0001", quiet=True)

        # Assertions
        assert isinstance(result, OpenESMDataset)
        assert result.dataset_id == "0001"
        assert result.version == "1.0.0"
        assert result.data.equals(sample_dataframe)

        # Verify calls
        mock_list_datasets.assert_called_once()
        mock_resolve_version.assert_called_once()

    @patch("openesm.get_dataset.list_datasets")
    def test_get_dataset_not_found(self, mock_list_datasets):
        """Test error when dataset ID not found."""
        # Mock empty dataset list
        datasets_df = pl.DataFrame(
            {
                "dataset_id": ["0002"],  # Different ID
                "first_author": ["Jones"],
            }
        )
        mock_list_datasets.return_value = datasets_df

        with pytest.raises(ValueError, match="Dataset with id '0001' not found"):
            get_dataset("0001")

    @patch("openesm.get_dataset._get_multiple_datasets")
    def test_get_multiple_datasets_calls_helper(self, mock_get_multiple):
        """Test that multiple dataset IDs call the helper function."""
        mock_get_multiple.return_value = MagicMock()

        dataset_ids = ["0001", "0002"]
        get_dataset(dataset_ids, quiet=True)

        mock_get_multiple.assert_called_once_with(
            dataset_ids, "latest", "latest", True, False, False, True
        )

    @patch("openesm.get_dataset.list_datasets")
    @patch("openesm.get_dataset.download_metadata_from_zenodo")
    @patch("openesm.get_dataset.read_json_safe")
    @patch("openesm.get_dataset.resolve_zenodo_version")
    @patch("openesm.get_dataset.download_from_zenodo")
    @patch("openesm.get_dataset.pl.read_csv")
    @patch("openesm.get_dataset.get_cache_path")
    def test_dataset_id_extraction_formats(
        self,
        mock_get_cache_path,
        mock_read_csv,
        mock_download_zenodo,
        mock_resolve_version,
        mock_read_json,
        mock_download_metadata,
        mock_list_datasets,
        sample_dataframe,
        sample_metadata,
        temp_cache_dir,
    ):
        """Test that different dataset ID formats are correctly parsed."""
        # Mock list_datasets response
        datasets_df = pl.DataFrame(
            {
                "dataset_id": ["0001"],
                "first_author": ["Smith"],
                "zenodo_doi": ["10.5072/zenodo.123456"],
            }
        )
        mock_list_datasets.return_value = datasets_df

        # Mock cache paths and other dependencies
        metadata_path = temp_cache_dir / "metadata.json"
        data_path = temp_cache_dir / "data.tsv"
        # Return paths for each get_dataset call (2 paths × 5 test formats = 10 calls)
        mock_get_cache_path.side_effect = [metadata_path, data_path] * 5
        metadata_path.touch()
        data_path.touch()
        mock_download_metadata.return_value = metadata_path
        mock_read_json.return_value = {"datasets": [sample_metadata]}
        mock_resolve_version.return_value = "1.0.0"
        mock_read_csv.return_value = sample_dataframe

        # Test different input formats that should all resolve to "0001"
        test_formats = [
            "0001",  # Standard format
            "0001_fried",  # With author suffix
            "dataset_0001",  # With prefix
            "0001_study_2024",  # With multiple suffixes
            " 0001 ",  # With whitespace
        ]

        for test_id in test_formats:
            # Reset mock calls for each test
            mock_list_datasets.reset_mock()

            # Call get_dataset with the test format
            result = get_dataset(test_id, quiet=True)

            # Verify the dataset was found and loaded correctly
            assert isinstance(result, OpenESMDataset)
            assert result.dataset_id == "0001"  # Should always be normalized to "0001"

    @patch("openesm.get_dataset.list_datasets")
    def test_dataset_id_extraction_no_digits(self, mock_list_datasets):
        """Test error when dataset ID contains no digits."""
        datasets_df = pl.DataFrame({"dataset_id": ["0001"], "first_author": ["Smith"]})
        mock_list_datasets.return_value = datasets_df

        # Test input with no digits should raise ValueError
        with pytest.raises(
            ValueError, match="No numeric dataset ID found in 'no_digits_here'"
        ):
            get_dataset("no_digits_here")

    @patch("openesm.get_dataset.list_datasets")
    @patch("openesm.get_dataset.download_metadata_from_zenodo")
    @patch("openesm.get_dataset.read_json_safe")
    @patch("openesm.get_dataset.resolve_zenodo_version")
    @patch("openesm.get_dataset.download_from_zenodo")
    @patch("openesm.get_dataset.pl.read_csv")
    @patch("openesm.get_dataset.get_cache_path")
    @patch("builtins.print")
    def test_get_dataset_prints_output_when_not_quiet(
        self,
        mock_print,
        mock_get_cache_path,
        mock_read_csv,
        mock_download_zenodo,
        mock_resolve_version,
        mock_read_json,
        mock_download_metadata,
        mock_list_datasets,
        sample_dataframe,
        sample_metadata,
        temp_cache_dir,
    ):
        """Test that get_dataset prints dataset info when quiet=False."""
        # Mock list_datasets response
        datasets_df = pl.DataFrame(
            {
                "dataset_id": ["0001"],
                "first_author": ["Smith"],
                "zenodo_doi": ["10.5072/zenodo.123456"],
            }
        )
        mock_list_datasets.return_value = datasets_df

        # Mock cache paths
        metadata_path = temp_cache_dir / "metadata.json"
        data_path = temp_cache_dir / "data.tsv"
        mock_get_cache_path.side_effect = [metadata_path, data_path]

        # Create files so no download needed
        metadata_path.touch()
        data_path.touch()

        # Mock other dependencies
        mock_download_metadata.return_value = metadata_path
        mock_read_json.return_value = {"datasets": [sample_metadata]}
        mock_resolve_version.return_value = "1.0.0"
        mock_read_csv.return_value = sample_dataframe

        # Call function with quiet=False (default)
        get_dataset("0001")

        # Should print the dataset string representation
        mock_print.assert_called()
        printed_arg = mock_print.call_args[0][0]
        assert isinstance(printed_arg, OpenESMDataset)


class TestCSVParameterHandling:
    """Test CSV reading with null_values and infer_schema_length parameters."""

    @patch("openesm.get_dataset.pl.read_csv")
    @patch("openesm.get_dataset.resolve_zenodo_version")
    @patch("openesm.get_dataset.read_json_safe")
    @patch("openesm.get_dataset.download_metadata_from_zenodo")
    @patch("openesm.get_dataset.get_cache_path")
    @patch("openesm.get_dataset.list_datasets")
    def test_read_csv_called_with_null_values(
        self,
        mock_list_datasets,
        mock_get_cache_path,
        mock_download_metadata,
        mock_read_json,
        mock_resolve_version,
        mock_read_csv,
        sample_dataframe,
        sample_metadata,
        temp_cache_dir,
    ):
        """Test that pl.read_csv is called with null_values parameter."""
        # Mock list_datasets response
        datasets_df = pl.DataFrame(
            {
                "dataset_id": ["0001"],
                "first_author": ["Smith"],
                "zenodo_doi": ["10.5072/zenodo.123456"],
            }
        )
        mock_list_datasets.return_value = datasets_df

        # Mock cache paths
        metadata_path = temp_cache_dir / "metadata.json"
        data_path = temp_cache_dir / "data.tsv"
        mock_get_cache_path.side_effect = [metadata_path, data_path]

        # Create files
        metadata_path.touch()
        data_path.touch()

        # Mock other dependencies
        mock_download_metadata.return_value = metadata_path
        mock_read_json.return_value = {"datasets": [sample_metadata]}
        mock_resolve_version.return_value = "1.0.0"
        mock_read_csv.return_value = sample_dataframe

        # Call function
        get_dataset("0001", quiet=True)

        # Verify pl.read_csv was called with correct parameters
        mock_read_csv.assert_called_once()
        call_kwargs = mock_read_csv.call_args[1]
        assert call_kwargs["separator"] == "\t"
        assert "null_values" in call_kwargs
        assert "NA" in call_kwargs["null_values"]
        assert "" in call_kwargs["null_values"]

    @patch("openesm.get_dataset.pl.read_csv")
    @patch("openesm.get_dataset.resolve_zenodo_version")
    @patch("openesm.get_dataset.read_json_safe")
    @patch("openesm.get_dataset.download_metadata_from_zenodo")
    @patch("openesm.get_dataset.get_cache_path")
    @patch("openesm.get_dataset.list_datasets")
    def test_read_csv_called_with_infer_schema_length(
        self,
        mock_list_datasets,
        mock_get_cache_path,
        mock_download_metadata,
        mock_read_json,
        mock_resolve_version,
        mock_read_csv,
        sample_dataframe,
        sample_metadata,
        temp_cache_dir,
    ):
        """Test that pl.read_csv is called with infer_schema_length parameter."""
        # Mock list_datasets response
        datasets_df = pl.DataFrame(
            {
                "dataset_id": ["0001"],
                "first_author": ["Smith"],
                "zenodo_doi": ["10.5072/zenodo.123456"],
            }
        )
        mock_list_datasets.return_value = datasets_df

        # Mock cache paths
        metadata_path = temp_cache_dir / "metadata.json"
        data_path = temp_cache_dir / "data.tsv"
        mock_get_cache_path.side_effect = [metadata_path, data_path]

        # Create files
        metadata_path.touch()
        data_path.touch()

        # Mock other dependencies
        mock_download_metadata.return_value = metadata_path
        mock_read_json.return_value = {"datasets": [sample_metadata]}
        mock_resolve_version.return_value = "1.0.0"
        mock_read_csv.return_value = sample_dataframe

        # Call function
        get_dataset("0001", quiet=True)

        # Verify pl.read_csv was called with infer_schema_length
        mock_read_csv.assert_called_once()
        call_kwargs = mock_read_csv.call_args[1]
        assert "infer_schema_length" in call_kwargs
        assert call_kwargs["infer_schema_length"] == 10000

    def test_dataframe_with_na_values(self, tmp_path):
        """Test that NA values in TSV are properly handled as null."""
        # Create a TSV file with NA values
        tsv_content = "id\tvalue\tsitting_id\n1\t10\t1\n2\tNA\t2\n3\t30\tNA\n"
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        # Read with the same parameters used in get_dataset
        df = pl.read_csv(
            tsv_file,
            separator="\t",
            null_values=["NA", ""],
            infer_schema_length=10000,
        )

        # Verify NA values are parsed as null
        assert df.shape == (3, 3)
        assert df["value"][1] is None  # Second row value should be null
        assert df["sitting_id"][2] is None  # Third row sitting_id should be null

    def test_dataframe_with_mixed_types(self, tmp_path):
        """Test that columns with mixed int/float are properly inferred."""
        # Create a TSV with integers first, then floats (similar to real data issue)
        tsv_content = "id\tscore\n1\t1\n2\t2\n3\t3\n4\t0.6666666666666666\n"
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        # Read with the same parameters used in get_dataset
        df = pl.read_csv(
            tsv_file,
            separator="\t",
            null_values=["NA", ""],
            infer_schema_length=10000,
        )

        # Verify all values are read correctly and column is float
        assert df.shape == (4, 2)
        assert df["score"].dtype == pl.Float64
        assert df["score"][3] == 0.6666666666666666
