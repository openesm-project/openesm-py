"""Tests for openesm.utils module."""

import json
import sys
import unittest.mock
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import requests_mock

# Add src to path to import openesm
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openesm.utils import (
    _is_interactive,
    _is_quiet,
    cache_info,
    clear_cache,
    download_with_progress,
    get_cache_dir,
    get_cache_path,
    get_data_dir,
    get_metadata_dir,
    msg_info,
    msg_success,
    msg_warn,
    process_specific_metadata,
    read_json_safe,
)


@patch("openesm.utils.user_cache_dir")
def test_get_cache_dir_no_type(mock_user_cache_dir, temp_cache_dir):
    """Test get_cache_dir without type parameter."""
    mock_user_cache_dir.return_value = str(temp_cache_dir)

    result = get_cache_dir()

    assert result == temp_cache_dir
    assert result.exists()


@patch("openesm.utils.get_cache_dir")
def test_get_metadata_dir(mock_get_cache_dir):
    """Test get_metadata_dir function."""
    mock_get_cache_dir.return_value = Path("/cache/metadata")

    result = get_metadata_dir()

    assert result == Path("/cache/metadata")
    mock_get_cache_dir.assert_called_once_with(type_="metadata")


@patch("openesm.utils.os.environ.get", return_value="true")
def test_is_quiet_true(mock_env_get):
    """Test _is_quiet returns True when OPENESM_QUIET is set."""
    assert _is_quiet() is True


@patch("openesm.utils.os.environ.get", return_value="false")
def test_is_quiet_false(mock_env_get):
    """Test _is_quiet returns False when OPENESM_QUIET is unset."""
    assert _is_quiet() is False


@patch("openesm.utils._is_quiet", return_value=False)
@patch("openesm.utils.console")
def test_msg_info(mock_console, mock_is_quiet):
    """Test msg_info function."""
    msg_info("Test message")
    mock_console.print.assert_called_once_with("[blue]ℹ[/blue] Test message")


def test_read_json_safe_success(temp_cache_dir):
    """Test read_json_safe with valid JSON."""
    json_file = temp_cache_dir / "test.json"
    test_data = {"key": "value", "number": 42}

    with open(json_file, "w") as f:
        json.dump(test_data, f)

    result = read_json_safe(json_file)
    assert result == test_data


def test_read_json_safe_invalid_json(temp_cache_dir):
    """Test read_json_safe with invalid JSON."""
    json_file = temp_cache_dir / "invalid.json"
    json_file.write_text("invalid json content")

    with pytest.raises(ValueError, match="Failed to read JSON file"):
        read_json_safe(json_file)


def test_process_complete_metadata(sample_metadata):
    """Test processing complete metadata."""
    result = process_specific_metadata(sample_metadata)

    assert result["dataset_id"] == "0001"
    assert result["first_author"] == "Smith"
    assert result["year"] == 2023
    assert result["n_participants"] == 100
    # Check that features field exists (don't test exact type due to polars complexity)
    assert "features" in result


def test_process_minimal_metadata():
    """Test processing minimal metadata with missing fields."""
    minimal_meta = {"dataset_id": "0002", "first_author": "Jones"}

    result = process_specific_metadata(minimal_meta)

    assert result["dataset_id"] == "0002"
    assert result["first_author"] == "Jones"
    assert result["year"] is None
    assert result["n_participants"] is None


@patch("openesm.utils.get_cache_dir")
def test_get_data_dir(mock_get_cache_dir):
    """Test get_data_dir function."""
    mock_get_cache_dir.return_value = Path("/cache/data")

    result = get_data_dir()

    assert result == Path("/cache/data")
    mock_get_cache_dir.assert_called_once_with(type_="data")


@patch("openesm.utils._is_quiet", return_value=False)
@patch("openesm.utils.console")
def test_msg_warn(mock_console, mock_is_quiet):
    """Test msg_warn function."""
    msg_warn("Warning message")
    mock_console.print.assert_called_once_with("[yellow]⚠[/yellow] Warning message")


@patch("openesm.utils._is_quiet", return_value=False)
@patch("openesm.utils.console")
def test_msg_success(mock_console, mock_is_quiet):
    """Test msg_success function."""
    msg_success("Success message")
    mock_console.print.assert_called_once_with("[green]✓[/green] Success message")


@patch("openesm.utils._is_quiet", return_value=True)
@patch("openesm.utils.console")
def test_msg_info_quiet_mode(mock_console, mock_is_quiet):
    """Test msg_info when quiet mode is enabled."""
    msg_info("Test message")
    mock_console.print.assert_not_called()


@patch("openesm.utils.os.isatty", return_value=True)
@patch("openesm.utils.os")
def test_is_interactive_true(mock_os, mock_isatty):
    """Test _is_interactive returns True when terminal is interactive."""
    mock_os.isatty.return_value = True

    result = _is_interactive()
    assert result is True


@patch("openesm.utils.os.isatty", return_value=False)
@patch("openesm.utils.os")
def test_is_interactive_false(mock_os, mock_isatty):
    """Test _is_interactive returns False when not interactive."""
    mock_os.isatty.return_value = False

    result = _is_interactive()
    assert result is False


@patch("openesm.utils.user_cache_dir")
def test_get_cache_dir_with_type(mock_user_cache_dir, temp_cache_dir):
    """Test get_cache_dir with type parameter."""
    mock_user_cache_dir.return_value = str(temp_cache_dir)

    result = get_cache_dir(type_="metadata")

    expected = temp_cache_dir / "metadata"
    assert result == expected
    assert result.exists()


@patch("openesm.utils.get_metadata_dir")
@patch("openesm.utils.get_data_dir")
def test_get_cache_path_metadata(
    mock_get_data_dir, mock_get_metadata_dir, temp_cache_dir
):
    """Test get_cache_path for metadata type."""
    metadata_dir = temp_cache_dir / "metadata"
    mock_get_metadata_dir.return_value = metadata_dir

    result = get_cache_path("dataset1", "v1.0", "file.json", "metadata")

    expected = metadata_dir / "dataset1" / "v1.0" / "file.json"
    assert result == expected
    assert result.parent.exists()  # Directory should be created
    mock_get_metadata_dir.assert_called_once()
    mock_get_data_dir.assert_not_called()


@patch("openesm.utils.get_metadata_dir")
@patch("openesm.utils.get_data_dir")
def test_get_cache_path_data(mock_get_data_dir, mock_get_metadata_dir, temp_cache_dir):
    """Test get_cache_path for data type."""
    data_dir = temp_cache_dir / "data"
    mock_get_data_dir.return_value = data_dir

    result = get_cache_path("dataset1", "v1.0", "file.tsv", "data")

    expected = data_dir / "dataset1" / "v1.0" / "file.tsv"
    assert result == expected
    assert result.parent.exists()  # Directory should be created
    mock_get_data_dir.assert_called_once()
    mock_get_metadata_dir.assert_not_called()


def test_get_cache_path_invalid_type():
    """Test get_cache_path raises error for invalid type."""
    with pytest.raises(ValueError, match="type_ must be 'metadata' or 'data'"):
        get_cache_path("dataset1", "v1.0", "file.json", "invalid")


@patch("openesm.utils.get_cache_dir")
@patch("openesm.utils.console")
def test_cache_info_no_cache(mock_console, mock_get_cache_dir, temp_cache_dir):
    """Test cache_info when cache directory doesn't exist."""
    non_existent = temp_cache_dir / "non_existent"
    mock_get_cache_dir.return_value = non_existent

    cache_info()

    # Check that appropriate messages are printed
    expected_calls = [
        unittest.mock.call("[blue]ℹ[/blue] Cache directory does not exist yet."),
        unittest.mock.call(f"[blue]ℹ[/blue] It will be created at: {non_existent}"),
    ]
    mock_console.print.assert_has_calls(expected_calls)


@patch("openesm.utils.get_cache_dir")
@patch("openesm.utils.console")
def test_cache_info_with_cache(mock_console, mock_get_cache_dir, temp_cache_dir):
    """Test cache_info when cache directory exists with files."""
    mock_get_cache_dir.return_value = temp_cache_dir

    # Create some test files
    test_file1 = temp_cache_dir / "file1.txt"
    test_file2 = temp_cache_dir / "subdir" / "file2.txt"
    test_file1.write_text("test content 1")
    test_file2.parent.mkdir(parents=True, exist_ok=True)
    test_file2.write_text("test content 2")

    cache_info()

    # Should print location and size
    calls = mock_console.print.call_args_list
    assert len(calls) == 2
    # Check that cache location is mentioned (handle Windows path escaping)
    location_call = str(calls[0])
    assert "Cache location:" in location_call
    assert "Cache size:" in str(calls[1])


@patch("openesm.utils.get_cache_dir")
@patch("openesm.utils.console")
def test_clear_cache_no_cache(mock_console, mock_get_cache_dir, temp_cache_dir):
    """Test clear_cache when cache directory doesn't exist."""
    non_existent = temp_cache_dir / "non_existent"
    mock_get_cache_dir.return_value = non_existent

    clear_cache(force=True)

    mock_console.print.assert_called_once_with(
        "[blue]ℹ[/blue] Cache directory does not exist. Nothing to clear."
    )


@patch("openesm.utils.get_cache_dir")
@patch("openesm.utils.shutil.rmtree")
@patch("openesm.utils.console")
def test_clear_cache_force(
    mock_console, mock_rmtree, mock_get_cache_dir, temp_cache_dir
):
    """Test clear_cache with force=True."""
    mock_get_cache_dir.return_value = temp_cache_dir

    # Create the directory to simulate it exists
    temp_cache_dir.mkdir(exist_ok=True)

    clear_cache(force=True)

    mock_rmtree.assert_called_once_with(temp_cache_dir)
    mock_console.print.assert_called_once_with("[green]✓[/green] Cache cleared.")


@patch("openesm.utils.get_cache_dir")
@patch("openesm.utils._is_interactive", return_value=False)
def test_clear_cache_non_interactive_no_force(
    mock_is_interactive, mock_get_cache_dir, temp_cache_dir
):
    """Test clear_cache raises error in non-interactive mode without force."""
    mock_get_cache_dir.return_value = temp_cache_dir
    temp_cache_dir.mkdir(exist_ok=True)

    with pytest.raises(RuntimeError, match="Cannot ask for confirmation"):
        clear_cache(force=False)


@patch("openesm.utils.get_cache_dir")
@patch("openesm.utils._is_interactive", return_value=True)
@patch("openesm.utils.input", return_value="y")
@patch("openesm.utils.shutil.rmtree")
@patch("openesm.utils.console")
def test_clear_cache_interactive_yes(
    mock_console,
    mock_rmtree,
    mock_input,
    mock_is_interactive,
    mock_get_cache_dir,
    temp_cache_dir,
):
    """Test clear_cache with interactive confirmation - user says yes."""
    mock_get_cache_dir.return_value = temp_cache_dir
    temp_cache_dir.mkdir(exist_ok=True)

    clear_cache(force=False)

    mock_rmtree.assert_called_once_with(temp_cache_dir)
    # Should ask for confirmation and then confirm deletion
    assert mock_console.print.call_count >= 2


@patch("openesm.utils.get_cache_dir")
@patch("openesm.utils._is_interactive", return_value=True)
@patch("openesm.utils.input", return_value="n")
@patch("openesm.utils.shutil.rmtree")
@patch("openesm.utils.console")
def test_clear_cache_interactive_no(
    mock_console,
    mock_rmtree,
    mock_input,
    mock_is_interactive,
    mock_get_cache_dir,
    temp_cache_dir,
):
    """Test clear_cache with interactive confirmation - user says no."""
    mock_get_cache_dir.return_value = temp_cache_dir
    temp_cache_dir.mkdir(exist_ok=True)

    clear_cache(force=False)

    mock_rmtree.assert_not_called()
    # Should print cancellation message
    last_call = mock_console.print.call_args_list[-1]
    assert "Cache not cleared" in str(last_call)


def test_download_with_progress_success(temp_cache_dir):
    """Test download_with_progress with successful download."""
    with requests_mock.Mocker() as m:
        test_content = "test file content"
        m.get("https://example.com/test.txt", text=test_content)

        dest_file = temp_cache_dir / "downloaded.txt"

        result = download_with_progress("https://example.com/test.txt", dest_file)

        assert result is True
        assert dest_file.exists()
        assert dest_file.read_text() == test_content


def test_download_with_progress_failure(temp_cache_dir):
    """Test download_with_progress with failed download."""
    with requests_mock.Mocker() as m:
        m.get("https://example.com/test.txt", status_code=404)

        dest_file = temp_cache_dir / "downloaded.txt"

        with pytest.raises(requests.RequestException, match="Download failed"):
            download_with_progress("https://example.com/test.txt", dest_file)


def test_read_json_safe_file_not_found():
    """Test read_json_safe with non-existent file."""
    with pytest.raises(ValueError, match="Failed to read JSON file"):
        read_json_safe("/path/that/does/not/exist.json")


def test_process_metadata_with_list_values():
    """Test processing metadata with list values."""
    meta_with_lists = {
        "dataset_id": "0003",
        "topics": ["mood", "anxiety", "stress"],
        "additional_comments": ["First comment", "Second comment"],
        "features": [
            {"name": "mood", "type": "numeric"},
            {"name": "anxiety", "type": "numeric"},
        ],
    }

    result = process_specific_metadata(meta_with_lists)

    assert result["topics"] == "mood, anxiety, stress"
    assert result["additional_comments"] == "First comment, Second comment"
    # Features should be converted to polars DataFrame or kept as list
    assert result["features"] is not None


def test_process_metadata_with_empty_values():
    """Test processing metadata with empty lists and dicts."""
    meta_empty = {
        "dataset_id": "0004",
        "topics": [],
        "features": [],
    }

    result = process_specific_metadata(meta_empty)

    assert result["topics"] is None
    assert result["features"] is None
    # Test a field that doesn't exist in the metadata
    assert result["additional_comments"] is None


def test_process_metadata_single_item_list():
    """Single-item list is unwrapped to the value itself."""
    meta = {"dataset_id": "0005", "topics": ["stress"]}
    result = process_specific_metadata(meta)
    assert result["topics"] == "stress"


def test_process_metadata_dict_value():
    """Dict values are converted to string representation."""
    meta = {"dataset_id": "0006", "topics": {"key": "value"}}
    result = process_specific_metadata(meta)
    assert isinstance(result["topics"], str)


def test_process_metadata_features_as_list_fallback():
    """Features that cannot be converted to DataFrame fall back to list repr."""
    # pass a list of non-uniform dicts that polars would reject
    meta = {
        "dataset_id": "0007",
        "features": [{"a": 1}, {"b": 2}],  # mismatched keys -> polars may fail
    }
    result = process_specific_metadata(meta)
    # result["features"] is either a DataFrame or the raw list — just not None
    assert result["features"] is not None


def test_format_bytes_tb_branch(temp_cache_dir):
    """cache_info format_bytes reaches TB branch for very large sizes."""
    from unittest.mock import patch as _patch

    large_size = 2 * 1024**4  # 2 TB
    fake_stat = unittest.mock.MagicMock()
    fake_stat.st_size = large_size

    # Build a fake file in temp_cache_dir so os.walk finds it
    fake_file = temp_cache_dir / "bigfile.dat"
    fake_file.write_bytes(b"x")

    with (
        _patch("openesm.utils.get_cache_dir", return_value=temp_cache_dir),
        _patch("openesm.utils.os.walk") as mock_walk,
        _patch("openesm.utils.os.path.getsize", return_value=large_size),
    ):
        mock_walk.return_value = [(str(temp_cache_dir), [], ["bigfile.dat"])]
        # should not raise; the TB branch executes
        cache_info()


def test_find_datasets_json_in_zip(tmp_path):
    """find_datasets_json_in_zip locates datasets.json inside a ZIP."""
    import zipfile

    from openesm.utils import find_datasets_json_in_zip

    zip_path = tmp_path / "meta.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("openesm-1.0.0/datasets.json", '{"datasets": []}')

    result = find_datasets_json_in_zip(zip_path)
    assert result == "openesm-1.0.0/datasets.json"


def test_find_datasets_json_in_zip_not_found(tmp_path):
    """find_datasets_json_in_zip returns None when no datasets.json present."""
    import zipfile

    from openesm.utils import find_datasets_json_in_zip

    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("other.txt", "hello")

    assert find_datasets_json_in_zip(zip_path) is None


def test_find_datasets_json_bad_zip(tmp_path):
    """find_datasets_json_in_zip returns None for a corrupt ZIP."""
    from openesm.utils import find_datasets_json_in_zip

    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"not a zip")

    assert find_datasets_json_in_zip(bad_zip) is None


def test_extract_datasets_json_from_zip_success(tmp_path):
    """extract_datasets_json_from_zip writes datasets.json to dest_path."""
    import zipfile

    from openesm.utils import extract_datasets_json_from_zip

    zip_path = tmp_path / "meta.zip"
    content = '{"datasets": [{"dataset_id": "0001"}]}'
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("sub/datasets.json", content)

    dest = tmp_path / "out" / "datasets.json"
    result = extract_datasets_json_from_zip(zip_path, dest)

    assert result is True
    assert dest.exists()
    assert dest.read_text() == content


def test_extract_datasets_json_from_zip_no_json(tmp_path):
    """extract_datasets_json_from_zip returns False when no datasets.json in ZIP."""
    import zipfile

    from openesm.utils import extract_datasets_json_from_zip

    zip_path = tmp_path / "meta.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "hello")

    dest = tmp_path / "datasets.json"
    result = extract_datasets_json_from_zip(zip_path, dest)
    assert result is False


def test_download_metadata_from_zenodo_cache_hit(tmp_path):
    """download_metadata_from_zenodo returns cached path when fresh enough."""
    from openesm.utils import download_metadata_from_zenodo

    # create a fake cached datasets.json that is "fresh" (just created)
    cache_dir = tmp_path / "v1.0.0"
    cache_dir.mkdir(parents=True)
    datasets_json = cache_dir / "datasets.json"
    datasets_json.write_text('{"datasets": []}')

    with (
        # resolve_zenodo_version is a local import inside download_metadata_from_zenodo
        patch("openesm.zenodo.resolve_zenodo_version", return_value="1.0.0"),
        patch("openesm.utils.get_metadata_dir", return_value=cache_dir),
    ):
        result = download_metadata_from_zenodo(metadata_version="1.0.0", cache_hours=24)

    assert result == datasets_json
