"""Tests for openesm.zenodo module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import requests_mock

# Add src to path to import openesm
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openesm.zenodo import (
    _extract_record_id,
    _get_zenodo_versions,
    download_from_zenodo,
    resolve_zenodo_version,
)


@pytest.fixture
def sample_zenodo_record():
    """Sample Zenodo record data for testing."""
    return {
        "id": 12345,
        "conceptrecid": 12340,
        "metadata": {
            "doi": "10.5072/zenodo.12345",
            "version": "1.0.0",
            "publication_date": "2023-01-15"
        }
    }


@pytest.fixture
def sample_versions_response():
    """Sample Zenodo versions API response."""
    return {
        "hits": {
            "hits": [
                {
                    "id": 12345,
                    "doi": "10.5072/zenodo.12345",
                    "metadata": {
                        "doi": "10.5072/zenodo.12345",
                        "version": "1.0.0",
                        "publication_date": "2023-01-15"
                    }
                },
                {
                    "id": 12344,
                    "doi": "10.5072/zenodo.12344", 
                    "metadata": {
                        "doi": "10.5072/zenodo.12344",
                        "version": "0.9.0",
                        "publication_date": "2022-12-10"
                    }
                }
            ]
        }
    }


class TestExtractRecordId:
    """Tests for _extract_record_id function."""

    def test_extract_from_doi_format(self):
        """Test extracting record ID from DOI format."""
        doi = "10.5072/zenodo.308201"
        result = _extract_record_id(doi)
        assert result == "308201"

    def test_extract_from_zenodo_doi(self):
        """Test extracting from zenodo DOI format.""" 
        doi = "10.5281/zenodo.7891234"
        result = _extract_record_id(doi)
        assert result == "7891234"

    def test_extract_from_url_records(self):
        """Test extracting from Zenodo URL with 'records'."""
        url = "https://zenodo.org/records/308201"
        result = _extract_record_id(url)
        assert result == "308201"

    def test_extract_from_url_record_singular(self):
        """Test extracting from Zenodo URL with 'record'."""
        url = "https://zenodo.org/record/308201"
        result = _extract_record_id(url)
        assert result == "308201"

    def test_extract_from_sandbox_url(self):
        """Test extracting from sandbox URL."""
        url = "https://sandbox.zenodo.org/records/123456"
        result = _extract_record_id(url)
        assert result == "123456"

    def test_invalid_doi_format(self):
        """Test error handling for invalid DOI format."""
        with pytest.raises(ValueError, match="Invalid Zenodo DOI format"):
            _extract_record_id("invalid-doi-format")

    def test_invalid_url_format(self):
        """Test error handling for invalid URL format."""
        with pytest.raises(ValueError, match="Invalid Zenodo DOI format"):
            _extract_record_id("https://example.com/invalid")


class TestGetZenodoVersions:
    """Tests for _get_zenodo_versions function."""

    def test_get_versions_success(self, sample_zenodo_record, sample_versions_response):
        """Test successful version retrieval."""
        record_id = "12345"
        
        with requests_mock.Mocker() as mock_requests:
            # Mock the single record API call
            mock_requests.get(
                f"https://zenodo.org/api/records/{record_id}",
                json=sample_zenodo_record
            )
            
            # Mock the search API call
            mock_requests.get(
                "https://zenodo.org/api/records",
                json=sample_versions_response
            )
            
            result = _get_zenodo_versions(record_id)
            
            # Should return sorted versions
            assert len(result) == 2
            assert result[0]["version"] == "1.0.0"
            assert result[0]["id"] == "12345"
            assert result[1]["version"] == "0.9.0"
            assert result[1]["id"] == "12344"

    def test_get_versions_sandbox(self, sample_zenodo_record, sample_versions_response):
        """Test version retrieval from sandbox."""
        record_id = "12345"
        
        with requests_mock.Mocker() as mock_requests:
            # Mock sandbox URLs
            mock_requests.get(
                f"https://sandbox.zenodo.org/api/records/{record_id}",
                json=sample_zenodo_record
            )
            
            mock_requests.get(
                "https://sandbox.zenodo.org/api/records",
                json=sample_versions_response
            )
            
            result = _get_zenodo_versions(record_id, sandbox=True)
            
            assert len(result) == 2
            assert result[0]["version"] == "1.0.0"

    def test_get_versions_no_conceptrecid(self):
        """Test error when conceptrecid is missing."""
        record_id = "12345"
        
        with requests_mock.Mocker() as mock_requests:
            # Mock record without conceptrecid
            mock_requests.get(
                f"https://zenodo.org/api/records/{record_id}",
                json={"id": 12345, "metadata": {}}
            )
            
            with pytest.raises(ValueError, match="Could not find conceptrecid"):
                _get_zenodo_versions(record_id)

    def test_get_versions_api_error(self):
        """Test handling of API request errors."""
        record_id = "12345"
        
        with requests_mock.Mocker() as mock_requests:
            # Mock API error
            mock_requests.get(
                f"https://zenodo.org/api/records/{record_id}",
                status_code=404
            )
            
            with pytest.raises(requests.RequestException, match="Failed to fetch versions from Zenodo"):
                _get_zenodo_versions(record_id)

    def test_get_versions_with_date_fallback(self):
        """Test version handling when explicit version is missing."""
        record_id = "12345"
        
        with requests_mock.Mocker() as mock_requests:
            # Record with conceptrecid
            sample_record = {
                "id": 12345,
                "conceptrecid": 12340,
                "metadata": {"doi": "10.5072/zenodo.12345"}
            }
            
            # Versions response with missing version (should use date)
            versions_response = {
                "hits": {
                    "hits": [
                        {
                            "id": 12345,
                            "doi": "10.5072/zenodo.12345",
                            "metadata": {
                                "doi": "10.5072/zenodo.12345",
                                "publication_date": "2023-01-15"
                                # No explicit version
                            }
                        }
                    ]
                }
            }
            
            mock_requests.get(
                f"https://zenodo.org/api/records/{record_id}",
                json=sample_record
            )
            
            mock_requests.get(
                "https://zenodo.org/api/records",
                json=versions_response
            )
            
            result = _get_zenodo_versions(record_id)
            
            assert len(result) == 1
            assert result[0]["version"] == "2023-01-15"  # Should use publication date


class TestResolveZenodoVersion:
    """Tests for resolve_zenodo_version function."""

    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_resolve_latest_version(self, mock_extract, mock_get_versions):
        """Test resolving 'latest' version."""
        mock_extract.return_value = "12345"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"},
            {"id": "12344", "version": "0.9.0", "doi": "10.5072/zenodo.12344"}
        ]
        
        result = resolve_zenodo_version("10.5072/zenodo.12340", version="latest")
        
        assert result == "1.0.0"
        mock_extract.assert_called_once_with("10.5072/zenodo.12340")
        mock_get_versions.assert_called_once_with("12345", sandbox=False)

    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_resolve_specific_version(self, mock_extract, mock_get_versions):
        """Test resolving specific version."""
        mock_extract.return_value = "12345"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"},
            {"id": "12344", "version": "0.9.0", "doi": "10.5072/zenodo.12344"}
        ]
        
        result = resolve_zenodo_version("10.5072/zenodo.12340", version="0.9.0")
        
        assert result == "0.9.0"

    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_resolve_version_not_found(self, mock_extract, mock_get_versions):
        """Test error when requested version doesn't exist."""
        mock_extract.return_value = "12345"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"}
        ]
        
        with pytest.raises(ValueError, match="Version 2.0.0 not found"):
            resolve_zenodo_version("10.5072/zenodo.12340", version="2.0.0")

    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_resolve_no_versions_found(self, mock_extract, mock_get_versions):
        """Test error when no versions are found."""
        mock_extract.return_value = "12345"
        mock_get_versions.return_value = []
        
        with pytest.raises(ValueError, match="No versions found"):
            resolve_zenodo_version("10.5072/zenodo.12340")

    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_resolve_with_sandbox(self, mock_extract, mock_get_versions):
        """Test resolving with sandbox parameter."""
        mock_extract.return_value = "12345"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"}
        ]
        
        result = resolve_zenodo_version("10.5072/zenodo.12340", sandbox=True)
        
        assert result == "1.0.0"
        mock_get_versions.assert_called_once_with("12345", sandbox=True)


class TestDownloadFromZenodo:
    """Tests for download_from_zenodo function."""

    @patch("openesm.zenodo.download_with_progress")
    @patch("openesm.zenodo.msg_info")
    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_download_success(self, mock_extract, mock_get_versions, mock_msg_info, mock_download):
        """Test successful download from Zenodo."""
        mock_extract.return_value = "12340"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"}
        ]
        
        result = download_from_zenodo(
            "10.5072/zenodo.12340",
            "0001", 
            "Smith",
            "1.0.0"
        )
        
        # Should construct correct filename and URL
        expected_filename = "0001_Smith_ts.tsv"
        expected_url = f"https://zenodo.org/records/12345/files/{expected_filename}"
        
        mock_download.assert_called_once_with(expected_url, Path(expected_filename))
        mock_msg_info.assert_called_once_with(f"Downloading {expected_filename} from Zenodo (version 1.0.0)")
        
        assert result == Path(expected_filename)

    @patch("openesm.zenodo.download_with_progress")
    @patch("openesm.zenodo.msg_info")
    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_download_with_custom_dest_path(self, mock_extract, mock_get_versions, mock_msg_info, mock_download):
        """Test download with custom destination path."""
        mock_extract.return_value = "12340"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"}
        ]
        
        custom_path = Path("/custom/path/data.tsv")
        result = download_from_zenodo(
            "10.5072/zenodo.12340",
            "0001",
            "Smith", 
            "1.0.0",
            dest_path=custom_path
        )
        
        mock_download.assert_called_once_with(
            "https://zenodo.org/records/12345/files/0001_Smith_ts.tsv",
            custom_path
        )
        assert result == custom_path

    @patch("openesm.zenodo.download_with_progress")
    @patch("openesm.zenodo.msg_info")
    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_download_from_sandbox(self, mock_extract, mock_get_versions, mock_msg_info, mock_download):
        """Test download from sandbox environment."""
        mock_extract.return_value = "12340"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"}
        ]
        
        result = download_from_zenodo(
            "10.5072/zenodo.12340",
            "0001",
            "Smith",
            "1.0.0",
            sandbox=True
        )
        
        # Should use sandbox URL
        expected_url = "https://sandbox.zenodo.org/records/12345/files/0001_Smith_ts.tsv"
        mock_download.assert_called_once_with(expected_url, Path("0001_Smith_ts.tsv"))

    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_download_version_not_found(self, mock_extract, mock_get_versions):
        """Test error when requested version is not found."""
        mock_extract.return_value = "12340"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"}
        ]
        
        with pytest.raises(ValueError, match="Version 2.0.0 not found"):
            download_from_zenodo(
                "10.5072/zenodo.12340",
                "0001",
                "Smith",
                "2.0.0"
            )

    @patch("openesm.zenodo.download_with_progress")
    @patch("openesm.zenodo.msg_info")
    @patch("openesm.zenodo._get_zenodo_versions")
    @patch("openesm.zenodo._extract_record_id")
    def test_download_with_string_dest_path(self, mock_extract, mock_get_versions, mock_msg_info, mock_download):
        """Test download with string destination path."""
        mock_extract.return_value = "12340"
        mock_get_versions.return_value = [
            {"id": "12345", "version": "1.0.0", "doi": "10.5072/zenodo.12345"}
        ]
        
        result = download_from_zenodo(
            "10.5072/zenodo.12340",
            "0001",
            "Smith",
            "1.0.0", 
            dest_path="custom_file.tsv"
        )
        
        mock_download.assert_called_once_with(
            "https://zenodo.org/records/12345/files/0001_Smith_ts.tsv",
            Path("custom_file.tsv")
        )
        assert result == Path("custom_file.tsv")
