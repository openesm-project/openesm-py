"""Zenodo API helpers for downloading datasets."""

import re
from pathlib import Path
from typing import Any, Optional, Union

import requests

from .utils import download_with_progress, msg_info


def resolve_zenodo_version(
    zenodo_doi: str, version: str = "latest", sandbox: bool = False
) -> str:
    """Resolve a Zenodo version.

    Given a concept DOI, finds the specific version tag. If "latest" is requested,
    it returns the most recent version tag.

    Args:
        zenodo_doi: Zenodo concept DOI
        version: Either "latest" or a specific version tag (e.g., "1.0.0")
        sandbox: Whether to use Zenodo sandbox

    Returns:
        Resolved version tag

    Raises:
        ValueError: If no versions found or version not available
    """
    # extract record ID from DOI
    record_id = _extract_record_id(zenodo_doi)

    # get versions from Zenodo API
    versions = _get_zenodo_versions(record_id, sandbox=sandbox)

    if not versions:
        raise ValueError(f"No versions found for DOI {zenodo_doi}")

    # sort versions by date descending (most recent first)
    versions.sort(key=lambda v: v.get("date", ""), reverse=True)

    if version == "latest":
        # return the latest version (most recent by publication date)
        latest_version: str = versions[0]["version"]
        return latest_version
    else:
        # check if requested version exists
        version_tags = [v["version"] for v in versions]
        if version not in version_tags:
            available_versions = ", ".join(version_tags)
            raise ValueError(
                f"Version {version} not found. Available versions: {available_versions}"
            )
        return version


def download_from_zenodo(
    zenodo_doi: str,
    dataset_id: str,
    author_name: str,
    version: str,
    sandbox: bool = False,
    dest_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Download dataset from Zenodo.

    Downloads a specific dataset file from Zenodo using the record ID and
    constructs the appropriate download URL based on dataset metadata.

    Args:
        zenodo_doi: Zenodo concept DOI
        dataset_id: Dataset identifier
        author_name: Author name
        version: Specific version tag (e.g., "1.0.0")
        sandbox: Whether to use Zenodo sandbox
        dest_path: Destination path. If None, uses filename only

    Returns:
        Path to downloaded file

    Raises:
        ValueError: If version not found
        requests.RequestException: If download fails
    """
    # get available versions to find the record ID for the specific version
    record_id = _extract_record_id(zenodo_doi)
    versions = _get_zenodo_versions(record_id, sandbox=sandbox)

    # find the specific version
    version_match = None
    for v in versions:
        if v["version"] == version:
            version_match = v
            break

    if version_match is None:
        version_tags = [v["version"] for v in versions]
        available_versions = ", ".join(version_tags)
        raise ValueError(
            f"Version {version} not found. Available versions: {available_versions}"
        )

    # get the specific record ID for this version
    specific_record_id = version_match["id"]

    # construct filename
    filename = f"{dataset_id}_{author_name}_ts.tsv"

    # construct download URL
    if sandbox:
        download_url = (
            f"https://sandbox.zenodo.org/records/{specific_record_id}/files/{filename}"
        )
    else:
        download_url = (
            f"https://zenodo.org/records/{specific_record_id}/files/{filename}"
        )

    # set destination path
    if dest_path is None:
        dest_path = Path(filename)
    else:
        dest_path = Path(dest_path)

    msg_info(f"Downloading {filename} from Zenodo (version {version})")

    # download file using the package's standard download utility
    download_with_progress(download_url, dest_path)

    return dest_path


def download_files_from_zenodo(
    zenodo_doi: str,
    version: str,
    dest_dir: Path,
    file_patterns: Optional[list[str]] = None,
    sandbox: bool = False,
) -> list[Path]:
    """Download files from Zenodo by pattern.

    Downloads files matching specified patterns from a Zenodo record.

    Args:
        zenodo_doi: Zenodo concept DOI
        version: Specific version tag (e.g., "1.0.0")
        dest_dir: Destination directory
        file_patterns: List of file patterns to match (e.g., ["*.zip", "*.json"])
        sandbox: Whether to use Zenodo sandbox

    Returns:
        List of paths to downloaded files

    Raises:
        ValueError: If version not found or no matching files
        requests.RequestException: If download fails
    """
    import fnmatch

    # get available versions to find the record ID for the specific version
    record_id = _extract_record_id(zenodo_doi)
    versions = _get_zenodo_versions(record_id, sandbox=sandbox)

    # find the specific version
    version_match = None
    for v in versions:
        if v["version"] == version:
            version_match = v
            break

    if version_match is None:
        version_tags = [v["version"] for v in versions]
        available_versions = ", ".join(version_tags)
        raise ValueError(
            f"Version {version} not found. Available versions: {available_versions}"
        )

    # get the specific record ID for this version
    specific_record_id = version_match["id"]

    # get files list from the record
    base_url = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    api_url = f"{base_url}/api/records/{specific_record_id}"

    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    record_data = response.json()

    # get available files
    available_files = record_data.get("files", [])

    if file_patterns is None:
        file_patterns = ["*"]  # match all files if no patterns specified

    # find matching files
    matching_files = []
    for file_info in available_files:
        filename = file_info["key"]  # Zenodo API uses "key" for filename
        for pattern in file_patterns:
            if fnmatch.fnmatch(filename, pattern):
                matching_files.append(file_info)
                break

    if not matching_files:
        available_filenames = [f["key"] for f in available_files]
        raise ValueError(
            f"No files matching patterns {file_patterns}. "
            f"Available files: {available_filenames}"
        )

    # download matching files
    downloaded_files = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file_info in matching_files:
        filename = file_info["key"]
        download_url = file_info["links"]["self"]  # direct download URL
        dest_path = dest_dir / filename

        msg_info(f"Downloading {filename} from Zenodo (version {version})")
        download_with_progress(download_url, dest_path)
        downloaded_files.append(dest_path)

    return downloaded_files


def _extract_record_id(zenodo_doi: str) -> str:
    """Extract record ID from Zenodo DOI.

    Args:
        zenodo_doi: Zenodo DOI (e.g., "10.5072/zenodo.308201")

    Returns:
        Record ID string

    Raises:
        ValueError: If DOI format is invalid
    """
    # handle both full URLs and DOI strings
    if zenodo_doi.startswith("http"):
        # extract from URL
        match = re.search(r"zenodo\.org/records?/(\d+)", zenodo_doi)
        if match:
            return match.group(1)

    # extract from DOI format (e.g., "10.5072/zenodo.308201")
    match = re.search(r"zenodo\.(\d+)", zenodo_doi)
    if match:
        return match.group(1)

    raise ValueError(f"Invalid Zenodo DOI format: {zenodo_doi}")


def _get_zenodo_versions(record_id: str, sandbox: bool = False) -> list[dict[str, Any]]:
    """Get all versions for a Zenodo record.

    Args:
        record_id: Zenodo record ID (can be any version of the concept)
        sandbox: Whether to use sandbox environment

    Returns:
        List of version dictionaries with 'id', 'version', and 'doi' fields

    Raises:
        requests.RequestException: If API request fails
    """
    base_url = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"

    try:
        # Step 1: Get the conceptrecid from any record in the concept
        single_record_url = f"{base_url}/records/{record_id}"
        response = requests.get(single_record_url, timeout=30)
        response.raise_for_status()
        single_record_data = response.json()
        conceptrecid = single_record_data.get("conceptrecid")

        if not conceptrecid:
            raise ValueError(f"Could not find conceptrecid for record {record_id}")

        # Step 2: Search for all versions using the conceptrecid
        search_url = f"{base_url}/records"
        params: dict[str, str] = {
            "q": f'conceptrecid:"{conceptrecid}"',
            "all_versions": "true",
            "sort": "version",
            "size": "100",
        }

        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # extract version information
        versions = []
        for hit in data.get("hits", {}).get("hits", []):
            meta = hit["metadata"]
            # use explicit version if available, otherwise use publication date
            version = meta.get("version")
            if version is None:
                # fallback to using publication date as version identifier
                version = meta.get("publication_date", "unknown")

            versions.append(
                {
                    "id": str(hit["id"]),
                    "version": version,
                    "doi": meta.get("doi", hit.get("doi")),
                    "publication_date": meta.get("publication_date", ""),
                }
            )

        # sort by publication date descending (most recent first)
        # this ensures "latest" always returns the most recent version
        versions.sort(key=lambda v: v["publication_date"], reverse=True)

        return versions

    except requests.RequestException as e:
        raise requests.RequestException(
            f"Failed to fetch versions from Zenodo: {e}"
        ) from e
