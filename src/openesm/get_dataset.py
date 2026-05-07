"""Download ESM datasets from openESM repository."""

from pathlib import Path
from typing import Any, Union

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .list_datasets import list_datasets
from .utils import (
    download_metadata_from_zenodo,
    get_cache_path,
    msg_info,
    msg_success,
    process_specific_metadata,
    read_json_safe,
)
from .zenodo import download_from_zenodo, get_zenodo_versions


class OpenESMDataset:
    """A single ESM dataset with data and metadata.

    This class represents a downloaded ESM dataset containing the actual data
    as a polars DataFrame along with comprehensive metadata and helper methods
    for citations and licensing information.

    Attributes:
        data: polars DataFrame containing the ESM data
        metadata: Dictionary containing dataset metadata
        dataset_id: String identifier for the dataset
        dataset_version: String version identifier for the dataset
        metadata_version: String version identifier for the metadata catalog
    """

    def __init__(
        self,
        data: pl.DataFrame,
        metadata: dict[str, Any],
        dataset_id: str,
        dataset_version: str,
        metadata_version: str,
    ):
        """Initialize an OpenESM dataset.

        Args:
            data: polars DataFrame with ESM data
            metadata: Dictionary with dataset metadata
            dataset_id: Dataset identifier
            dataset_version: Dataset version string
            metadata_version: Metadata catalog version string
        """
        self.data = data
        self.metadata = metadata
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.metadata_version = metadata_version
        # Keep 'version' for backward compatibility
        self.version = dataset_version

    def cite(self) -> str:
        """Get citation information for this dataset.

        Displays formatted citation information and returns the citation string.

        Returns:
            Citation string for use in publications
        """
        console = Console()

        # collect valid citations
        citations = []

        ref_a = self.metadata.get("reference_a")
        ref_b = self.metadata.get("reference_b")

        if ref_a and str(ref_a).strip():
            citations.append(str(ref_a).strip())
        if ref_b and str(ref_b).strip():
            citations.append(str(ref_b).strip())

        if not citations:
            console.print(
                "ℹ️  No citation information available for this dataset.", style="blue"
            )
            return ""

        # display formatted output
        console.print(
            "\n📖 To cite this dataset in publications, please use:\n",
            style="bold blue",
        )

        for i, citation in enumerate(citations):
            if i > 0:
                console.print()  # empty line between citations

            # display citation in a panel for better formatting
            panel = Panel(
                Text(citation, style="italic"), border_style="dim", padding=(0, 1)
            )
            console.print(panel)

        console.print()  # empty line at end

        return "\n\n".join(citations)

    def license(self) -> str:
        """Get license information for this dataset.

        Displays formatted license information and returns the license string.

        Returns:
            License information string
        """
        console = Console()

        license_info = self.metadata.get("license")

        if license_info and str(license_info).strip():
            license_text = str(license_info).strip()
            console.print("\n📄 License Information:", style="bold green")
            console.print(f"   {license_text}", style="green")
            console.print()
            return license_text
        else:
            console.print(
                "ℹ️  License information not available. Please check the original "
                "publication.",
                style="blue",
            )
            return (
                "License information not available. Please check the original "
                "publication."
            )

    def notes(self) -> str:
        """Get additional notes and comments for this dataset.

        Displays formatted notes and returns the notes string.

        Returns:
            Notes and comments string
        """
        console = Console()

        # collect notes from various sources
        notes = []

        # add comments if available
        comments = self.metadata.get("additional_comments")
        if comments and str(comments).strip():
            # split by semicolons and clean up
            comment_parts = [part.strip() for part in str(comments).split(";")]
            comment_parts = [
                part for part in comment_parts if part
            ]  # remove empty parts
            notes.extend(comment_parts)

        # add basic dataset info
        n_participants = self.metadata.get("n_participants")
        n_timepoints = self.metadata.get("n_time_points")

        if n_participants:
            notes.append(f"Participants: {n_participants}")
        if n_timepoints:
            notes.append(f"Time points: {n_timepoints}")

        if not notes:
            console.print(
                "ℹ️  No additional notes available for this dataset.", style="blue"
            )
            return ""

        # display formatted output
        console.print(f"\n📝 Notes for dataset {self.dataset_id}:", style="bold yellow")

        for note in notes:
            console.print(f"   • {note}", style="yellow")

        console.print()  # empty line at end

        return "\n".join(notes)

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"OpenESMDataset(id='{self.dataset_id}', "
            f"dataset_version='{self.dataset_version}', "
            f"metadata_version='{self.metadata_version}', "
            f"shape={self.data.shape})"
        )

    def __str__(self) -> str:
        """User-friendly string representation matching R package style."""
        console = Console(file=None, width=80)  # for capturing output

        with console.capture() as capture:
            # main header
            console.print(f"\n📊 OpenESM Dataset: {self.dataset_id}", style="bold blue")

            # metadata bullets
            meta = self.metadata
            author = meta.get("first_author", "Unknown")
            year = meta.get("year", "Unknown")
            paper_doi = meta.get("paper_doi", "Not available")
            license_info = meta.get("license", "Not specified")
            n_participants = meta.get("n_participants", "Unknown")
            n_timepoints = meta.get("n_time_points", "Unknown")

            console.print(f"   • Version: {self.dataset_version}", style="cyan")
            console.print(f"   • Authors: {author} et al. ({year})", style="cyan")
            console.print(f"   • Paper DOI: {paper_doi}", style="cyan")
            console.print(f"   • License: {license_info}", style="cyan")
            console.print(
                f"   • Data: A DataFrame with {n_participants} participants and "
                f"{n_timepoints} maximum time points per participant",
                style="cyan",
            )

            # helper info
            console.print(
                "\nℹ️  Use dataset.cite() for citation information.", style="dim blue"
            )
            console.print(
                "ℹ️  Use dataset.notes() for additional information about the dataset.",
                style="dim blue",
            )
            console.print(
                "ℹ️  Please ensure you follow the license terms for this dataset.",
                style="dim blue",
            )
            console.print()

        return capture.get()


class OpenESMDatasetList:
    """A collection of multiple ESM datasets.

    This class represents multiple downloaded datasets, providing convenient
    access methods and batch operations.
    """

    def __init__(self, datasets: dict[str, OpenESMDataset], metadata_version: str):
        """Initialize a dataset list.

        Args:
            datasets: Dictionary mapping dataset IDs to OpenESMDataset objects
            metadata_version: Version of metadata catalog used
        """
        self.datasets = datasets
        self.metadata_version = metadata_version

    def __getitem__(self, key: str) -> OpenESMDataset:
        """Get a dataset by ID."""
        return self.datasets[key]

    def __iter__(self) -> Any:
        """Iterate over dataset IDs."""
        return iter(self.datasets)

    def __len__(self) -> int:
        """Number of datasets."""
        return len(self.datasets)

    def keys(self) -> Any:
        """Get dataset IDs."""
        return self.datasets.keys()

    def values(self) -> Any:
        """Get dataset objects."""
        return self.datasets.values()

    def items(self) -> Any:
        """Get (ID, dataset) pairs."""
        return self.datasets.items()

    def __repr__(self) -> str:
        """String representation."""
        keys = list(self.datasets.keys())
        return f"OpenESMDatasetList({keys}, metadata_version='{self.metadata_version}')"

    def __str__(self) -> str:
        """User-friendly string representation matching R package style."""
        console = Console(file=None, width=80)

        with console.capture() as capture:
            num_datasets = len(self.datasets)
            dataset_word = "Dataset" if num_datasets == 1 else "Datasets"

            console.print(
                f"\n📚 Collection of {num_datasets} OpenESM {dataset_word}",
                style="bold blue",
            )

            # show dataset names (limit to first 5)
            max_show = 5
            dataset_ids = list(self.datasets.keys())

            for dataset_id in dataset_ids[:max_show]:
                dataset = self.datasets[dataset_id]
                author = dataset.metadata.get("first_author", "Unknown")
                year = dataset.metadata.get("year", "Unknown")
                shape = dataset.data.shape
                console.print(
                    f"   • {dataset_id}: {author} et al. ({year}) - "
                    f"{shape[0]}×{shape[1]}",
                    style="cyan",
                )

            if num_datasets > max_show:
                remaining = num_datasets - max_show
                console.print(f"   ... and {remaining} more.", style="dim cyan")

            console.print(
                f"\nℹ️  Access individual datasets using collection['{dataset_ids[0]}']",
                style="dim blue",
            )
            console.print()

        return capture.get()


def get_dataset(
    dataset_id: Union[str, list[str]],
    version: Union[str, list[str]] = "latest",
    metadata_version: str = "latest",
    cache: bool = True,
    path: Union[str, Path, None] = None,
    force_download: bool = False,
    sandbox: bool = False,
    quiet: bool = False,
) -> Union[OpenESMDataset, OpenESMDatasetList]:
    """Download ESM dataset(s) from openESM repository.

    Downloads one or more Experience Sampling Method (ESM) datasets from the
    openESM repository hosted on Zenodo. Returns an OpenESMDataset object or
    OpenESMDatasetList containing the dataset(s) and associated metadata.

    Args:
        dataset_id: String or list of dataset IDs. Use list_datasets() to see
            available datasets.
        version: Dataset version to download. Default is "latest". When downloading
            multiple datasets, can be a list of the same length as dataset_id to
            pin each dataset to a specific version. A single non-"latest" version
            is recycled with a warning.
        metadata_version: Version of metadata catalog to use. Default is "latest".
        cache: If True, uses cached version if available. Default is True.
        path: Custom download path. If None, files are cached in user's cache directory.
        force_download: If True, forces re-download even if cached version exists.
        sandbox: If True, uses Zenodo sandbox environment for testing.
        quiet: If True, suppresses informational messages.

    Returns:
        For single dataset: OpenESMDataset object containing data and metadata.
        For multiple datasets: OpenESMDatasetList containing multiple datasets.

    Examples:
        >>> # List available datasets first
        >>> available = list_datasets()
        >>>
        >>> # Download a single dataset
        >>> dataset = get_dataset("0001")
        >>>
        >>> # Download with specific versions for reproducibility
        >>> dataset = get_dataset("0001", version="1.0.0", metadata_version="1.0.0")
        >>>
        >>> # Access the data
        >>> print(dataset.data.head())
        >>>
        >>> # View metadata and version info
        >>> print(dataset.metadata)
        >>> print(f"Dataset version: {dataset.dataset_version}")
        >>> print(f"Metadata version: {dataset.metadata_version}")
        >>>
        >>> # Download multiple datasets
        >>> datasets = get_dataset(["0001", "0002"])
        >>>
        >>> # Access individual datasets
        >>> print(datasets["0001"].data.head())
    """
    # handle multiple datasets
    if isinstance(dataset_id, list):
        return _get_multiple_datasets(
            dataset_id, version, metadata_version, cache, force_download, sandbox, quiet
        )

    # get dataset catalog using specified metadata version
    all_datasets = list_datasets(metadata_version=metadata_version)

    # extract first sequence of digits from dataset_id
    original_dataset_id = str(dataset_id).strip()
    dataset_id = ""

    # find first continuous sequence of digits
    for char in original_dataset_id:
        if char.isdigit():
            dataset_id += char
        elif dataset_id:  # if we started collecting digits and hit non-digit, stop
            break

    # validate that we found some digits
    if not dataset_id:
        raise ValueError(f"No numeric dataset ID found in '{original_dataset_id}'")

    # filter out datasets with None/null IDs
    all_datasets = all_datasets.filter(pl.col("dataset_id").is_not_null())

    # check if dataset exists
    if dataset_id not in all_datasets["dataset_id"].to_list():
        raise ValueError(f"Dataset with id '{dataset_id}' not found.")

    # get dataset info
    dataset_info = all_datasets.filter(pl.col("dataset_id") == dataset_id).row(
        0, named=True
    )

    author_raw = dataset_info["first_author"]
    # normalise umlauts before lowercasing
    author_raw = author_raw.replace("ö", "oe").replace("Ö", "oe")
    author_raw = author_raw.replace("ä", "ae").replace("Ä", "ae")
    author_raw = author_raw.replace("ü", "ue").replace("Ü", "ue")
    author_lower = author_raw.lower()
    # remove spaces from author name
    author_lower = author_lower.replace(" ", "")

    # get metadata using Zenodo infrastructure (same as list_datasets uses)
    # download_metadata_from_zenodo resolves "latest" internally; derive the
    # actual version from the cache path (.../metadata/v<ver>/datasets.json)
    metadata_path = download_metadata_from_zenodo(metadata_version=metadata_version)
    resolved_metadata_version = metadata_path.parent.name.lstrip("v")
    metadata_dict = read_json_safe(metadata_path)

    # extract the specific dataset metadata
    specific_datasets = metadata_dict.get("datasets", [])
    specific_meta_raw = None
    for dataset_meta in specific_datasets:
        if dataset_meta.get("dataset_id") == dataset_id:
            specific_meta_raw = dataset_meta
            break

    if specific_meta_raw is None:
        raise ValueError(f"Dataset {dataset_id} not found in metadata")

    # get concept DOI from metadata
    zenodo_doi = specific_meta_raw.get("zenodo_doi")

    if not zenodo_doi:
        raise ValueError(f"No Zenodo DOI found in metadata for dataset {dataset_id}")

    # at this point version is always str (list case was dispatched above)
    if not isinstance(version, str):
        raise TypeError(f"Expected str for version, got {type(version)}")

    # fetch versions once, then resolve both the actual version tag and its DOI
    all_versions = get_zenodo_versions(zenodo_doi, sandbox=sandbox)
    if not all_versions:
        raise ValueError(f"No versions found for dataset {dataset_id} DOI {zenodo_doi}")

    if version == "latest":
        actual_version = all_versions[0]["version"]
    else:
        if version not in [v["version"] for v in all_versions]:
            available = ", ".join(v["version"] for v in all_versions)
            raise ValueError(
                f"Version {version} not found for dataset {dataset_id}. "
                f"Available: {available}"
            )
        actual_version = version

    version_doi = next(
        (v["doi"] for v in all_versions if v["version"] == actual_version), None
    )
    if not version_doi:
        raise ValueError(
            f"Could not resolve version-specific DOI for dataset {dataset_id} "
            f"version {actual_version}"
        )

    # determine cache/destination path
    filename = f"{dataset_id}_{author_lower}_ts.tsv"
    if path is None:
        local_data_path = get_cache_path(
            dataset_id, filename=filename, type_="data", version=actual_version
        )
    else:
        local_data_path = Path(path) / filename

    # download from Zenodo if needed
    if not local_data_path.exists() or force_download:
        download_from_zenodo(
            version_doi=version_doi,
            dataset_id=dataset_id,
            author_name=author_lower,
            version=actual_version,
            sandbox=sandbox,
            dest_path=local_data_path,
        )

    # load dataset
    if not quiet:
        msg_success(f"Loading dataset {dataset_id} version {actual_version}")

    data = pl.read_csv(
        local_data_path,
        separator="\t",
        null_values=["NA", ""],
        # increase schema inference length for large datasets
        infer_schema_length=10000,
    )

    # format metadata for cleaner output
    formatted_meta = process_specific_metadata(specific_meta_raw)

    # create dataset object
    dataset = OpenESMDataset(
        data=data,
        metadata=formatted_meta,
        dataset_id=dataset_id,
        dataset_version=actual_version,
        metadata_version=resolved_metadata_version,
    )

    # print dataset info unless silenced
    if not quiet:
        repro_call = (
            f'openesm.get_dataset("{dataset_id}", '
            f'metadata_version="{resolved_metadata_version}", '
            f'version="{actual_version}")'
        )
        msg_info(f"For full reproducibility, use:\n{repro_call}")
        print(dataset)

    return dataset


def _get_multiple_datasets(
    dataset_ids: list[str],
    version: Union[str, list[str]],
    metadata_version: str,
    cache: bool,
    force_download: bool,
    sandbox: bool,
    quiet: bool,
) -> OpenESMDatasetList:
    """Helper function for downloading multiple datasets.

    Args:
        dataset_ids: List of dataset IDs to download
        version: Dataset version(s). Single string or list matching dataset_ids length.
        metadata_version: Metadata version
        cache: Whether to use cache
        force_download: Whether to force re-download
        sandbox: Whether to use sandbox
        quiet: Whether to suppress messages

    Returns:
        OpenESMDatasetList containing the downloaded datasets
    """
    import warnings

    n = len(dataset_ids)

    # normalise version to a list of length n
    if isinstance(version, str):
        if version != "latest":
            warnings.warn(
                f"Recycling version '{version}' across all {n} datasets. "
                "Pass a list of versions to pin each dataset individually.",
                stacklevel=4,
            )
        versions = [version] * n
    elif isinstance(version, list):
        if len(version) != n:
            raise ValueError(
                f"'version' must be length 1 or the same length as 'dataset_id' "
                f"({n}), not {len(version)}."
            )
        versions = version
    else:
        raise TypeError(f"'version' must be a str or list, got {type(version)}")

    datasets = {}

    for dataset_id, ver in zip(dataset_ids, versions):
        if not quiet:
            msg_info(f"Downloading dataset {dataset_id}")

        # download individual dataset in quiet mode
        dataset = get_dataset(
            dataset_id,
            version=ver,
            metadata_version=metadata_version,
            cache=cache,
            force_download=force_download,
            sandbox=sandbox,
            quiet=True,
        )
        # Single dataset_id always returns OpenESMDataset
        if isinstance(dataset, OpenESMDataset):
            datasets[dataset_id] = dataset
        else:
            raise TypeError(f"Expected OpenESMDataset, got {type(dataset)}")

    # create dataset list
    dataset_list = OpenESMDatasetList(datasets, metadata_version)

    # print summary unless silenced
    if not quiet and datasets:
        resolved_ids = list(datasets.keys())
        resolved_dvs = [datasets[did].dataset_version for did in resolved_ids]
        resolved_mv = datasets[resolved_ids[0]].metadata_version

        ids_repr = '["' + '", "'.join(resolved_ids) + '"]'
        dvs_repr = '["' + '", "'.join(resolved_dvs) + '"]'
        repro_call = (
            f"openesm.get_dataset({ids_repr},\n"
            f'    metadata_version="{resolved_mv}",\n'
            f"    version={dvs_repr})"
        )
        msg_info(f"For full reproducibility, use:\n{repro_call}")
        print(dataset_list)

    return dataset_list
