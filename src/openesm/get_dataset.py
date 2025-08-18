"""Download ESM datasets from OpenESM repository."""

from pathlib import Path
from typing import Any, Union

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .list_datasets import list_datasets
from .utils import (
    download_with_progress,
    get_cache_path,
    msg_info,
    msg_success,
    process_specific_metadata,
    read_json_safe,
)
from .zenodo import download_from_zenodo, resolve_zenodo_version


class OpenESMDataset:
    """A single ESM dataset with data and metadata.

    This class represents a downloaded ESM dataset containing the actual data
    as a polars DataFrame along with comprehensive metadata and helper methods
    for citations and licensing information.

    Attributes:
        data: polars DataFrame containing the ESM data
        metadata: Dictionary containing dataset metadata
        dataset_id: String identifier for the dataset
        version: String version identifier
    """

    def __init__(
        self,
        data: pl.DataFrame,
        metadata: dict[str, Any],
        dataset_id: str,
        version: str,
    ):
        """Initialize an OpenESM dataset.

        Args:
            data: polars DataFrame with ESM data
            metadata: Dictionary with dataset metadata
            dataset_id: Dataset identifier
            version: Version string
        """
        self.data = data
        self.metadata = metadata
        self.dataset_id = dataset_id
        self.version = version

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
            f"version='{self.version}', "
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

            console.print(f"   • Version: {self.version}", style="cyan")
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

    def __init__(self, datasets: dict[str, OpenESMDataset]):
        """Initialize a dataset list.

        Args:
            datasets: Dictionary mapping dataset IDs to OpenESMDataset objects
        """
        self.datasets = datasets

    def __getitem__(self, key: str) -> OpenESMDataset:
        """Get a dataset by ID."""
        return self.datasets[key]

    def __iter__(self):
        """Iterate over dataset IDs."""
        return iter(self.datasets)

    def __len__(self) -> int:
        """Number of datasets."""
        return len(self.datasets)

    def keys(self):
        """Get dataset IDs."""
        return self.datasets.keys()

    def values(self):
        """Get dataset objects."""
        return self.datasets.values()

    def items(self):
        """Get (ID, dataset) pairs."""
        return self.datasets.items()

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenESMDatasetList({list(self.datasets.keys())})"

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
    version: str = "latest",
    cache: bool = True,
    path: Union[str, Path, None] = None,
    force_download: bool = False,
    sandbox: bool = False,
    quiet: bool = False,
) -> Union[OpenESMDataset, OpenESMDatasetList]:
    """Download ESM dataset(s) from OpenESM repository.

    Downloads one or more Experience Sampling Method (ESM) datasets from the
    OpenESM repository hosted on Zenodo. Returns an OpenESMDataset object or
    OpenESMDatasetList containing the dataset(s) and associated metadata.

    Args:
        dataset_id: String or list of dataset IDs. Use list_datasets() to see
            available datasets.
        version: Dataset version to download. Default is "latest".
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
        >>> # Access the data
        >>> print(dataset.data.head())
        >>>
        >>> # View metadata
        >>> print(dataset.metadata)
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
            dataset_id, version, cache, force_download, sandbox, quiet
        )

    # get dataset catalog
    all_datasets = list_datasets()

    # filter out datasets with None/null IDs
    all_datasets = all_datasets.filter(pl.col("dataset_id").is_not_null())

    # check if dataset exists
    if dataset_id not in all_datasets["dataset_id"].to_list():
        raise ValueError(f"Dataset with id '{dataset_id}' not found.")

    # get dataset info
    dataset_info = all_datasets.filter(pl.col("dataset_id") == dataset_id).row(
        0, named=True
    )

    author_lower = dataset_info["first_author"].lower()

    # get metadata from github
    metadata_gh_folder = f"{dataset_info['dataset_id']}_{author_lower}/"
    metadata_gh_path = f"{dataset_info['dataset_id']}_{author_lower}_metadata.json"
    metadata_url = (
        "https://raw.githubusercontent.com/bsiepe/openesm-metadata/main/datasets/"
        f"{metadata_gh_folder}{metadata_gh_path}"
    )

    # cache metadata
    local_metadata_path = get_cache_path(
        dataset_id,
        filename=metadata_gh_path,
        type_="metadata",
        version="latest",  # metadata is not version specific
    )

    if not local_metadata_path.exists() or force_download:
        if not quiet:
            msg_info(f"Downloading metadata for dataset {dataset_id}")
        download_with_progress(metadata_url, local_metadata_path)

    specific_meta_raw = read_json_safe(local_metadata_path)

    # get concept DOI from metadata
    zenodo_doi = specific_meta_raw.get("zenodo_doi")

    if not zenodo_doi:
        raise ValueError(f"No Zenodo DOI found in metadata for dataset {dataset_id}")

    # resolve actual version if "latest" is requested
    actual_version = resolve_zenodo_version(zenodo_doi, version, sandbox)

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
            zenodo_doi=zenodo_doi,
            dataset_id=dataset_id,
            author_name=author_lower,
            version=actual_version,
            sandbox=sandbox,
            dest_path=local_data_path,
        )

    # load dataset
    if not quiet:
        msg_success(f"Loading dataset {dataset_id} version {actual_version}")

    data = pl.read_csv(local_data_path, separator="\t")

    # format metadata for cleaner output
    formatted_meta = process_specific_metadata(specific_meta_raw)

    # create dataset object
    dataset = OpenESMDataset(
        data=data,
        metadata=formatted_meta,
        dataset_id=dataset_id,
        version=actual_version,
    )

    # print dataset info unless silenced
    if not quiet:
        print(dataset)

    return dataset


def _get_multiple_datasets(
    dataset_ids: list[str],
    version: str,
    cache: bool,
    force_download: bool,
    sandbox: bool,
    quiet: bool,
) -> OpenESMDatasetList:
    """Helper function for downloading multiple datasets.

    Args:
        dataset_ids: List of dataset IDs to download
        version: Dataset version
        cache: Whether to use cache
        force_download: Whether to force re-download
        sandbox: Whether to use sandbox
        quiet: Whether to suppress messages

    Returns:
        OpenESMDatasetList containing the downloaded datasets
    """
    datasets = {}

    for dataset_id in dataset_ids:
        if not quiet:
            msg_info(f"Downloading dataset {dataset_id}")

        # download individual dataset in quiet mode
        dataset = get_dataset(
            dataset_id,
            version=version,
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
    dataset_list = OpenESMDatasetList(datasets)

    # print summary unless silenced
    if not quiet:
        print(dataset_list)

    return dataset_list
