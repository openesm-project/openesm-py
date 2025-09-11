"""Internal utility functions for openesm package."""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
import requests
from platformdirs import user_cache_dir
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_cache_dir(type_: Optional[str] = None) -> Path:
    """Get the openesm cache directory.

    Args:
        type_: Optional subdirectory within the cache

    Returns:
        Path to cache directory
    """
    # user cache directory
    base_cache = Path(user_cache_dir("openesm"))

    # if a type is specified, return the subdirectory
    if type_ is not None:
        cache_dir = base_cache / type_
    else:
        cache_dir = base_cache

    # ensure directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_info() -> None:
    """Display information about the openesm cache.

    Shows the location and total size of the local file cache.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        console.print("[blue]ℹ[/blue] Cache directory does not exist yet.")
        console.print(f"[blue]ℹ[/blue] It will be created at: {cache_dir}")
        return

    # calculate total size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())

    # format size
    def format_bytes(size: int) -> str:
        size_float = float(size)
        for unit in ["B", "KB", "MB", "GB"]:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.1f} TB"

    console.print(f"[blue]ℹ[/blue] Cache location: {cache_dir}")
    console.print(f"[blue]ℹ[/blue] Cache size: {format_bytes(total_size)}")


def clear_cache(force: bool = False) -> None:
    """Clear the openESM cache.

    Removes all cached openESM data from your local machine.

    Args:
        force: If True, will not ask for confirmation before deleting
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        console.print(
            "[blue]ℹ[/blue] Cache directory does not exist. Nothing to clear."
        )
        return

    confirmed = False
    if force:
        confirmed = True
    elif _is_interactive():
        console.print(f"[blue]ℹ[/blue] This will delete all cached data at {cache_dir}")
        response = input("Are you sure you want to proceed? [y/N]: ").lower().strip()
        confirmed = response in ("y", "yes")
    else:
        raise RuntimeError(
            "Cannot ask for confirmation in a non-interactive session. "
            "Use clear_cache(force=True)."
        )

    if confirmed:
        shutil.rmtree(cache_dir)
        console.print("[green]✓[/green] Cache cleared.")
    else:
        console.print("[blue]ℹ[/blue] Cache not cleared.")


def get_metadata_dir() -> Path:
    """Get path to metadata cache."""
    return get_cache_dir(type_="metadata")


def get_data_dir() -> Path:
    """Get path to data cache."""
    return get_cache_dir(type_="data")


def _is_interactive() -> bool:
    """Check if running in interactive mode."""
    return hasattr(__builtins__, "__IPYTHON__") or (
        hasattr(os, "isatty") and os.isatty(0)
    )


def _is_quiet() -> bool:
    """Check if quiet mode is enabled."""
    return os.environ.get("OPENESM_QUIET", "").lower() in ("true", "1", "yes")


def msg_info(message: str) -> None:
    """Display an info message."""
    if not _is_quiet():
        console.print(f"[blue]ℹ[/blue] {message}")


def msg_warn(message: str) -> None:
    """Display a warning message."""
    if not _is_quiet():
        console.print(f"[yellow]⚠[/yellow] {message}")


def msg_success(message: str) -> None:
    """Display a success message."""
    if not _is_quiet():
        console.print(f"[green]✓[/green] {message}")


def read_json_safe(path: Union[str, Path]) -> dict[str, Any]:
    """Read JSON with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If JSON parsing fails
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        raise ValueError(f"Failed to read JSON file at {path}: {e}") from e


def download_with_progress(url: str, destfile: Union[str, Path]) -> bool:
    """Download file with progress indicator.

    Args:
        url: URL to download from
        destfile: Destination file path

    Returns:
        True if successful

    Raises:
        requests.RequestException: If download fails
    """
    destfile = Path(destfile)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console if not _is_quiet() else Console(file=open(os.devnull, "w")),
        ) as progress:
            task = progress.add_task(f"Downloading from {url}", total=None)

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # ensure parent directory exists
            destfile.parent.mkdir(parents=True, exist_ok=True)

            with open(destfile, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            progress.update(task, description="Download completed")

        return True

    except Exception as e:
        raise requests.RequestException(f"Download failed from {url}: {e}") from e


def get_cache_path(
    dataset_id: str, version: str, filename: str, type_: str = "metadata"
) -> Path:
    """Construct dataset cache path.

    Args:
        dataset_id: Dataset identifier
        version: Dataset version
        filename: File name
        type_: Cache type ("metadata" or "data")

    Returns:
        Path to cached file
    """
    if type_ not in ("metadata", "data"):
        raise ValueError("type_ must be 'metadata' or 'data'")

    base_dir = get_metadata_dir() if type_ == "metadata" else get_data_dir()

    # simplified path
    path = base_dir / dataset_id / version / filename

    # ensure the directory for the file exists
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def process_specific_metadata(raw_meta: dict[str, Any]) -> dict[str, Any]:
    """Process specific dataset metadata.

    Helper function to process the raw dict from a specific dataset's
    metadata JSON into a clean, structured dict.

    Args:
        raw_meta: The raw dict parsed from the metadata json file

    Returns:
        A clean metadata dict
    """

    def get_val(field: str, default_type: str = "str") -> Any:
        """Safely get a value, converting None or empty list to appropriate default."""
        val = raw_meta.get(field)
        if (
            val is None
            or (isinstance(val, list) and len(val) == 0)
            or (isinstance(val, dict) and len(val) == 0)
        ):
            if default_type == "str":
                return None
            elif default_type == "int":
                return None
            return None
        if isinstance(val, list) and len(val) > 1:
            return ", ".join(str(v) for v in val)
        elif isinstance(val, list) and len(val) == 1:
            return val[0]
        elif isinstance(val, dict):
            # handle non-empty dicts by converting to string
            return str(val)
        return val

    # process features into polars dataframe if available
    features_df = None
    if raw_meta.get("features") and len(raw_meta["features"]) > 0:
        try:
            features_df = pl.DataFrame(raw_meta["features"])
        except Exception:
            # if features can't be converted to dataframe, store as list
            features_df = raw_meta["features"]

    # create clean metadata dict
    return {
        "dataset_id": get_val("dataset_id"),
        "first_author": get_val("first_author"),
        "year": get_val("year", "int"),
        "reference_a": get_val("reference_a"),
        "reference_b": get_val("reference_b"),
        "paper_doi": get_val("paper_doi"),
        "zenodo_doi": get_val("zenodo_doi"),
        "license": get_val("license"),
        "link_to_data": get_val("link_to_data"),
        "link_to_codebook": get_val("link_to_codebook"),
        "link_to_code": get_val("link_to_code"),
        "n_participants": get_val("n_participants", "int"),
        "n_time_points": get_val("n_time_points", "int"),
        "n_days": get_val("n_days", "int"),
        "n_beeps_per_day": get_val("n_beeps_per_day"),
        "passive_data_available": get_val("passive_data_available"),
        "cross_sectional_available": get_val("cross_sectional_available"),
        "topics": get_val("topics"),
        "implicit_missingness": get_val("implicit_missingness"),
        "raw_time_stamp": get_val("raw_time_stamp"),
        "sampling_scheme": get_val("sampling_scheme"),
        "participants": get_val("participants"),
        "coding_file": get_val("coding_file"),
        "additional_comments": get_val("additional_comments"),
        "features": features_df,
    }
