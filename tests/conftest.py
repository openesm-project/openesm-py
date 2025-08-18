"""Pytest configuration and fixtures for openesm tests."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Sample dataset metadata for testing."""
    return {
        "dataset_id": "0001",
        "first_author": "Smith",
        "year": 2023,
        "reference_a": "Smith, J. et al. (2023). Sample study.",
        "reference_b": None,
        "paper_doi": "10.1000/sample",
        "zenodo_doi": "10.5072/zenodo.123456",
        "license": "CC BY 4.0",
        "n_participants": 100,
        "n_time_points": 50,
        "topics": "mood, stress",
        "features": [
            {"name": "mood", "type": "numeric"},
            {"name": "stress", "type": "numeric"}
        ],
        "additional_comments": "Sample dataset for testing; Multiple notes here"
    }
