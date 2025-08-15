#!/usr/bin/env python3
"""Quick test of the utility functions."""

from src.openesm.utils import (
    cache_info,
    clear_cache, 
    get_cache_dir,
    get_cache_path,
    msg_info,
    msg_success,
    msg_warn,
    process_specific_metadata
)

def test_basic_functions():
    """Test basic utility functions."""
    print("Testing utility functions...")
    
    # Test cache directory creation
    cache_dir = get_cache_dir()
    print(f"Cache directory: {cache_dir}")
    
    # Test cache info
    cache_info()
    
    # Test messaging
    msg_info("This is an info message")
    msg_warn("This is a warning message") 
    msg_success("This is a success message")
    
    # Test cache path construction
    cache_path = get_cache_path("test_dataset", "v1.0", "data.tsv", "data")
    print(f"Cache path: {cache_path}")
    
    # Test metadata processing
    test_metadata = {
        "dataset_id": "test_001",
        "first_author": "Doe",
        "year": 2023,
        "n_participants": 100,
        "features": [
            {"name": "feature1", "type": "numeric"},
            {"name": "feature2", "type": "categorical"}
        ]
    }
    
    processed = process_specific_metadata(test_metadata)
    print(f"Processed metadata keys: {list(processed.keys())}")
    print(f"Features type: {type(processed['features'])}")
    
    print("All tests completed!")

if __name__ == "__main__":
    test_basic_functions()
