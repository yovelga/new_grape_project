"""
Utility to dynamically resolve the project root directory.

Usage in any Python file:
    from project_root import PROJECT_ROOT

    DATA_PATH = PROJECT_ROOT / "data" / "raw"
    RESULTS_PATH = PROJECT_ROOT / "results" / "my_results"

This makes all paths portable regardless of where the project folder is located.
"""
from pathlib import Path

# Always resolves to the directory containing this file (i.e. the project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent
