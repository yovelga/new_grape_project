from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env in project root
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def project_path():
    return Path(os.getenv("PROJECT_PATH", Path(__file__).resolve().parents[2]))


def datasets_path():
    return Path(
        os.getenv(
            "DATASETS_PATH",
            Path(__file__).resolve().parents[2]
            / "src"
            / "preprocessing"
            / "dataset_builder_grapes"
            / "dataset",
        )
    )
