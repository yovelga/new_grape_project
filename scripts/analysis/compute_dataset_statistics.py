from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class PixelCsvStats:
    path: str
    size_mb: float
    total_rows: int
    label_counts: dict[str, int]
    unique_json_files: int
    unique_hs_dir: int


def _read_csv_stream_counts(
    csv_path: Path,
    *,
    label_col: str = "label",
    extra_cols: Iterable[str] = ("json_file", "hs_dir"),
    chunksize: int = 1_000_000,
) -> PixelCsvStats:
    counts: Counter[str] = Counter()
    total_rows = 0
    unique: dict[str, set[str]] = {c: set() for c in extra_cols}

    usecols = [label_col, *extra_cols]

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        total_rows += len(chunk)
        counts.update(chunk[label_col].astype(str).value_counts().to_dict())
        for col in extra_cols:
            unique[col].update(chunk[col].astype(str).unique().tolist())

    return PixelCsvStats(
        path=str(csv_path),
        size_mb=round(csv_path.stat().st_size / 1024 / 1024, 2),
        total_rows=total_rows,
        label_counts=dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        unique_json_files=len(unique.get("json_file", set())),
        unique_hs_dir=len(unique.get("hs_dir", set())),
    )


def _full_image_stats(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # Heuristic: detect common column names across variants
    label_col = "label" if "label" in df.columns else None
    grape_id_col = None
    for candidate in ("grape_id", "grapeId", "id", "cluster_id", "clusterId"):
        if candidate in df.columns:
            grape_id_col = candidate
            break

    date_col = None
    for candidate in ("week_date", "date", "week", "timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break

    path_col = None
    for candidate in ("path", "img_path", "image_path", "hs_dir", "rgb_path"):
        if candidate in df.columns:
            path_col = candidate
            break

    out: dict[str, Any] = {
        "path": str(csv_path),
        "size_kb": round(csv_path.stat().st_size / 1024, 2),
        "n_samples": int(len(df)),
        "columns": cols,
    }

    if label_col is not None:
        out["label_counts"] = df[label_col].astype(str).value_counts().to_dict()

    if grape_id_col is not None:
        out["n_unique_grapes"] = int(df[grape_id_col].astype(str).nunique())

    if date_col is not None:
        out["n_unique_dates"] = int(df[date_col].astype(str).nunique())

    if grape_id_col is not None and date_col is not None:
        out["n_unique_grape_date_pairs"] = int(
            df[[grape_id_col, date_col]].astype(str).drop_duplicates().shape[0]
        )

    if path_col is not None:
        out["n_unique_paths"] = int(df[path_col].astype(str).nunique())

    return out


def _temporal_stats(tags_csv: Path, tags_xlsx: Path | None) -> dict[str, Any]:
    # HSI_tags.csv is a tag x date presence table.
    raw = pd.read_csv(tags_csv)
    # first column is tag\date
    date_cols = [c for c in raw.columns if c != raw.columns[0]]

    out: dict[str, Any] = {
        "tags_csv": str(tags_csv),
        "n_tags": int(len(raw)),
        "n_timepoints": int(len(date_cols)),
        "timepoints": date_cols,
    }

    if tags_xlsx is not None and tags_xlsx.exists():
        try:
            df = pd.read_excel(tags_xlsx)
            out["tags_xlsx"] = str(tags_xlsx)
            out["tags_xlsx_columns"] = list(df.columns)
            # Try to find irrigation color column
            irr_col = None
            for candidate in ("irrigation_color", "IrrigationColor", "color", "Color"):
                if candidate in df.columns:
                    irr_col = candidate
                    break
            if irr_col is not None:
                out["irrigation_color_counts"] = (
                    df[irr_col].astype(str).value_counts().to_dict()
                )
        except Exception as e:  # noqa: BLE001
            out["tags_xlsx_error"] = f"{type(e).__name__}: {e}"

    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    # Pixel-level CSVs (large)
    pixel_bin = repo / "src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results.csv"
    pixel_mc = repo / "src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results_multiclass_2026-01-16.csv"

    # Full-image split CSVs (user confirmed these are used)
    full_image_dir = repo / "src/models/classification/full_image/inference_to_see_results_of_models_feature_selection/data"
    full_image_csvs = [
        full_image_dir / "val_row1_early copy.csv",
        full_image_dir / "val_row1_late.csv",
        full_image_dir / "test_row2_early copy.csv",
        full_image_dir / "test_row2_late.csv",
    ]

    # Temporal metadata
    tags_csv = repo / "data/raw/HSI_tags.csv"
    tags_xlsx = repo / "data/raw/tags.xlsx"

    results: dict[str, Any] = {"repo": str(repo)}

    print("[1/3] Computing pixel-level CSV stats (streaming)...")
    results["pixel_binary"] = _read_csv_stream_counts(pixel_bin).__dict__
    results["pixel_multiclass"] = _read_csv_stream_counts(pixel_mc).__dict__

    print("[2/3] Computing full-image split stats...")
    results["full_image"] = {
        p.name: _full_image_stats(p) for p in full_image_csvs if p.exists()
    }

    print("[3/3] Computing temporal stats from tag tables...")
    results["temporal"] = _temporal_stats(tags_csv, tags_xlsx)

    print("\n=== DATASET STATISTICS (JSON) ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
