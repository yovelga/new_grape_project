"""
Full-image dataset builder (single-file, single-config).

Goal
----
Generate EXACTLY 4 CSV files for full-image crack detection from a single pipeline.
- Train/Val: Row 1
  - Positive (label=1): ONLY images from TXT file (trusted crack positives)
  - Negative (label=0): ONLY pre-August images (before 01.08.24 - no cracks exist)
  - Train/Val split by grape_id to prevent leakage across weeks
- Test: Row 2 (two variants)
  - EARLY: for each grape_id choose the FIRST crack week (>= cutoff). If never cracked -> fallback_week (01.08.24) label=0
  - LATE:  for each grape_id choose the LAST  crack week (>= cutoff). If never cracked -> fallback_week (01.08.24) label=0

Inputs
------
1) Excel with week columns (dd.mm.yy) and a "Grape ID" column (can have duplicate grape IDs).
2) TXT file with Windows paths that point to *positive* row-1 grape folders:
     ...\raw\<grape_id>\<week_date>
   Those (grape_id, week_date) pairs are treated as the ONLY positives for row 1.

Outputs (CSV)
-------------
- train_row1.csv
- val_row1.csv
- test_row2_early.csv
- test_row2_late.csv

Each CSV contains: cluster_id (grape_id), row_id (row), date (week_date), label, image_path
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import List, Optional, Set, Tuple, Dict
import re
import sys
import pandas as pd
import numpy as np
import datetime as dt


# =========================
# Configuration (EDIT HERE)
# =========================

@dataclass(frozen=True)
class Config:
    # Inputs (required fields first)
    excel_path: Path
    txt_row1_crack_list_path: Path
    
    # Inputs (optional fields)
    sheet_name: Optional[int | str] = 0  # 0 -> first sheet
    grape_id_col: str = "Grape ID"
    base_raw_dir: str = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"  # Absolute path to raw data directory

    # Dates
    august_cutoff: str = "01.08.24"   # dd.mm.yy: pre-August < cutoff ; August+ >= cutoff
    test_fallback_week: str = "01.08.24"  # used for row2 grapes with no crack

    # Splits
    val_ratio: float = 0.20
    seed: int = 42

    # Which vineyard rows
    trainval_row: int = 1
    test_row: int = 2

    # Output
    out_dir: Path = Path("dataset_csvs_final")


# Determine script directory for relative paths
SCRIPT_DIR = Path(__file__).parent

CFG = Config(
    excel_path=SCRIPT_DIR / "crack_by_weeks_for_dataset.xlsx",
    txt_row1_crack_list_path=SCRIPT_DIR / "row_1_taged_with_crack_112_clusters.txt",
    base_raw_dir=r"C:\Users\yovel\Desktop\Grape_Project\data\raw",
    out_dir=SCRIPT_DIR / "dataset_csvs_final",
)


# =========================
# Helpers
# =========================

_WEEK_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")


def parse_date(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%d.%m.%y").date()


def is_week_column(col_name: str) -> bool:
    return bool(_WEEK_RE.match(str(col_name).strip()))


def sort_week_cols(week_cols: List[str]) -> List[str]:
    return sorted(week_cols, key=lambda x: parse_date(x))


def extract_row_from_grape_id(grape_id: str) -> Optional[int]:
    try:
        return int(str(grape_id).split("_")[0])
    except Exception:
        return None


def is_cracked_cell(v) -> bool:
    """Excel crack indicator.
    Current file seems to use 1 for crack and NaN for not-crack.
    This function is generic: non-empty and non-zero => cracked.
    """
    if pd.isna(v):
        return False
    if isinstance(v, str):
        t = v.strip()
        if t == "" or t == "0":
            return False
        try:
            return float(t) != 0.0
        except Exception:
            return True
    try:
        return float(v) != 0.0
    except Exception:
        return True


def parse_crack_txt_pairs(txt_path: Path) -> Set[Tuple[str, str]]:
    """Parse a TXT list of Windows paths:
       ...\\raw\\<grape_id>\\<week_date>
    Returns a set of (grape_id, week_date).
    """
    crack = set()
    lines = [l.strip() for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for line in lines:
        p = PureWindowsPath(line)
        parts = p.parts
        raw_idx = None
        for i, part in enumerate(parts):
            if str(part).lower() == "raw":
                raw_idx = i
                break
        if raw_idx is None or raw_idx + 2 >= len(parts):
            print(f"WARNING: Could not parse line: {line}")
            continue

        grape_id = parts[raw_idx + 1]
        week_date = parts[raw_idx + 2]
        crack.add((str(grape_id), str(week_date)))

    return crack


def build_image_path(base_dir: str, grape_id: str, week_date: str) -> str:
    """Build absolute Windows path to image directory."""
    return str(Path(base_dir) / grape_id / week_date)


def train_val_split_by_cluster_id(cluster_ids: List[str], val_ratio: float, seed: int) -> Dict[str, str]:
    """Return mapping cluster_id -> 'train'/'val'."""
    rng = np.random.default_rng(seed)
    cids = np.array(sorted(set(cluster_ids)))
    rng.shuffle(cids)

    n_val = int(round(len(cids) * val_ratio))
    val_set = set(cids[:n_val].tolist())

    return {cid: ("val" if cid in val_set else "train") for cid in cids.tolist()}


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ Saved: {out_path} ({len(df)} rows)")


# =========================
# Builders
# =========================

def build_row1_trainval_dataset(
    row1_grapes: List[str],
    week_cols: List[str],
    crack_pairs: Set[Tuple[str, str]],
    cutoff: str,
    base_dir: str,
) -> pd.DataFrame:
    """Return row1_full dataset with ONLY pre-August negatives and TXT positives."""
    cutoff_date = parse_date(cutoff)
    pre_weeks = [w for w in week_cols if parse_date(w) < cutoff_date]

    crack_pairs_row1 = {(gid, wk) for (gid, wk) in crack_pairs if extract_row_from_grape_id(gid) == 1}

    # Build records: all pre-August negatives + ONLY crack_pairs positives (override duplicates)
    records: Dict[Tuple[str, str], dict] = {}

    for gid in row1_grapes:
        for wk in pre_weeks:
            records[(gid, wk)] = {
                "grape_id": gid,
                "row": 1,
                "week_date": wk,
                "label": 0,
                "image_path": build_image_path(base_dir, gid, wk),
                "source": "pre_august",
            }

    for gid, wk in crack_pairs_row1:
        records[(gid, wk)] = {
            "grape_id": gid,
            "row": 1,
            "week_date": wk,
            "label": 1,
            "image_path": build_image_path(base_dir, gid, wk),
            "source": "txt_positive",
        }

    return pd.DataFrame(list(records.values()))


def build_row2_test_early_late(
    df_excel: pd.DataFrame,
    week_cols: List[str],
    cutoff: str,
    fallback_week: str,
    base_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row2_test_early, row2_test_late).

    Aggregates duplicate grape IDs by "any cracked in that week".
    """
    cutoff_date = parse_date(cutoff)
    consider_weeks = [w for w in week_cols if parse_date(w) >= cutoff_date]

    # Filter row2 only, normalize grape_id
    df = df_excel.copy()
    df["grape_id"] = df["grape_id"].astype(str).str.strip()
    df["row"] = df["grape_id"].apply(extract_row_from_grape_id)
    df = df[df["row"] == 2].copy()

    # Aggregate duplicates by "any cracked"
    df_agg = df.groupby("grape_id")[consider_weeks].agg(lambda s: int(any(is_cracked_cell(v) for v in s))).reset_index()

    def pick_week(row, mode: str) -> Tuple[str, int]:
        cracked_weeks = [w for w in consider_weeks if int(row[w]) == 1]
        if cracked_weeks:
            chosen = min(cracked_weeks, key=parse_date) if mode == "early" else max(cracked_weeks, key=parse_date)
            return chosen, 1
        return fallback_week, 0

    early_records = []
    late_records = []

    for _, r in df_agg.iterrows():
        gid = r["grape_id"]
        w_early, y_early = pick_week(r, "early")
        w_late, y_late = pick_week(r, "late")

        early_records.append({
            "grape_id": gid,
            "row": 2,
            "week_date": w_early,
            "label": y_early,
            "image_path": build_image_path(base_dir, gid, w_early),
            "mode": "early",
        })
        late_records.append({
            "grape_id": gid,
            "row": 2,
            "week_date": w_late,
            "label": y_late,
            "image_path": build_image_path(base_dir, gid, w_late),
            "mode": "late",
        })

    df_early = pd.DataFrame(early_records).drop_duplicates(subset=["grape_id"], keep="first")
    df_late = pd.DataFrame(late_records).drop_duplicates(subset=["grape_id"], keep="first")
    return df_early, df_late


# =========================
# Main
# =========================

def main() -> None:
    print("=" * 80)
    print("Full-image dataset builder - Generates EXACTLY 4 CSV files")
    print("=" * 80)

    # Read Excel
    if not CFG.excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {CFG.excel_path}")

    df_excel = pd.read_excel(CFG.excel_path, sheet_name=CFG.sheet_name)
    df_excel.columns = [str(c).strip() for c in df_excel.columns]

    if CFG.grape_id_col not in df_excel.columns:
        raise ValueError(f"Grape ID col '{CFG.grape_id_col}' not found. Available: {list(df_excel.columns)}")

    # Detect and sort week columns
    week_cols = sort_week_cols([c for c in df_excel.columns if is_week_column(c)])
    if not week_cols:
        raise ValueError("No week columns detected (expected dd.mm.yy).")

    # Normalize grape_id
    df_excel["grape_id"] = df_excel[CFG.grape_id_col].astype(str).str.strip()
    df_excel["row"] = df_excel["grape_id"].apply(extract_row_from_grape_id)

    # Row lists
    row1_grapes = sorted(df_excel.loc[df_excel["row"] == CFG.trainval_row, "grape_id"].dropna().unique().tolist())

    print(f"- Excel rows: {len(df_excel)}")
    print(f"- Week columns detected: {len(week_cols)} ({week_cols[0]} ... {week_cols[-1]})")
    print(f"- Row1 grapes detected: {len(row1_grapes)}")

    # Parse TXT positives
    if not CFG.txt_row1_crack_list_path.exists():
        raise FileNotFoundError(f"TXT crack list not found: {CFG.txt_row1_crack_list_path}")

    crack_pairs = parse_crack_txt_pairs(CFG.txt_row1_crack_list_path)
    print(f"- TXT crack pairs parsed: {len(crack_pairs)}")

    # Build row1 dataset
    row1_full = build_row1_trainval_dataset(
        row1_grapes=row1_grapes,
        week_cols=week_cols,
        crack_pairs=crack_pairs,
        cutoff=CFG.august_cutoff,
        base_dir=CFG.base_raw_dir,
    )

    # Train/Val split by grape_id
    split_map = train_val_split_by_cluster_id(row1_grapes, CFG.val_ratio, CFG.seed)
    row1_full["split"] = row1_full["grape_id"].map(split_map).fillna("train")

    # Build row2 tests
    row2_early, row2_late = build_row2_test_early_late(
        df_excel=df_excel,
        week_cols=week_cols,
        cutoff=CFG.august_cutoff,
        fallback_week=CFG.test_fallback_week,
        base_dir=CFG.base_raw_dir,
    )

    # Save outputs - EXACTLY 4 CSV files
    out_dir = CFG.out_dir
    
    # Row 1: Train and Val (drop internal columns)
    row1_train = row1_full[row1_full["split"] == "train"].drop(columns=["split", "source"])
    row1_val = row1_full[row1_full["split"] == "val"].drop(columns=["split", "source"])
    
    # Row 2: Early and Late (drop mode column)
    row2_early_clean = row2_early.drop(columns=["mode"])
    row2_late_clean = row2_late.drop(columns=["mode"])
    
    save_csv(row1_train, out_dir / "train_row1.csv")
    save_csv(row1_val, out_dir / "val_row1.csv")
    save_csv(row2_early_clean, out_dir / "test_row2_early.csv")
    save_csv(row2_late_clean, out_dir / "test_row2_late.csv")

    # Validation report
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    # Row 1 validation
    n_pos_train = len(row1_train[row1_train["label"] == 1])
    n_neg_train = len(row1_train[row1_train["label"] == 0])
    n_pos_val = len(row1_val[row1_val["label"] == 1])
    n_neg_val = len(row1_val[row1_val["label"] == 0])
    
    print(f"\nRow 1 Train: {len(row1_train)} total ({n_pos_train} positive, {n_neg_train} negative)")
    print(f"Row 1 Val:   {len(row1_val)} total ({n_pos_val} positive, {n_neg_val} negative)")
    
    # Row 2 validation
    n_pos_early = len(row2_early_clean[row2_early_clean["label"] == 1])
    n_neg_early = len(row2_early_clean[row2_early_clean["label"] == 0])
    n_pos_late = len(row2_late_clean[row2_late_clean["label"] == 1])
    n_neg_late = len(row2_late_clean[row2_late_clean["label"] == 0])
    
    print(f"\nRow 2 Test Early: {len(row2_early_clean)} total ({n_pos_early} positive, {n_neg_early} negative)")
    print(f"Row 2 Test Late:  {len(row2_late_clean)} total ({n_pos_late} positive, {n_neg_late} negative)")
    
    print("\n" + "=" * 80)
    print("✓ Successfully generated EXACTLY 4 CSV files")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
