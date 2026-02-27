"""
Full-image dataset builder (single-file, single-config).

Goal
----
Generate EXACTLY 5 CSV files for full-image crack detection from a single pipeline.
- Train: Row 1
  - Positive (label=1): ONLY images from TXT file (trusted crack positives)
  - Negative (label=0): ONLY images with week_date < 01.09.24 (STRICT constraint)
- Val: Row 1 (two variants - EARLY and LATE)
  - Positive grapes: ONE sample per grape (earliest/latest TXT week for that grape, label=1)
  - Negative grapes: MULTIPLE samples per grape (N_NEGATIVE_WEEKS_PER_GRAPE weeks sampled deterministically, label=0)
  - CRITICAL: Both val_early and val_late have SAME negative samples (same grapes, same weeks)
  - Split by grape_id to prevent leakage between train/val
- Test: Row 2 (two variants)
  - EARLY: for each grape_id choose the FIRST crack week. If never cracked -> random week < 01.09.24, label=0
  - LATE:  for each grape_id choose the LAST  crack week. If never cracked -> random week < 01.09.24, label=0

Inputs
------
1) Excel with week columns (dd.mm.yy) and a "Grape ID" column (can have duplicate grape IDs).
2) TXT file with Windows paths that point to *positive* row-1 grape folders:
     ...\raw\<grape_id>\<week_date>
   Those (grape_id, week_date) pairs are treated as the ONLY positives for row 1.

Outputs (CSV)
-------------
- train_row1.csv
- val_row1_early.csv
- val_row1_late.csv
- test_row2_early.csv
- test_row2_late.csv

Each CSV contains: grape_id, row, week_date, label, image_path

GLOBAL CONSTRAINT: All label=0 samples must have week_date < 01.09.24 (exclusive)
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
_PROJECT_ROOT = Path(__file__).resolve().parents[5]


# =========================
# Global Constants
# =========================

# CRITICAL: All negative samples (label=0) must come from weeks BEFORE this date
NEGATIVE_MAX_DATE_EXCLUSIVE = dt.date(2024, 9, 1)  # 01.09.24

# Seed for deterministic random sampling
RANDOM_SEED = 42

# Number of negative weeks to sample per negative grape in validation sets
N_NEGATIVE_WEEKS_PER_GRAPE = 6


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
    base_raw_dir: str = str(_PROJECT_ROOT / r"data/raw")  # Absolute path to raw data directory

    # Dates
    august_cutoff: str = "01.08.24"   # dd.mm.yy: used for identifying crack consideration period

    # Splits
    val_ratio: float = 0.3

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
    base_raw_dir=str(_PROJECT_ROOT / r"data/raw"),
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


def train_val_split_by_cluster_id(cluster_ids: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    """Return (train_grapes, val_grapes) split by cluster_id."""
    rng = np.random.default_rng(RANDOM_SEED)
    cids = np.array(sorted(set(cluster_ids)))
    rng.shuffle(cids)

    n_val = int(round(len(cids) * val_ratio))
    val_grapes = cids[:n_val].tolist()
    train_grapes = cids[n_val:].tolist()

    return train_grapes, val_grapes


def get_negative_pool_weeks(week_cols: List[str]) -> List[str]:
    """Return weeks that are strictly before NEGATIVE_MAX_DATE_EXCLUSIVE."""
    return [w for w in week_cols if parse_date(w) < NEGATIVE_MAX_DATE_EXCLUSIVE]


def sample_random_week_for_grape(grape_id: str, week_pool: List[str]) -> str:
    """Deterministically sample one week for a grape from pool."""
    rng = np.random.default_rng(RANDOM_SEED + hash(grape_id) % (2**31))
    return rng.choice(week_pool)


def sample_multiple_weeks_for_grape(grape_id: str, week_pool: List[str], n: int) -> List[str]:
    """Deterministically sample up to n weeks for a grape from pool (without replacement)."""
    rng = np.random.default_rng(RANDOM_SEED + hash(grape_id) % (2**31))
    available = min(n, len(week_pool))
    return rng.choice(week_pool, size=available, replace=False).tolist()


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ Saved: {out_path} ({len(df)} rows)")


# =========================
# Builders
# =========================

def build_row1_train_dataset(
    train_grapes: List[str],
    week_cols: List[str],
    crack_pairs: Set[Tuple[str, str]],
    base_dir: str,
) -> pd.DataFrame:
    """Build train_row1.csv with negatives only from weeks < NEGATIVE_MAX_DATE_EXCLUSIVE."""
    negative_weeks = get_negative_pool_weeks(week_cols)
    crack_pairs_row1 = {(gid, wk) for (gid, wk) in crack_pairs if extract_row_from_grape_id(gid) == 1}

    records: Dict[Tuple[str, str], dict] = {}

    # Add all negatives from valid week pool
    for gid in train_grapes:
        for wk in negative_weeks:
            records[(gid, wk)] = {
                "grape_id": gid,
                "row": 1,
                "week_date": wk,
                "label": 0,
                "image_path": build_image_path(base_dir, gid, wk),
            }

    # Override with TXT positives (only for train grapes)
    for gid, wk in crack_pairs_row1:
        if gid in train_grapes:
            records[(gid, wk)] = {
                "grape_id": gid,
                "row": 1,
                "week_date": wk,
                "label": 1,
                "image_path": build_image_path(base_dir, gid, wk),
            }

    return pd.DataFrame(list(records.values()))


def build_row1_val_early_late(
    val_grapes: List[str],
    week_cols: List[str],
    crack_pairs: Set[Tuple[str, str]],
    base_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build val_row1_early.csv and val_row1_late.csv.
    
    OPTION 3 implementation:
    - Positives: ONE sample per grape (earliest/latest TXT week)
    - Negatives: MULTIPLE samples per grape (N_NEGATIVE_WEEKS_PER_GRAPE weeks sampled deterministically)
    - CRITICAL: Both early and late use the SAME negative samples (same grapes, same weeks)
    """
    negative_weeks = get_negative_pool_weeks(week_cols)
    crack_pairs_row1 = {(gid, wk) for (gid, wk) in crack_pairs if extract_row_from_grape_id(gid) == 1}
    
    # Group crack pairs by grape_id
    crack_by_grape: Dict[str, List[str]] = {}
    for gid, wk in crack_pairs_row1:
        if gid not in crack_by_grape:
            crack_by_grape[gid] = []
        crack_by_grape[gid].append(wk)
    
    # Sort weeks for each grape
    for gid in crack_by_grape:
        crack_by_grape[gid] = sorted(crack_by_grape[gid], key=parse_date)
    
    early_records = []
    late_records = []
    
    # Pre-compute negative samples for ALL negative grapes (same for both early and late)
    negative_samples: Dict[str, List[str]] = {}
    for gid in val_grapes:
        if gid not in crack_by_grape:
            # Sample N_NEGATIVE_WEEKS_PER_GRAPE weeks for this negative grape
            sampled_weeks = sample_multiple_weeks_for_grape(gid, negative_weeks, N_NEGATIVE_WEEKS_PER_GRAPE)
            negative_samples[gid] = sampled_weeks
    
    # Build records
    for gid in val_grapes:
        if gid in crack_by_grape:
            # Positive grape - ONE sample per grape (early = earliest, late = latest)
            weeks = crack_by_grape[gid]
            early_week = weeks[0]
            late_week = weeks[-1]
            
            early_records.append({
                "grape_id": gid,
                "row": 1,
                "week_date": early_week,
                "label": 1,
                "image_path": build_image_path(base_dir, gid, early_week),
            })
            late_records.append({
                "grape_id": gid,
                "row": 1,
                "week_date": late_week,
                "label": 1,
                "image_path": build_image_path(base_dir, gid, late_week),
            })
        else:
            # Negative grape - MULTIPLE samples (SAME for both early and late)
            sampled_weeks = negative_samples[gid]
            
            for week in sampled_weeks:
                record = {
                    "grape_id": gid,
                    "row": 1,
                    "week_date": week,
                    "label": 0,
                    "image_path": build_image_path(base_dir, gid, week),
                }
                early_records.append(record.copy())
                late_records.append(record.copy())
    
    return pd.DataFrame(early_records), pd.DataFrame(late_records)


def build_row2_test_early_late(
    df_excel: pd.DataFrame,
    week_cols: List[str],
    cutoff: str,
    base_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row2_test_early, row2_test_late).

    Aggregates duplicate grape IDs by "any cracked in that week".
    For negatives (no crack), sample random week from negative pool < NEGATIVE_MAX_DATE_EXCLUSIVE.
    """
    cutoff_date = parse_date(cutoff)
    consider_weeks = [w for w in week_cols if parse_date(w) >= cutoff_date]
    negative_weeks = get_negative_pool_weeks(week_cols)

    # Filter row2 only, normalize grape_id
    df = df_excel.copy()
    df["grape_id"] = df["grape_id"].astype(str).str.strip()
    df["row"] = df["grape_id"].apply(extract_row_from_grape_id)
    df = df[df["row"] == 2].copy()

    # Aggregate duplicates by "any cracked"
    df_agg = df.groupby("grape_id")[consider_weeks].agg(lambda s: int(any(is_cracked_cell(v) for v in s))).reset_index()

    def pick_week(row, mode: str, grape_id: str) -> Tuple[str, int]:
        cracked_weeks = [w for w in consider_weeks if int(row[w]) == 1]
        if cracked_weeks:
            chosen = min(cracked_weeks, key=parse_date) if mode == "early" else max(cracked_weeks, key=parse_date)
            return chosen, 1
        # No crack - sample random week from negative pool
        fallback_week = sample_random_week_for_grape(grape_id, negative_weeks)
        return fallback_week, 0

    early_records = []
    late_records = []

    for _, r in df_agg.iterrows():
        gid = r["grape_id"]
        w_early, y_early = pick_week(r, "early", gid)
        w_late, y_late = pick_week(r, "late", gid)

        early_records.append({
            "grape_id": gid,
            "row": 2,
            "week_date": w_early,
            "label": y_early,
            "image_path": build_image_path(base_dir, gid, w_early),
        })
        late_records.append({
            "grape_id": gid,
            "row": 2,
            "week_date": w_late,
            "label": y_late,
            "image_path": build_image_path(base_dir, gid, w_late),
        })

    df_early = pd.DataFrame(early_records).drop_duplicates(subset=["grape_id"], keep="first")
    df_late = pd.DataFrame(late_records).drop_duplicates(subset=["grape_id"], keep="first")
    return df_early, df_late


# =========================
# Main
# =========================

def main() -> None:
    print("=" * 80)
    print("Full-image dataset builder - Generates EXACTLY 5 CSV files")
    print("=" * 80)
    print(f"NEGATIVE_MAX_DATE_EXCLUSIVE = {NEGATIVE_MAX_DATE_EXCLUSIVE.strftime('%d.%m.%y')}")
    print(f"RANDOM_SEED = {RANDOM_SEED}")
    print(f"N_NEGATIVE_WEEKS_PER_GRAPE = {N_NEGATIVE_WEEKS_PER_GRAPE}")
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

    # Split row1 grapes into train/val
    train_grapes, val_grapes = train_val_split_by_cluster_id(row1_grapes, CFG.val_ratio)
    print(f"- Train grapes: {len(train_grapes)}")
    print(f"- Val grapes: {len(val_grapes)}")

    # Build row1 train dataset
    row1_train = build_row1_train_dataset(
        train_grapes=train_grapes,
        week_cols=week_cols,
        crack_pairs=crack_pairs,
        base_dir=CFG.base_raw_dir,
    )

    # Build row1 val datasets (early and late)
    val_early, val_late = build_row1_val_early_late(
        val_grapes=val_grapes,
        week_cols=week_cols,
        crack_pairs=crack_pairs,
        base_dir=CFG.base_raw_dir,
    )

    # Build row2 test datasets
    row2_early, row2_late = build_row2_test_early_late(
        df_excel=df_excel,
        week_cols=week_cols,
        cutoff=CFG.august_cutoff,
        base_dir=CFG.base_raw_dir,
    )

    # Save outputs - EXACTLY 5 CSV files
    out_dir = CFG.out_dir
    
    save_csv(row1_train, out_dir / "train_row1.csv")
    save_csv(val_early, out_dir / "val_row1_early.csv")
    save_csv(val_late, out_dir / "val_row1_late.csv")
    save_csv(row2_early, out_dir / "test_row2_early.csv")
    save_csv(row2_late, out_dir / "test_row2_late.csv")

    # Validation report
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    # Row 1 validation
    n_pos_train = len(row1_train[row1_train["label"] == 1])
    n_neg_train = len(row1_train[row1_train["label"] == 0])
    n_unique_train = row1_train["grape_id"].nunique()
    
    n_pos_val_early = len(val_early[val_early["label"] == 1])
    n_neg_val_early = len(val_early[val_early["label"] == 0])
    n_unique_val_early = val_early["grape_id"].nunique()
    
    n_pos_val_late = len(val_late[val_late["label"] == 1])
    n_neg_val_late = len(val_late[val_late["label"] == 0])
    n_unique_val_late = val_late["grape_id"].nunique()
    
    print(f"\nRow 1 Train: {len(row1_train)} total | {n_unique_train} unique grapes | {n_pos_train} positive | {n_neg_train} negative")
    print(f"Row 1 Val Early: {len(val_early)} total | {n_unique_val_early} unique grapes | {n_pos_val_early} positive | {n_neg_val_early} negative")
    print(f"Row 1 Val Late:  {len(val_late)} total | {n_unique_val_late} unique grapes | {n_pos_val_late} positive | {n_neg_val_late} negative")
    
    # Row 2 validation
    n_pos_early = len(row2_early[row2_early["label"] == 1])
    n_neg_early = len(row2_early[row2_early["label"] == 0])
    n_unique_early = row2_early["grape_id"].nunique()
    
    n_pos_late = len(row2_late[row2_late["label"] == 1])
    n_neg_late = len(row2_late[row2_late["label"] == 0])
    n_unique_late = row2_late["grape_id"].nunique()
    
    print(f"\nRow 2 Test Early: {len(row2_early)} total | {n_unique_early} unique grapes | {n_pos_early} positive | {n_neg_early} negative")
    print(f"Row 2 Test Late:  {len(row2_late)} total | {n_unique_late} unique grapes | {n_pos_late} positive | {n_neg_late} negative")
    
    # Check no grape_id overlap between train and val
    train_set = set(row1_train["grape_id"].unique())
    val_early_set = set(val_early["grape_id"].unique())
    val_late_set = set(val_late["grape_id"].unique())
    
    overlap_early = train_set & val_early_set
    overlap_late = train_set & val_late_set
    
    print("\n" + "-" * 80)
    print("GRAPE_ID OVERLAP CHECK")
    print("-" * 80)
    print(f"Train vs Val Early overlap: {len(overlap_early)} grapes")
    print(f"Train vs Val Late overlap:  {len(overlap_late)} grapes")
    
    if overlap_early or overlap_late:
        print("⚠ WARNING: Found grape_id overlap between train and val!")
    else:
        print("✓ No grape_id overlap between train and val sets")
    
    # Verify negative date constraint
    print("\n" + "-" * 80)
    print("NEGATIVE DATE CONSTRAINT CHECK")
    print("-" * 80)
    
    all_dfs = {
        "train_row1": row1_train,
        "val_row1_early": val_early,
        "val_row1_late": val_late,
        "test_row2_early": row2_early,
        "test_row2_late": row2_late,
    }
    
    constraint_violations = False
    for name, df in all_dfs.items():
        negatives = df[df["label"] == 0]
        if len(negatives) > 0:
            invalid = negatives[negatives["week_date"].apply(lambda w: parse_date(w) >= NEGATIVE_MAX_DATE_EXCLUSIVE)]
            if len(invalid) > 0:
                print(f"⚠ {name}: {len(invalid)} negatives violate date constraint!")
                constraint_violations = True
            else:
                print(f"✓ {name}: All {len(negatives)} negatives satisfy date constraint")
    
    if not constraint_violations:
        print("✓ All negative samples satisfy week_date < 01.09.24")
    
    # Verify val_early and val_late have same negatives
    print("\n" + "-" * 80)
    print("VAL NEGATIVE CONSISTENCY CHECK")
    print("-" * 80)
    
    val_early_neg = val_early[val_early["label"] == 0][["grape_id", "week_date"]].sort_values(["grape_id", "week_date"]).reset_index(drop=True)
    val_late_neg = val_late[val_late["label"] == 0][["grape_id", "week_date"]].sort_values(["grape_id", "week_date"]).reset_index(drop=True)
    
    if val_early_neg.equals(val_late_neg):
        print(f"✓ val_row1_early and val_row1_late have IDENTICAL negative samples ({len(val_early_neg)} samples)")
    else:
        print(f"⚠ WARNING: val_row1_early and val_row1_late have DIFFERENT negative samples!")
        print(f"  Early negatives: {len(val_early_neg)} | Late negatives: {len(val_late_neg)}")
    
    # Check negative grapes consistency
    early_neg_grapes = set(val_early_neg["grape_id"].unique())
    late_neg_grapes = set(val_late_neg["grape_id"].unique())
    
    if early_neg_grapes == late_neg_grapes:
        print(f"✓ Both val sets have same negative grapes ({len(early_neg_grapes)} grapes)")
    else:
        print(f"⚠ Different negative grapes: Early={len(early_neg_grapes)} vs Late={len(late_neg_grapes)}")
    
    # CRITICAL: Verify no image path overlap between train and val
    print("\n" + "-" * 80)
    print("IMAGE PATH OVERLAP CHECK (Train vs Val)")
    print("-" * 80)
    
    train_image_paths = set(row1_train["image_path"].unique())
    val_early_image_paths = set(val_early["image_path"].unique())
    val_late_image_paths = set(val_late["image_path"].unique())
    
    overlap_train_early = train_image_paths & val_early_image_paths
    overlap_train_late = train_image_paths & val_late_image_paths
    
    print(f"Train unique images: {len(train_image_paths)}")
    print(f"Val Early unique images: {len(val_early_image_paths)}")
    print(f"Val Late unique images: {len(val_late_image_paths)}")
    print(f"Train vs Val Early overlap: {len(overlap_train_early)} images")
    print(f"Train vs Val Late overlap: {len(overlap_train_late)} images")
    
    if overlap_train_early or overlap_train_late:
        print("⚠ WARNING: Found IMAGE OVERLAP between train and val!")
        if overlap_train_early:
            print(f"  Overlapping paths (train vs val_early): {list(overlap_train_early)[:5]}...")
        if overlap_train_late:
            print(f"  Overlapping paths (train vs val_late): {list(overlap_train_late)[:5]}...")
    else:
        print("✓ NO IMAGE OVERLAP: Train and val use completely different images")
    
    print("\n" + "=" * 80)
    print("✓ Successfully generated EXACTLY 5 CSV files")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
