#!/usr/bin/env python3
"""
make_disease_map.py
-------------------
Convert an imageCHD diagnosis CSV into disease_map.json consumed by
DiseaseConditioningMixin, AuxiliaryDiagnosisMixin, and CrossAttentionConditioningMixin.

Output format:
    { "ct_1001": [0, 1, 1, 1, 0, 0, 0, 0], ... }

Disease order (K=8):
    index 0  HLHS
    index 1  ASD
    index 2  VSD
    index 3  AVSD
    index 4  DORV
    index 5  PuA
    index 6  ToF
    index 7  TGA

Supported file formats
----------------------
.xlsx / .xls  — Excel workbook (e.g. imageCHD_dataset_info.xlsx, binary layout)
.csv          — comma-separated (or other delimiter)

Supported CSV/sheet layouts
----------------------------
Layout A — binary columns (one column per disease, None/0/blank = absent, 1 = present):
    index,ASD,VSD,AVSD,ToF,TGA,DORV,PA,...
    1001,1,1,,,,,,
    (actual imageCHD_dataset_info.xlsx format — PA maps to PuA via alias)

Layout A alt — explicit zeros:
    id,HLHS,ASD,VSD,AVSD,DORV,PuA,ToF,TGA
    1001,0,1,1,1,0,0,0,0

Layout B — single string diagnosis column (comma-separated tags per row):
    id,diagnosis
    1001,"ASD,VSD,AVSD"
    ct_1001,Normal

Layout C — one-hot integer index:
    id,type
    1001,2

The script auto-detects the layout from column names.

Usage
-----
    python scripts/make_disease_map.py \
        --csv    /path/to/nnUNet_raw/Dataset030_imageCHD_HU/imageCHD_dataset_info.xlsx \
        --out    /path/to/nnUNet_preprocessed/Dataset030_imageCHD_HU/disease_map.json \
        [--id-col index]               # default: first column
        [--id-prefix ct_]             # prefix to prepend to numeric IDs, default "ct_"
        [--dry-run]                    # print mapping, don't write file

    # Or rely on nnUNet env vars:
    python scripts/make_disease_map.py \
        --dataset-id 30                # reads from $nnUNet_raw / writes to $nnUNet_preprocessed
        [--csv-name imageCHD_dataset_info.xlsx]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Disease order (must match disease_map.json expected by all trainers)
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_ORDER = ["HLHS", "ASD", "VSD", "AVSD", "DORV", "PuA", "ToF", "TGA"]

# Aliases for flexible column-name matching (lowercase → canonical)
ALIASES: dict[str, str] = {
    # HLHS
    "hlhs": "HLHS",
    "hypoplastic left heart": "HLHS",
    "hypoplastic_left_heart": "HLHS",
    # ASD
    "asd": "ASD",
    "atrial septal defect": "ASD",
    "atrial_septal_defect": "ASD",
    # VSD
    "vsd": "VSD",
    "ventricular septal defect": "VSD",
    "ventricular_septal_defect": "VSD",
    # AVSD
    "avsd": "AVSD",
    "av septal defect": "AVSD",
    "atrioventricular septal defect": "AVSD",
    "atrioventricular_septal_defect": "AVSD",
    # DORV
    "dorv": "DORV",
    "double outlet right ventricle": "DORV",
    "double_outlet_right_ventricle": "DORV",
    # PuA
    "pua": "PuA",
    "pa": "PuA",        # context-dependent; prefer PuA when adjacent to others
    "pulmonary atresia": "PuA",
    "pulmonary_atresia": "PuA",
    # ToF
    "tof": "ToF",
    "tet": "ToF",
    "tetralogy of fallot": "ToF",
    "tetralogy_of_fallot": "ToF",
    "fallot": "ToF",
    # TGA
    "tga": "TGA",
    "transposition": "TGA",
    "transposition of the great arteries": "TGA",
    "transposition_of_the_great_arteries": "TGA",
    "d-tga": "TGA",
    "l-tga": "TGA",
    "dtga": "TGA",
    "ltga": "TGA",
}

# imageCHD integer type → canonical disease name
# Based on the imageCHD dataset paper (Xu et al., 2020) 16-class taxonomy
# We map the 8 common types; "Normal" and rarer types map to all-zero vector.
IMAGECHD_TYPE_MAP: dict[int, str] = {
    0: "Normal",    # no CHD
    1: "HLHS",
    2: "ASD",
    3: "VSD",
    4: "AVSD",
    5: "DORV",
    6: "PuA",
    7: "ToF",
    8: "TGA",
    # types 9–15 are less common; map to None (all-zero)
}


def resolve_alias(s: str) -> str | None:
    """Return canonical disease name or None if not recognised."""
    return ALIASES.get(s.lower().strip())


def id_to_case_key(raw_id: str, prefix: str) -> str:
    """Normalise a case ID to the form the trainer expects (e.g. 'ct_1001')."""
    raw_id = str(raw_id).strip()
    if raw_id.lower().startswith(prefix.lower()):
        return raw_id  # already has prefix
    # strip any existing alphabetic prefix before adding the canonical one
    numeric = re.sub(r"^[a-zA-Z_]+", "", raw_id)
    return f"{prefix}{numeric}"


# ─────────────────────────────────────────────────────────────────────────────
# Layout detectors
# ─────────────────────────────────────────────────────────────────────────────

def detect_layout(columns: list[str]) -> str:
    """Return 'binary', 'string', or 'integer'."""
    cols_lower = [c.lower().strip() for c in columns]
    # Check for binary layout: at least 4 of the 8 disease names appear as columns
    disease_cols = sum(
        1 for c in cols_lower
        if resolve_alias(c) in DISEASE_ORDER
    )
    if disease_cols >= 4:
        return "binary"
    # Check for string/integer diagnosis column
    for c in cols_lower:
        if c in ("diagnosis", "type", "chd_type", "disease", "label", "category"):
            return "string"
    # Fall back to string (let parser try)
    return "string"


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_binary(rows: list[dict], id_col: str, prefix: str) -> dict[str, list[int]]:
    """Layout A: one binary column per disease."""
    result = {}
    # Build column→index mapping
    col_to_idx: dict[str, int] = {}
    for col in rows[0].keys():
        canon = resolve_alias(col)
        if canon in DISEASE_ORDER:
            col_to_idx[col] = DISEASE_ORDER.index(canon)

    for row in rows:
        key = id_to_case_key(row[id_col], prefix)
        vec = [0] * 8
        for col, idx in col_to_idx.items():
            val = str(row.get(col, "0")).strip()
            vec[idx] = 1 if val in ("1", "true", "yes", "True", "TRUE", "YES") else 0
        result[key] = vec
    return result


def parse_string(rows: list[dict], id_col: str, diag_col: str, prefix: str) -> dict[str, list[int]]:
    """Layout B: comma-separated disease tags in one column."""
    result = {}
    for row in rows:
        key = id_to_case_key(row[id_col], prefix)
        raw = str(row.get(diag_col, "")).strip()
        vec = [0] * 8
        # Try integer type first
        if raw.lstrip("-").isdigit():
            type_int = int(raw)
            disease = IMAGECHD_TYPE_MAP.get(type_int)
            if disease and disease in DISEASE_ORDER:
                vec[DISEASE_ORDER.index(disease)] = 1
        else:
            tags = [t.strip() for t in re.split(r"[,;/|&]+", raw)]
            for tag in tags:
                canon = resolve_alias(tag)
                if canon in DISEASE_ORDER:
                    vec[DISEASE_ORDER.index(canon)] = 1
        result[key] = vec
    return result


def find_id_and_diag_cols(columns: list[str]) -> tuple[str, str]:
    """Best-effort detection of ID column and diagnosis/type column."""
    id_candidates = ["id", "patient_id", "case_id", "subject_id", "name", "filename"]
    diag_candidates = ["diagnosis", "type", "chd_type", "disease", "label", "category",
                       "diag", "diagnose", "diagnoses"]
    id_col = columns[0]  # default: first column
    diag_col = columns[1] if len(columns) > 1 else columns[0]
    for c in columns:
        if c.lower().strip() in id_candidates:
            id_col = c
            break
    for c in columns:
        if c.lower().strip() in diag_candidates:
            diag_col = c
            break
    return id_col, diag_col


# ─────────────────────────────────────────────────────────────────────────────
# Dataset name lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ID_TO_NAME = {
    1:  "Dataset001_all_imageCHD",
    13: "Dataset013_Fanweidatacleaned",
    20: "Dataset020FanweiDataandImageCHD_HU",
    30: "Dataset030_imageCHD_HU",
}


def find_csv_in_raw(raw_dir: Path, csv_name: str | None) -> Path:
    """Find the spreadsheet (xlsx/csv) in the raw dataset directory."""
    if csv_name:
        candidates = [raw_dir / csv_name]
    else:
        # Prefer xlsx (imageCHD ships an xlsx); fall back to csv
        candidates = (
            sorted(raw_dir.glob("*.xlsx")) + sorted(raw_dir.glob("*.XLSX")) +
            sorted(raw_dir.glob("*.xls"))  + sorted(raw_dir.glob("*.XLS"))  +
            sorted(raw_dir.glob("*.csv"))  + sorted(raw_dir.glob("*.CSV"))
        )
    if not candidates:
        raise FileNotFoundError(
            f"No spreadsheet (.xlsx/.xls/.csv) found in {raw_dir}.\n"
            "Specify --csv-name <filename.xlsx> or pass --csv <full_path>."
        )
    if len(candidates) > 1:
        print(f"[WARN] Multiple spreadsheets found in {raw_dir}; using first: {candidates[0].name}")
        print("       Use --csv-name to be explicit.")
    return candidates[0]


def read_spreadsheet(path: Path, sep: str = ",") -> list[dict]:
    """Read a CSV or Excel file and return a list of row dicts."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError:
            sys.exit(
                "ERROR: openpyxl is required to read .xlsx files.\n"
                "       Install with: pip install openpyxl"
            )
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_raw = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows_raw:
            return []
        headers = [str(h).strip() if h is not None else "" for h in rows_raw[0]]
        result = []
        for row in rows_raw[1:]:
            row_dict = {headers[i]: (row[i] if i < len(row) else None)
                        for i in range(len(headers))}
            result.append(row_dict)
        return result
    else:
        import csv as csvlib
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csvlib.DictReader(f, delimiter=sep)
            return list(reader)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_disease_map(
    csv_path: Path,
    out_path: Path,
    id_col: str | None,
    sep: str,
    id_prefix: str,
    dry_run: bool,
) -> None:
    rows = read_spreadsheet(csv_path, sep)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    columns = list(rows[0].keys())
    print(f"[INFO] CSV columns: {columns}")
    print(f"[INFO] Rows: {len(rows)}")

    layout = detect_layout(columns)
    print(f"[INFO] Detected layout: {layout}")

    if id_col is None:
        auto_id_col, auto_diag_col = find_id_and_diag_cols(columns)
        id_col = auto_id_col
        diag_col = auto_diag_col
    else:
        _, auto_diag_col = find_id_and_diag_cols(columns)
        diag_col = auto_diag_col

    print(f"[INFO] ID column: '{id_col}'  |  ID prefix: '{id_prefix}'")

    if layout == "binary":
        disease_map = parse_binary(rows, id_col, id_prefix)
    else:
        print(f"[INFO] Diagnosis column: '{diag_col}'")
        disease_map = parse_string(rows, id_col, diag_col, id_prefix)

    # Summary
    n_cases = len(disease_map)
    n_positive = sum(1 for v in disease_map.values() if any(v))
    n_normal = n_cases - n_positive
    print(f"\n[SUMMARY] {n_cases} cases  |  {n_positive} with ≥1 disease  |  {n_normal} Normal/unknown")
    per_disease = {d: sum(v[i] for v in disease_map.values()) for i, d in enumerate(DISEASE_ORDER)}
    for d, cnt in per_disease.items():
        print(f"          {d:6s}: {cnt}")

    if dry_run:
        print("\n[DRY-RUN] First 5 entries:")
        for k, v in list(disease_map.items())[:5]:
            print(f"  {k}: {v}")
        print("[DRY-RUN] Output NOT written (pass without --dry-run to save).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(disease_map, f, indent=2)
    print(f"\n[OK] Written: {out_path}  ({n_cases} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert imageCHD diagnosis CSV → disease_map.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv", type=Path, help="Full path to diagnosis spreadsheet (.xlsx or .csv)")
    src.add_argument("--dataset-id", type=int, choices=[1, 13, 20, 30],
                     help="Dataset ID; auto-locates CSV in $nnUNet_raw and writes to $nnUNet_preprocessed")

    parser.add_argument("--out", type=Path,
                        help="Output path for disease_map.json (required unless --dataset-id)")
    parser.add_argument("--csv-name", default=None,
                        help="Spreadsheet filename inside the raw dataset folder (used with --dataset-id); "
                             "defaults to auto-detecting any .xlsx/.csv in that directory")
    parser.add_argument("--id-col", default=None,
                        help="Name of the ID column in the CSV (auto-detected if omitted)")
    parser.add_argument("--sep", default=",",
                        help="CSV field separator (default: ',')")
    parser.add_argument("--id-prefix", default="ct_",
                        help="Prefix to prepend to numeric IDs (default: 'ct_')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and summarise without writing output")

    args = parser.parse_args()

    if args.dataset_id is not None:
        raw_base = os.environ.get("nnUNet_raw")
        prep_base = os.environ.get("nnUNet_preprocessed")
        if not raw_base or not prep_base:
            sys.exit(
                "ERROR: $nnUNet_raw and $nnUNet_preprocessed must be set when using --dataset-id.\n"
                "       Export them or use --csv / --out directly."
            )
        dataset_name = DATASET_ID_TO_NAME.get(args.dataset_id)
        if dataset_name is None:
            sys.exit(f"ERROR: Unknown dataset ID {args.dataset_id}")
        raw_dir = Path(raw_base) / dataset_name
        csv_path = find_csv_in_raw(raw_dir, args.csv_name)
        out_path = Path(prep_base) / dataset_name / "disease_map.json"
    else:
        if args.csv is None:
            sys.exit("ERROR: Provide --csv <path> or --dataset-id <id>")
        csv_path = args.csv
        if args.out is None:
            sys.exit("ERROR: Provide --out <path> when using --csv directly")
        out_path = args.out

    print(f"[INFO] CSV   : {csv_path}")
    print(f"[INFO] Output: {out_path}")

    build_disease_map(
        csv_path=csv_path,
        out_path=out_path,
        id_col=args.id_col,
        sep=args.sep,
        id_prefix=args.id_prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
