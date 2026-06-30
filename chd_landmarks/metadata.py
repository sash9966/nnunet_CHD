"""
chd_landmarks.metadata
=======================

Parse CHD diagnosis flags from the imageCHD spreadsheet (or a CSV) into a
per-case dict of {disease_key: bool}, using the `flag_columns` section of
chd_disease_rules.yaml so disease->column mapping is configurable (never
hardcoded).

Also tracks ANNOTATION STATUS per case/label: whether a disease landmark was
derived+verified for that case. This matters because, for diseases not
annotated in every positive case, absence must NOT be treated as confirmed
background (acceptance criterion D / design principle 4).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import io


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def id_to_case_key(raw_id: str, prefix: str = "ct_") -> str:
    """Normalise '1001' / 'ct_1001' -> 'ct_1001' to match nnU-Net case ids."""
    raw_id = str(raw_id).strip()
    if raw_id.lower().startswith(prefix.lower()):
        return raw_id
    numeric = re.sub(r"^[a-zA-Z_]+", "", raw_id)
    return f"{prefix}{numeric}" if numeric else raw_id


# ---------------------------------------------------------------------------
# Spreadsheet reading (mirrors scripts/make_disease_map.py, kept independent)
# ---------------------------------------------------------------------------
def _read_rows(path: Path, sep: str = ",") -> List[dict]:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openpyxl required to read .xlsx; `pip install openpyxl`") from e
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_raw = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows_raw:
            return []
        headers = [str(h).strip() if h is not None else "" for h in rows_raw[0]]
        out = []
        for row in rows_raw[1:]:
            if row is None or all(v is None for v in row):
                continue
            out.append({headers[i]: (row[i] if i < len(row) else None) for i in range(len(headers))})
        return out
    else:
        import csv as csvlib
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csvlib.DictReader(f, delimiter=sep))


def _is_true(val) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("1", "1.0", "true", "yes", "y", "x")


@dataclass
class CaseMetadata:
    case_id: str
    flags: Dict[str, bool]                  # disease_key -> present?

    def active_diseases(self) -> List[str]:
        return [d for d, v in self.flags.items() if v]


@dataclass
class AnnotationStatus:
    """
    Per-case record of which derived labels are *known* (derived/verified) vs
    *unknown* (not derivable -> absence is NOT confirmed background).
    """
    case_id: str
    derived: Dict[str, str] = field(default_factory=dict)   # region -> confidence
    unknown: List[str] = field(default_factory=list)         # regions whose absence is unverified
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "derived": self.derived,
            "unknown_absence": self.unknown,
            "warnings": self.warnings,
        }


def load_disease_flags(
    metadata_path: str,
    flag_columns: Dict[str, List[str]],
    id_col: Optional[str] = None,
    id_prefix: str = "ct_",
    sep: str = ",",
) -> Dict[str, CaseMetadata]:
    """
    Read the spreadsheet and return {case_id: CaseMetadata}.

    `flag_columns` maps a canonical disease key -> list of acceptable column
    names (case/space-insensitive). A disease is present for a case if ANY of
    its columns is truthy.
    """
    path = Path(metadata_path)
    rows = _read_rows(path, sep=sep)
    if not rows:
        raise ValueError(f"metadata file is empty: {metadata_path}")

    columns = list(rows[0].keys())
    col_norm = {_norm(c): c for c in columns}

    # resolve id column
    if id_col is None:
        for cand in ("index", "id", "patient_id", "case_id", "subject_id", "name"):
            if cand in col_norm:
                id_col = col_norm[cand]
                break
        if id_col is None:
            id_col = columns[0]

    # resolve disease -> actual column names present in this file
    resolved_cols: Dict[str, List[str]] = {}
    for disease, accepted in flag_columns.items():
        present = [col_norm[_norm(a)] for a in accepted if _norm(a) in col_norm]
        if present:
            resolved_cols[disease] = present

    result: Dict[str, CaseMetadata] = {}
    for row in rows:
        raw_id = row.get(id_col)
        if raw_id is None or str(raw_id).strip() == "":
            continue
        key = id_to_case_key(raw_id, id_prefix)
        flags = {d: any(_is_true(row.get(c)) for c in cols)
                 for d, cols in resolved_cols.items()}
        result[key] = CaseMetadata(case_id=key, flags=flags)
    return result


def summarize(meta: Dict[str, CaseMetadata]) -> Dict[str, int]:
    """Count cases per disease for a quick sanity report."""
    counts: Dict[str, int] = {}
    for cm in meta.values():
        for d, v in cm.flags.items():
            counts[d] = counts.get(d, 0) + (1 if v else 0)
    counts["_total_cases"] = len(meta)
    counts["_cases_with_any_disease"] = sum(1 for cm in meta.values() if any(cm.flags.values()))
    return counts
