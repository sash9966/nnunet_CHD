"""
chd_landmarks.labels
====================

Resolve canonical CHD anatomy structures -> integer label IDs by NAME, using
configs/chd_label_map.yaml and the source dataset.json. Label IDs are NEVER
hardcoded in code; if a structure cannot be resolved by name it falls back to
the config `default_ids`, and if that is null the structure is reported as
*absent* (downstream builders skip it with a warning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import io


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


@dataclass
class LabelMap:
    """Resolved mapping for one dataset."""
    background_id: int
    structure_to_id: Dict[str, Optional[int]]   # canonical name -> id or None (absent)
    id_to_structure: Dict[int, str] = field(default_factory=dict)
    vessel_structures: List[str] = field(default_factory=list)
    chamber_structures: List[str] = field(default_factory=list)
    raw_labels: Dict[str, int] = field(default_factory=dict)  # original dataset.json labels
    warnings: List[str] = field(default_factory=list)

    # -- queries ----------------------------------------------------------
    def has(self, structure: str) -> bool:
        return self.structure_to_id.get(structure) is not None

    def id_of(self, structure: str) -> Optional[int]:
        return self.structure_to_id.get(structure)

    def require(self, structures: List[str]) -> bool:
        """True only if every requested structure is present."""
        return all(self.has(s) for s in structures)

    def ids_of(self, structures: List[str]) -> List[int]:
        """Resolve a list of structure names to present integer ids (absent dropped)."""
        out = []
        for s in structures:
            i = self.structure_to_id.get(s)
            if i is not None:
                out.append(i)
        return out

    def present_structures(self) -> List[str]:
        return [s for s, i in self.structure_to_id.items() if i is not None]

    def next_free_id(self) -> int:
        used = [i for i in self.structure_to_id.values() if i is not None]
        used.append(self.background_id)
        return max(used) + 1 if used else 1


def load_label_map(label_map_cfg_path: str, dataset_dir: Optional[str] = None) -> LabelMap:
    """
    Build a LabelMap. If `dataset_dir` is given, resolve each structure by name
    from that dataset's dataset.json (preferred). Otherwise use config defaults.
    """
    cfg = io.load_yaml(label_map_cfg_path)
    background_id = int(cfg.get("background_id", 0))
    aliases: Dict[str, List[str]] = cfg.get("aliases", {})
    default_ids: Dict[str, Optional[int]] = cfg.get("default_ids", {})

    raw_labels: Dict[str, int] = {}
    name_to_id_norm: Dict[str, int] = {}
    if dataset_dir is not None:
        try:
            ds = io.read_dataset_json(dataset_dir)
            for name, val in ds.get("labels", {}).items():
                # region-based dataset.json may map name -> list; skip those here
                if isinstance(val, (int,)) or (isinstance(val, str) and str(val).isdigit()):
                    iv = int(val)
                    raw_labels[name] = iv
                    name_to_id_norm[_norm(name)] = iv
        except Exception as e:  # noqa: BLE001
            io.warn(f"could not read labels from {dataset_dir}: {e}; using config defaults")

    warns: List[str] = []
    structure_to_id: Dict[str, Optional[int]] = {}
    for structure, default in default_ids.items():
        resolved: Optional[int] = None
        # 1) try resolving by name via aliases against the dataset labels
        for alias in aliases.get(structure, []) + [structure]:
            key = _norm(alias)
            if key in name_to_id_norm:
                resolved = name_to_id_norm[key]
                break
        # 2) fall back to default id from config
        if resolved is None and default is not None and not raw_labels:
            resolved = int(default)
        elif resolved is None and default is not None and raw_labels:
            # dataset present but name not found -> warn, fall back cautiously
            resolved = int(default) if int(default) in raw_labels.values() else None
            if resolved is None:
                warns.append(
                    f"structure '{structure}' not found by name in dataset labels "
                    f"and default id {default} not present -> treated as ABSENT"
                )
        if resolved is None and default is None:
            warns.append(f"structure '{structure}' is configured absent (null) -> skipped")
        structure_to_id[structure] = resolved

    id_to_structure = {i: s for s, i in structure_to_id.items() if i is not None}

    lm = LabelMap(
        background_id=background_id,
        structure_to_id=structure_to_id,
        id_to_structure=id_to_structure,
        vessel_structures=list(cfg.get("vessel_structures", [])),
        chamber_structures=list(cfg.get("chamber_structures", [])),
        raw_labels=raw_labels,
        warnings=warns,
    )
    for w in warns:
        io.warn(w)
    return lm
