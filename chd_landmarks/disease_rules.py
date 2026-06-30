"""
chd_landmarks.disease_rules
===========================

Loads chd_disease_rules.yaml and decides, per case, which derivation rules are
ACTIVE. A rule is active only when:
  (a) its diagnosis flag is set for the case, AND
  (b) every `relevant_label` it needs is present in the dataset label map.

Rules whose flag is set but whose anatomy is missing are reported as SKIPPED
(so their absence is recorded as "unknown", never as confirmed background).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from . import io
from .labels import LabelMap
from .metadata import CaseMetadata


@dataclass
class DiseaseRule:
    name: str
    definition: str
    relevant_labels: List[str]
    derived_regions: List[str]
    expected_topology: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    source: str = ""
    requires_missing_anatomy: bool = False


@dataclass
class RuleSet:
    rules: Dict[str, DiseaseRule]
    flag_columns: Dict[str, List[str]]

    def for_case(self, meta: CaseMetadata, label_map: LabelMap):
        """
        Returns (active_rules, skipped) where:
          active_rules: List[DiseaseRule] firing for this case
          skipped: List[(disease, reason)] flagged but not derivable
        """
        active: List[DiseaseRule] = []
        skipped: List[tuple] = []
        for disease in meta.active_diseases():
            rule = self.rules.get(disease)
            if rule is None:
                skipped.append((disease, "no rule defined"))
                continue
            missing = [l for l in rule.relevant_labels if not label_map.has(l)]
            if missing or rule.requires_missing_anatomy:
                skipped.append(
                    (disease, f"missing required anatomy {missing or rule.relevant_labels}")
                )
                continue
            active.append(rule)
        return active, skipped


def load_rules(rules_cfg_path: str) -> RuleSet:
    cfg = io.load_yaml(rules_cfg_path)
    flag_columns = cfg.get("flag_columns", {})
    rules: Dict[str, DiseaseRule] = {}
    for name, body in (cfg.get("rules") or {}).items():
        rules[name] = DiseaseRule(
            name=name,
            definition=body.get("definition", ""),
            relevant_labels=list(body.get("relevant_labels", [])),
            derived_regions=list(body.get("derived_regions", [])),
            expected_topology=list(body.get("expected_topology", [])),
            metrics=list(body.get("metrics", [])),
            source=body.get("source", ""),
            requires_missing_anatomy=bool(body.get("requires_missing_anatomy", False)),
        )
    return RuleSet(rules=rules, flag_columns=flag_columns)
