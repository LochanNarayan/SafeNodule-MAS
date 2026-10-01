"""Hierarchical Memory (NEW).

Three layers, upgrading the paper's flat memory:
  * working   : the current case (images, measurements, report, live conversation)
  * episodic  : summaries of past cases, retrievable by similar features
  * semantic  : handle to the knowledge graph (owned by GraphRAG)

Episodic recall lets the board reason "we saw a similar nodule that was benign".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EpisodicRecord:
    case_id: str
    features: Dict[str, Any]
    final_label: str


class HierarchicalMemory:
    def __init__(self):
        self.working: Dict[str, Any] = {}
        self.episodic: List[EpisodicRecord] = []
        self.conversation: List[Dict[str, Any]] = []

    # ---- working ----
    def set(self, key: str, value: Any) -> None:
        self.working[key] = value

    def get(self, key: str, default=None) -> Any:
        return self.working.get(key, default)

    def log_turn(self, role: str, content: Any) -> None:
        self.conversation.append({"role": role, "content": content})

    def reset_working(self) -> None:
        self.working.clear()
        self.conversation.clear()

    # ---- episodic ----
    def remember_case(self, case_id: str, features: Dict[str, Any], label: str) -> None:
        self.episodic.append(EpisodicRecord(case_id, features, label))

    def recall_similar(self, features: Dict[str, Any], k: int = 3) -> List[EpisodicRecord]:
        def sim(rec: EpisodicRecord) -> int:
            keys = ("density", "margin", "shape", "spiculation")
            return sum(1 for kk in keys if rec.features.get(kk) == features.get(kk))
        return sorted(self.episodic, key=sim, reverse=True)[:k]
