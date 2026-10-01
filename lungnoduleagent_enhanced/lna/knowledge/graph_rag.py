"""Medical Graph RAG (lightweight, self-contained).

A faithful-in-spirit, dependency-free version of the GraphRAG pipeline
(D -> G -> S -> A): documents are split into sections ("communities"),
each section is summarized (its heading + first sentences), and retrieval scores
sections against a keyword query with a simple bag-of-words overlap. This is
deliberately transparent; swap in a real vector store / graph DB for production.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

from ..config import Config


@dataclass
class Community:
    title: str
    text: str
    summary: str


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class MedicalGraphRAG:
    def __init__(self, config: Config, kb_dir: str = None):
        self.config = config
        if kb_dir is None:
            kb_dir = os.path.join(os.path.dirname(__file__), "kb")
        self.communities: List[Community] = []
        self._build(kb_dir)

    def _build(self, kb_dir: str) -> None:
        """D -> G -> S : split docs into '##' sections and summarize each."""
        for fn in sorted(os.listdir(kb_dir)):
            if not fn.endswith((".md", ".txt")):
                continue
            with open(os.path.join(kb_dir, fn), encoding="utf-8") as fh:
                content = fh.read()
            # split on level-2 headings -> communities
            parts = re.split(r"\n##\s+", content)
            for part in parts:
                part = part.strip()
                if not part or part.startswith("#"):
                    continue
                lines = part.splitlines()
                title = lines[0].strip()
                body = " ".join(lines[1:]).strip()
                if not body:
                    continue
                summary = " ".join(re.split(r"(?<=[.])\s+", body)[:2])
                self.communities.append(Community(title, body, summary))

    def search(self, query: str, top_k: int = None) -> List[Community]:
        top_k = top_k or self.config.top_k_chunks
        q = set(_tokenize(query))
        scored = []
        for c in self.communities:
            toks = set(_tokenize(c.title + " " + c.text))
            overlap = len(q & toks)
            scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored[:top_k] if score > 0] or \
               [c for _, c in scored[:1]]

    def summarize_for(self, query: str) -> str:
        hits = self.search(query)
        return " ".join(f"[{c.title}] {c.summary}" for c in hits)
