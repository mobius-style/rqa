"""Question Graph — append-only SQLite store with read paths (SPEC_v0_2.md §5.4, §8.4).

Retrieval backend: char-trigram overlap (language-neutral, zero extra deps).
ME5 embedding retrieval is a Stage B upgrade (reuse MMV Box M infra); the
Fragment interface is stable across that swap.

No delete API exists by design (§8.4): status transitions + audit log only.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

NODE_KINDS = {"claim", "question", "tension", "note"}
STATUS_TRANSITIONS = {
    "open": {"revisited", "resolved", "abandoned", "archived"},
    "revisited": {"resolved", "abandoned", "archived"},
    "resolved": {"archived"},
    "abandoned": {"archived"},
    "archived": set(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    provenance TEXT NOT NULL,
    created_at TEXT NOT NULL,
    session TEXT,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    op TEXT NOT NULL,
    node_id INTEGER,
    detail TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trigrams(text: str) -> set[str]:
    t = "".join(text.lower().split())
    return {t[i : i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}


@dataclass
class Fragment:
    node_id: int
    kind: str
    text: str
    status: str
    provenance: str
    created_at: str
    score: float

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "text": self.text,
            "status": self.status,
            "provenance": self.provenance,
            "created_at": self.created_at,
            "score": round(self.score, 4),
        }


class QuestionGraph:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- write path -------------------------------------------------------

    def add_node(
        self,
        kind: str,
        text: str,
        provenance: str,
        session: str | None = None,
        meta: dict | None = None,
    ) -> int:
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind: {kind}")
        text = text.strip()
        if not text:
            raise ValueError("empty node text")
        cur = self._conn.execute(
            "INSERT INTO nodes (kind, text, status, provenance, created_at, session, meta)"
            " VALUES (?, ?, 'open', ?, ?, ?, ?)",
            (kind, text, provenance, _now(), session, json.dumps(meta or {}, ensure_ascii=False)),
        )
        node_id = cur.lastrowid
        self._audit("add_node", node_id, f"{kind}: {text[:80]}")
        self._conn.commit()
        return node_id

    def set_status(self, node_id: int, new_status: str) -> None:
        row = self._conn.execute("SELECT status FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"node {node_id} not found")
        current = row["status"]
        if new_status not in STATUS_TRANSITIONS.get(current, set()):
            raise ValueError(f"illegal transition {current} -> {new_status}")
        self._conn.execute("UPDATE nodes SET status=? WHERE id=?", (new_status, node_id))
        self._audit("set_status", node_id, f"{current} -> {new_status}")
        self._conn.commit()

    def _audit(self, op: str, node_id: int | None, detail: str) -> None:
        self._conn.execute(
            "INSERT INTO audit (ts, op, node_id, detail) VALUES (?, ?, ?, ?)",
            (_now(), op, node_id, detail),
        )

    # -- read path 1: pre-noticing retrieval (§5.4) ------------------------

    def search(self, query: str, top_k: int = 8, exclude_archived: bool = True) -> list[Fragment]:
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY id DESC LIMIT 5000"
        ).fetchall()
        q_tri = _trigrams(query)
        results: list[Fragment] = []
        for row in rows:
            if exclude_archived and row["status"] == "archived":
                continue
            n_tri = _trigrams(row["text"])
            union = q_tri | n_tri
            if not union:
                continue
            score = len(q_tri & n_tri) / len(union)
            if score > 0.0:
                results.append(
                    Fragment(
                        node_id=row["id"],
                        kind=row["kind"],
                        text=row["text"],
                        status=row["status"],
                        provenance=row["provenance"],
                        created_at=row["created_at"],
                        score=score,
                    )
                )
        results.sort(key=lambda f: f.score, reverse=True)
        return results[:top_k]

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT kind, status, COUNT(*) AS n FROM nodes GROUP BY kind, status"
        ).fetchall()
        audit_n = self._conn.execute("SELECT COUNT(*) AS n FROM audit").fetchone()["n"]
        return {
            "nodes": [{"kind": r["kind"], "status": r["status"], "count": r["n"]} for r in rows],
            "audit_entries": audit_n,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        self._conn.close()
