"""SQLite persistence for Clanker-LM session state."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Optional

from .memory import ConversationMemory
from .models import AffectVector, Entity, Fact


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS lm_sessions (
    session_id TEXT PRIMARY KEY,
    turn INTEGER NOT NULL,
    entity_counter INTEGER NOT NULL,
    fact_counter INTEGER NOT NULL,
    v INTEGER NOT NULL,
    a INTEGER NOT NULL,
    d INTEGER NOT NULL,
    u INTEGER NOT NULL,
    g INTEGER NOT NULL,
    w INTEGER NOT NULL,
    i INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lm_entities (
    session_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    gender TEXT NOT NULL,
    grammatical_number TEXT NOT NULL,
    relation_to_user TEXT,
    determiner TEXT,
    salience REAL NOT NULL,
    last_turn INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (session_id, entity_id),
    FOREIGN KEY (session_id) REFERENCES lm_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lm_aliases (
    session_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    PRIMARY KEY (session_id, entity_id, alias),
    FOREIGN KEY (session_id, entity_id)
        REFERENCES lm_entities(session_id, entity_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lm_facts (
    session_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    tense TEXT NOT NULL,
    polarity INTEGER NOT NULL,
    modality TEXT,
    repeated INTEGER NOT NULL,
    roles_json TEXT NOT NULL,
    surface TEXT NOT NULL,
    provenance TEXT NOT NULL,
    certainty INTEGER NOT NULL,
    turn_id INTEGER NOT NULL,
    PRIMARY KEY (session_id, fact_id),
    FOREIGN KEY (session_id) REFERENCES lm_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lm_facts_predicate
    ON lm_facts(session_id, predicate, turn_id);
CREATE INDEX IF NOT EXISTS idx_lm_aliases_alias
    ON lm_aliases(session_id, alias);
"""


class SQLiteSessionStore:
    """Normalized, inspectable persistence for entities, aliases, facts, and state."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(_SCHEMA)
        self.connection.commit()

    def save(self, memory: ConversationMemory) -> None:
        state = memory.running_state
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO lm_sessions(
                    session_id, turn, entity_counter, fact_counter,
                    v, a, d, u, g, w, i, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    turn=excluded.turn,
                    entity_counter=excluded.entity_counter,
                    fact_counter=excluded.fact_counter,
                    v=excluded.v,
                    a=excluded.a,
                    d=excluded.d,
                    u=excluded.u,
                    g=excluded.g,
                    w=excluded.w,
                    i=excluded.i,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    memory.session_id,
                    memory.turn,
                    memory._entity_counter,
                    memory._fact_counter,
                    state.v,
                    state.a,
                    state.d,
                    state.u,
                    state.g,
                    state.w,
                    state.i,
                ),
            )
            self.connection.execute(
                "DELETE FROM lm_aliases WHERE session_id = ?", (memory.session_id,)
            )
            self.connection.execute(
                "DELETE FROM lm_entities WHERE session_id = ?", (memory.session_id,)
            )
            self.connection.execute(
                "DELETE FROM lm_facts WHERE session_id = ?", (memory.session_id,)
            )
            for entity in memory.entities.values():
                self.connection.execute(
                    """
                    INSERT INTO lm_entities(
                        session_id, entity_id, canonical_name, display_name,
                        kind, gender, grammatical_number, relation_to_user,
                        determiner, salience, last_turn, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.session_id,
                        entity.entity_id,
                        entity.canonical_name,
                        entity.display_name,
                        entity.kind.value,
                        entity.gender.value,
                        entity.number.value,
                        entity.relation_to_user,
                        entity.determiner,
                        entity.salience,
                        entity.last_turn,
                        json.dumps(entity.metadata, sort_keys=True),
                    ),
                )
                for alias in sorted(entity.aliases):
                    self.connection.execute(
                        "INSERT INTO lm_aliases(session_id, entity_id, alias) VALUES (?, ?, ?)",
                        (memory.session_id, entity.entity_id, alias),
                    )
            for fact in memory.facts:
                frame = fact.frame
                self.connection.execute(
                    """
                    INSERT INTO lm_facts(
                        session_id, fact_id, predicate, tense, polarity,
                        modality, repeated, roles_json, surface,
                        provenance, certainty, turn_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.session_id,
                        fact.fact_id,
                        frame.predicate,
                        frame.tense,
                        1 if frame.polarity else 0,
                        frame.modality,
                        1 if frame.repeated else 0,
                        json.dumps(
                            {
                                role.value: value.to_dict()
                                for role, value in frame.roles.items()
                            },
                            sort_keys=True,
                        ),
                        frame.surface,
                        fact.provenance.value,
                        fact.certainty,
                        fact.turn_id,
                    ),
                )

    def load(self, session_id: str) -> Optional[ConversationMemory]:
        session = self.connection.execute(
            "SELECT * FROM lm_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if session is None:
            return None
        entity_rows = self.connection.execute(
            "SELECT * FROM lm_entities WHERE session_id = ? ORDER BY entity_id",
            (session_id,),
        ).fetchall()
        aliases = self.connection.execute(
            "SELECT entity_id, alias FROM lm_aliases WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        aliases_by_entity: dict[str, list[str]] = {}
        for row in aliases:
            aliases_by_entity.setdefault(str(row["entity_id"]), []).append(str(row["alias"]))
        fact_rows = self.connection.execute(
            "SELECT * FROM lm_facts WHERE session_id = ? ORDER BY fact_id",
            (session_id,),
        ).fetchall()

        payload = {
            "session_id": session_id,
            "turn": int(session["turn"]),
            "entity_counter": int(session["entity_counter"]),
            "fact_counter": int(session["fact_counter"]),
            "running_state": {
                key: int(session[key]) for key in "vadugwi"
            },
            "entities": [],
            "facts": [],
        }
        for row in entity_rows:
            payload["entities"].append(
                {
                    "entity_id": str(row["entity_id"]),
                    "canonical_name": str(row["canonical_name"]),
                    "display_name": str(row["display_name"]),
                    "kind": str(row["kind"]),
                    "gender": str(row["gender"]),
                    "number": str(row["grammatical_number"]),
                    "relation_to_user": row["relation_to_user"],
                    "determiner": row["determiner"],
                    "aliases": aliases_by_entity.get(str(row["entity_id"]), []),
                    "salience": float(row["salience"]),
                    "last_turn": int(row["last_turn"]),
                    "metadata": json.loads(str(row["metadata_json"])),
                }
            )
        for row in fact_rows:
            payload["facts"].append(
                {
                    "fact_id": str(row["fact_id"]),
                    "frame": {
                        "predicate": str(row["predicate"]),
                        "roles": json.loads(str(row["roles_json"])),
                        "tense": str(row["tense"]),
                        "polarity": bool(row["polarity"]),
                        "modality": row["modality"],
                        "repeated": bool(row["repeated"]),
                        "surface": str(row["surface"]),
                    },
                    "provenance": str(row["provenance"]),
                    "certainty": int(row["certainty"]),
                    "turn_id": int(row["turn_id"]),
                }
            )
        return ConversationMemory.from_dict(payload)

    def reset(self, session_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM lm_sessions WHERE session_id = ?", (session_id,)
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteSessionStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
