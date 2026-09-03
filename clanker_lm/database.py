"""SQLite state for atomic language, learning, resolvers, and trajectories.

The database deliberately contains no response sentences and no string
interpolation templates.  Generation is compositional: semantic frames are
realized by grammar code, while this store supplies single-token atoms,
abstract grammar productions, gates, learned lexical evidence, resolver
observations, and mathematical trajectory statistics.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .model import AffectVector


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS atoms (
    id TEXT PRIMARY KEY,
    surface TEXT NOT NULL,
    lemma TEXT NOT NULL,
    category TEXT NOT NULL,
    register_name TEXT NOT NULL DEFAULT 'neutral',
    severity INTEGER NOT NULL DEFAULT 0,
    features_json TEXT NOT NULL DEFAULT '{}',
    dv INTEGER NOT NULL DEFAULT 128,
    da INTEGER NOT NULL DEFAULT 128,
    dd INTEGER NOT NULL DEFAULT 128,
    du INTEGER NOT NULL DEFAULT 0,
    dg INTEGER NOT NULL DEFAULT 128,
    dw INTEGER NOT NULL DEFAULT 128,
    di INTEGER NOT NULL DEFAULT 128,
    CHECK (length(trim(surface)) > 0),
    CHECK (instr(trim(surface), ' ') = 0),
    UNIQUE(surface, category, register_name)
);

CREATE TABLE IF NOT EXISTS grammar_rules (
    id TEXT PRIMARY KEY,
    parent_symbol TEXT NOT NULL,
    child_symbols_json TEXT NOT NULL,
    condition_json TEXT NOT NULL DEFAULT '{}',
    priority INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS gate_rules (
    id TEXT PRIMARY KEY,
    condition_json TEXT NOT NULL,
    lock_pools_json TEXT NOT NULL DEFAULT '[]',
    allow_pools_json TEXT NOT NULL DEFAULT '[]',
    priority INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS learned_terms (
    term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized TEXT NOT NULL UNIQUE,
    preferred_surface TEXT NOT NULL,
    first_seen_turn INTEGER NOT NULL DEFAULT 0,
    last_seen_turn INTEGER NOT NULL DEFAULT 0,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'unresolved',
    active_sense_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS learned_senses (
    sense_id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_id INTEGER NOT NULL REFERENCES learned_terms(term_id) ON DELETE CASCADE,
    sense_index INTEGER NOT NULL DEFAULT 1,
    scope_type TEXT NOT NULL DEFAULT 'session',
    scope_id TEXT NOT NULL DEFAULT 'default',
    part_of_speech TEXT NOT NULL DEFAULT 'unknown',
    semantic_class TEXT NOT NULL DEFAULT 'unknown',
    register_name TEXT NOT NULL DEFAULT 'neutral',
    confidence REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'provisional',
    version INTEGER NOT NULL DEFAULT 1,
    support_weight REAL NOT NULL DEFAULT 0.0,
    contradiction_weight REAL NOT NULL DEFAULT 0.0,
    vector_json TEXT NOT NULL,
    conditions_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(term_id, sense_index, scope_type, scope_id)
);

CREATE TABLE IF NOT EXISTS lexical_evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_id INTEGER NOT NULL REFERENCES learned_terms(term_id) ON DELETE CASCADE,
    sense_id INTEGER REFERENCES learned_senses(sense_id) ON DELETE SET NULL,
    evidence_kind TEXT NOT NULL,
    raw_context TEXT NOT NULL DEFAULT '',
    raw_explanation TEXT NOT NULL DEFAULT '',
    context_hash TEXT NOT NULL,
    context_features_json TEXT NOT NULL DEFAULT '{}',
    vector_json TEXT NOT NULL,
    interpreted_vector_json TEXT NOT NULL,
    semantic_class TEXT NOT NULL DEFAULT 'unknown',
    polarity INTEGER NOT NULL DEFAULT 0,
    support_weight REAL NOT NULL DEFAULT 1.0,
    contradiction_weight REAL NOT NULL DEFAULT 0.0,
    observed_turn INTEGER NOT NULL DEFAULT 0,
    reanalyzed_version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS resolver_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_hash TEXT NOT NULL,
    resolver_name TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    source_name TEXT NOT NULL,
    certainty INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    expires_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS trajectory_turns (
    trajectory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_hash TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    incoming_act TEXT NOT NULL,
    response_act TEXT NOT NULL,
    context_key TEXT NOT NULL,
    input_vector_json TEXT NOT NULL,
    state_before_json TEXT NOT NULL,
    target_vector_json TEXT NOT NULL,
    response_vector_json TEXT NOT NULL,
    predicted_after_json TEXT NOT NULL,
    observed_next_json TEXT,
    reaction_vector_json TEXT,
    residual_json TEXT,
    profile_id TEXT,
    finalized INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transition_stats (
    context_key TEXT PRIMARY KEY,
    sample_count INTEGER NOT NULL DEFAULT 0,
    mean_residual_json TEXT NOT NULL,
    mean_reaction_json TEXT NOT NULL,
    success_mean REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS corpus_profiles (
    profile_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    quote_count INTEGER NOT NULL,
    trajectory_blob BLOB NOT NULL,
    delta_blob BLOB NOT NULL,
    centroid_json TEXT NOT NULL,
    variance_json TEXT NOT NULL,
    delta_centroid_json TEXT NOT NULL,
    act_distribution_json TEXT NOT NULL,
    transition_matrix_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trajectory_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL REFERENCES corpus_profiles(profile_id) ON DELETE CASCADE,
    start_index INTEGER NOT NULL,
    window_size INTEGER NOT NULL,
    vector_blob BLOB NOT NULL,
    delta_blob BLOB NOT NULL,
    vector_hash TEXT NOT NULL,
    delta_hash TEXT NOT NULL,
    UNIQUE(profile_id, start_index, window_size)
);

CREATE INDEX IF NOT EXISTS idx_atoms_category
ON atoms(category, register_name, severity);
CREATE INDEX IF NOT EXISTS idx_grammar_parent
ON grammar_rules(parent_symbol, priority DESC);
CREATE INDEX IF NOT EXISTS idx_learned_senses_term
ON learned_senses(term_id, confidence DESC, status);
CREATE INDEX IF NOT EXISTS idx_lexical_evidence_term
ON lexical_evidence(term_id, observed_turn);
CREATE INDEX IF NOT EXISTS idx_resolver_request
ON resolver_observations(request_hash, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trajectory_context
ON trajectory_turns(context_key, finalized);
CREATE INDEX IF NOT EXISTS idx_trajectory_chunk_hash
ON trajectory_chunks(window_size, vector_hash, delta_hash);
"""


_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9_:*.-]*$")


@dataclass(frozen=True)
class Atom:
    atom_id: str
    surface: str
    lemma: str
    category: str
    register: str
    severity: int
    features: Dict[str, Any]
    effects: AffectVector


@dataclass(frozen=True)
class GrammarRule:
    rule_id: str
    parent: str
    children: Tuple[str, ...]
    conditions: Dict[str, Any]
    priority: int


# Compatibility name for callers from the first vertical slice.  It is an
# abstract production, not a surface construction and has no template field.
Construction = GrammarRule


@dataclass(frozen=True)
class LearnedSenseRecord:
    sense_id: int
    term_id: int
    sense_index: int
    surface: str
    part_of_speech: str
    semantic_class: str
    register: str
    confidence: float
    status: str
    version: int
    support_weight: float
    contradiction_weight: float
    vector: AffectVector
    conditions: Dict[str, Any]


class LanguageStore:
    """Template-free atomic language and adaptive state store."""

    SQLITE_TIMEOUT_SECONDS = 30.0

    OVERLAY_TABLES = (
        "learned_terms",
        "learned_senses",
        "lexical_evidence",
        "resolver_observations",
        "trajectory_turns",
        "transition_stats",
        "corpus_profiles",
        "trajectory_chunks",
    )

    def __init__(self, path: str | Path = ":memory:", *, seed: bool = True) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, timeout=self.SQLITE_TIMEOUT_SECONDS)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA_SQL)
        if seed and not self._has_atoms():
            self.seed_defaults()
        self.assert_template_free()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LanguageStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _has_atoms(self) -> bool:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM atoms").fetchone()
        return bool(row and row["n"])

    @staticmethod
    def _load_seed() -> Mapping[str, Any]:
        try:
            seed_file = resources.files("clanker_lm.data").joinpath("language_seed.json")
            return json.loads(seed_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, ModuleNotFoundError, AttributeError):
            path = Path(__file__).parent / "data" / "language_seed.json"
            return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _validate_atom_surface(surface: str) -> None:
        if not surface or any(character.isspace() for character in surface):
            raise ValueError(f"Language atoms must be single tokens, got {surface!r}")

    @staticmethod
    def _validate_grammar_symbols(symbols: Sequence[str]) -> None:
        for symbol in symbols:
            if not _SYMBOL_RE.fullmatch(symbol):
                raise ValueError(
                    f"Grammar productions may contain only abstract symbols, got {symbol!r}"
                )

    def seed_defaults(self) -> None:
        data = self._load_seed()
        with self.connection:
            for atom in data.get("atoms", []):
                self._validate_atom_surface(str(atom["surface"]))
                effects = atom.get("effects", {})
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO atoms(
                        id, surface, lemma, category, register_name, severity,
                        features_json, dv, da, dd, du, dg, dw, di
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        atom["id"],
                        atom["surface"],
                        atom.get("lemma", atom["surface"]),
                        atom["category"],
                        atom.get("register", "neutral"),
                        int(atom.get("severity", 0)),
                        json.dumps(atom.get("features", {}), sort_keys=True),
                        int(effects.get("v", 128)),
                        int(effects.get("a", 128)),
                        int(effects.get("d", 128)),
                        int(effects.get("u", 0)),
                        int(effects.get("g", 128)),
                        int(effects.get("w", 128)),
                        int(effects.get("i", 128)),
                    ),
                )
            for rule in data.get("grammar_rules", []):
                children = tuple(str(item) for item in rule.get("symbols", []))
                self._validate_grammar_symbols(children)
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO grammar_rules(
                        id, parent_symbol, child_symbols_json, condition_json, priority
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        rule["id"],
                        rule["parent"],
                        json.dumps(children),
                        json.dumps(rule.get("conditions", {}), sort_keys=True),
                        int(rule.get("priority", 0)),
                    ),
                )
            for rule in data.get("gate_rules", []):
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO gate_rules(
                        id, condition_json, lock_pools_json, allow_pools_json, priority
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        rule["id"],
                        json.dumps(rule.get("condition", {}), sort_keys=True),
                        json.dumps(rule.get("lock_pools", [])),
                        json.dumps(rule.get("allow_pools", [])),
                        int(rule.get("priority", 0)),
                    ),
                )

    # ------------------------------------------------------------------
    # Atomic language and abstract grammar
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_atom(row: sqlite3.Row) -> Atom:
        return Atom(
            atom_id=str(row["id"]),
            surface=str(row["surface"]),
            lemma=str(row["lemma"]),
            category=str(row["category"]),
            register=str(row["register_name"]),
            severity=int(row["severity"]),
            features=json.loads(row["features_json"] or "{}"),
            effects=AffectVector(
                v=int(row["dv"]),
                a=int(row["da"]),
                d=int(row["dd"]),
                u=int(row["du"]),
                g=int(row["dg"]),
                w=int(row["dw"]),
                i=int(row["di"]),
            ),
        )

    def atom_candidates(
        self,
        category: str,
        *,
        register: Optional[str] = None,
        severity_max: Optional[int] = None,
        features: Optional[Mapping[str, Any]] = None,
    ) -> List[Atom]:
        rows = self.connection.execute(
            "SELECT * FROM atoms WHERE category = ? ORDER BY id ASC",
            (category,),
        ).fetchall()
        results: List[Atom] = []
        for row in rows:
            atom = self._row_to_atom(row)
            if register and atom.register not in {register, "any", "neutral"}:
                continue
            if severity_max is not None and atom.severity > severity_max:
                continue
            if features and not all(atom.features.get(key) == value for key, value in features.items()):
                continue
            results.append(atom)
        return results

    def atom_by_id(self, atom_id: str) -> Optional[Atom]:
        row = self.connection.execute("SELECT * FROM atoms WHERE id = ?", (atom_id,)).fetchone()
        return self._row_to_atom(row) if row else None

    def atom_surface(self, atom_id: str, default: str = "") -> str:
        atom = self.atom_by_id(atom_id)
        return atom.surface if atom else default

    def known_atom_surfaces(self) -> set[str]:
        return {
            str(row["surface"]).lower()
            for row in self.connection.execute("SELECT surface FROM atoms")
        }

    def grammar_rules(
        self,
        parent: str,
        *,
        features: Optional[Mapping[str, Any]] = None,
    ) -> List[GrammarRule]:
        rows = self.connection.execute(
            "SELECT * FROM grammar_rules WHERE parent_symbol = ? ORDER BY priority DESC, id ASC",
            (parent,),
        ).fetchall()
        results: List[GrammarRule] = []
        for row in rows:
            conditions = json.loads(row["condition_json"] or "{}")
            if features and not all(self._condition_matches(features.get(key), value) for key, value in conditions.items()):
                continue
            if conditions and not features:
                continue
            results.append(
                GrammarRule(
                    rule_id=str(row["id"]),
                    parent=str(row["parent_symbol"]),
                    children=tuple(json.loads(row["child_symbols_json"])),
                    conditions=conditions,
                    priority=int(row["priority"]),
                )
            )
        return results

    def assert_template_free(self) -> None:
        """Fail closed if a sentence/template sneaks into the language store."""

        table_names = {
            str(row["name"])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        forbidden_tables = {"constructions", "construction_slots", "graph_edges"}
        if table_names & forbidden_tables:
            raise ValueError("Legacy sentence-construction tables are not permitted")
        for row in self.connection.execute("SELECT id, surface FROM atoms"):
            self._validate_atom_surface(str(row["surface"]))
        for row in self.connection.execute("SELECT id, child_symbols_json FROM grammar_rules"):
            self._validate_grammar_symbols(tuple(json.loads(row["child_symbols_json"])))

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def applicable_gate_rules(self, features: Mapping[str, Any]) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM gate_rules ORDER BY priority DESC, id ASC"
        ).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            condition = json.loads(row["condition_json"])
            if all(self._condition_matches(features.get(key), expected) for key, expected in condition.items()):
                results.append(
                    {
                        "id": str(row["id"]),
                        "lock_pools": json.loads(row["lock_pools_json"]),
                        "allow_pools": json.loads(row["allow_pools_json"]),
                        "priority": int(row["priority"]),
                    }
                )
        return results

    @staticmethod
    def _condition_matches(actual: Any, expected: Any) -> bool:
        if isinstance(expected, list):
            return actual in expected
        if isinstance(expected, dict):
            if "gte" in expected and not (actual is not None and actual >= expected["gte"]):
                return False
            if "lte" in expected and not (actual is not None and actual <= expected["lte"]):
                return False
            if "eq" in expected and actual != expected["eq"]:
                return False
            return True
        return actual == expected

    # ------------------------------------------------------------------
    # Lexical learning persistence
    # ------------------------------------------------------------------

    def touch_unknown_term(self, surface: str, normalized: str, turn_index: int) -> int:
        row = self.connection.execute(
            "SELECT term_id FROM learned_terms WHERE normalized = ?",
            (normalized,),
        ).fetchone()
        with self.connection:
            if row:
                term_id = int(row["term_id"])
                self.connection.execute(
                    """
                    UPDATE learned_terms
                    SET preferred_surface = ?, last_seen_turn = ?,
                        occurrence_count = occurrence_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE term_id = ?
                    """,
                    (surface, int(turn_index), term_id),
                )
                return term_id
            cursor = self.connection.execute(
                """
                INSERT INTO learned_terms(
                    normalized, preferred_surface, first_seen_turn, last_seen_turn
                ) VALUES (?, ?, ?, ?)
                """,
                (normalized, surface, int(turn_index), int(turn_index)),
            )
            return int(cursor.lastrowid)

    def term_row(self, normalized: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM learned_terms WHERE normalized = ?",
            (normalized,),
        ).fetchone()
        return dict(row) if row else None

    def add_lexical_evidence(
        self,
        *,
        term_id: int,
        evidence_kind: str,
        raw_context: str,
        raw_explanation: str,
        context_hash: str,
        context_features: Mapping[str, Any],
        vector: AffectVector,
        semantic_class: str,
        polarity: int,
        support_weight: float,
        contradiction_weight: float,
        observed_turn: int,
        sense_id: Optional[int] = None,
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO lexical_evidence(
                    term_id, sense_id, evidence_kind, raw_context,
                    raw_explanation, context_hash, context_features_json,
                    vector_json, interpreted_vector_json, semantic_class, polarity, support_weight,
                    contradiction_weight, observed_turn
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(term_id),
                    sense_id,
                    evidence_kind,
                    raw_context,
                    raw_explanation,
                    context_hash,
                    json.dumps(dict(context_features), sort_keys=True),
                    json.dumps(vector.to_dict(), sort_keys=True),
                    json.dumps(vector.to_dict(), sort_keys=True),
                    semantic_class,
                    int(polarity),
                    float(support_weight),
                    float(contradiction_weight),
                    int(observed_turn),
                ),
            )
            return int(cursor.lastrowid)

    def lexical_evidence(self, term_id: int) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM lexical_evidence WHERE term_id = ? ORDER BY evidence_id",
            (int(term_id),),
        ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["context_features"] = json.loads(item.pop("context_features_json") or "{}")
            item["vector"] = AffectVector(**json.loads(item.pop("vector_json")))
            item["interpreted_vector"] = AffectVector(
                **json.loads(item.pop("interpreted_vector_json"))
            )
            result.append(item)
        return result

    def replace_senses(
        self,
        term_id: int,
        senses: Sequence[Mapping[str, Any]],
        *,
        status: str,
    ) -> None:
        with self.connection:
            existing = {
                int(row["sense_index"]): row
                for row in self.connection.execute(
                    "SELECT * FROM learned_senses WHERE term_id = ?",
                    (int(term_id),),
                )
            }
            active_indexes: set[int] = set()
            for item in senses:
                index = int(item.get("sense_index", 1))
                active_indexes.add(index)
                previous = existing.get(index)
                version = int(previous["version"]) + 1 if previous else 1
                payload = (
                    item.get("scope_type", "session"),
                    item.get("scope_id", "default"),
                    item.get("part_of_speech", "unknown"),
                    item.get("semantic_class", "unknown"),
                    item.get("register", "neutral"),
                    float(item.get("confidence", 0.0)),
                    item.get("status", "provisional"),
                    version,
                    float(item.get("support_weight", 0.0)),
                    float(item.get("contradiction_weight", 0.0)),
                    json.dumps(dict(item.get("vector", AffectVector().to_dict())), sort_keys=True),
                    json.dumps(dict(item.get("conditions", {})), sort_keys=True),
                )
                if previous:
                    self.connection.execute(
                        """
                        UPDATE learned_senses SET
                            scope_type=?, scope_id=?, part_of_speech=?,
                            semantic_class=?, register_name=?, confidence=?,
                            status=?, version=?, support_weight=?,
                            contradiction_weight=?, vector_json=?, conditions_json=?,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE sense_id=?
                        """,
                        (*payload, int(previous["sense_id"])),
                    )
                else:
                    self.connection.execute(
                        """
                        INSERT INTO learned_senses(
                            term_id, sense_index, scope_type, scope_id,
                            part_of_speech, semantic_class, register_name,
                            confidence, status, version, support_weight,
                            contradiction_weight, vector_json, conditions_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (int(term_id), index, *payload),
                    )
            for index, row in existing.items():
                if index not in active_indexes:
                    self.connection.execute(
                        "UPDATE learned_senses SET status='deprecated', updated_at=CURRENT_TIMESTAMP WHERE sense_id=?",
                        (int(row["sense_id"]),),
                    )
            self.connection.execute(
                """
                UPDATE learned_terms SET status=?, active_sense_count=?,
                    updated_at=CURRENT_TIMESTAMP WHERE term_id=?
                """,
                (status, len(active_indexes), int(term_id)),
            )
            # Every old occurrence is now interpretable under the new version.
            max_version = max(
                (
                    int(row["version"])
                    for row in self.connection.execute(
                        "SELECT version FROM learned_senses WHERE term_id=? AND status!='deprecated'",
                        (int(term_id),),
                    )
                ),
                default=0,
            )
            self._reanalyze_evidence(int(term_id), max_version)

    def _reanalyze_evidence(self, term_id: int, version: int) -> None:
        """Re-run every old context through the current sense hypotheses."""

        senses = list(
            self.connection.execute(
                "SELECT * FROM learned_senses WHERE term_id=? AND status!='deprecated' ORDER BY sense_index",
                (int(term_id),),
            )
        )
        evidence_rows = list(
            self.connection.execute(
                "SELECT * FROM lexical_evidence WHERE term_id=? ORDER BY evidence_id",
                (int(term_id),),
            )
        )
        for evidence in evidence_rows:
            base = AffectVector(**json.loads(evidence["vector_json"]))
            if not senses:
                self.connection.execute(
                    "UPDATE lexical_evidence SET interpreted_vector_json=?, sense_id=NULL, reanalyzed_version=? WHERE evidence_id=?",
                    (json.dumps(base.to_dict(), sort_keys=True), int(version), int(evidence["evidence_id"])),
                )
                continue
            polarity = int(evidence["polarity"])
            desired = "positive" if polarity > 0 else "negative" if polarity < 0 else ""
            matching = [
                row
                for row in senses
                if json.loads(row["conditions_json"] or "{}").get("context_polarity") == desired
            ]
            candidates = matching or senses
            selected = min(
                candidates,
                key=lambda row: AffectVector(**json.loads(row["vector_json"])).distance(base),
            )
            sense_vector = AffectVector(**json.loads(selected["vector_json"]))
            confidence = float(selected["confidence"] )
            weight = min(0.82, max(0.20, 0.25 + confidence * 0.55))
            interpreted = AffectVector(**{
                axis: round(getattr(base, axis) * (1.0 - weight) + getattr(sense_vector, axis) * weight)
                for axis in ("v", "a", "d", "u", "g", "w", "i")
            })
            self.connection.execute(
                """
                UPDATE lexical_evidence SET interpreted_vector_json=?, sense_id=?,
                    reanalyzed_version=? WHERE evidence_id=?
                """,
                (
                    json.dumps(interpreted.to_dict(), sort_keys=True),
                    int(selected["sense_id"]),
                    int(version),
                    int(evidence["evidence_id"]),
                ),
            )

    def learned_senses(
        self,
        normalized: str,
        *,
        min_confidence: float = 0.0,
        include_disputed: bool = True,
    ) -> List[LearnedSenseRecord]:
        query = """
            SELECT s.*, t.preferred_surface
            FROM learned_senses s
            JOIN learned_terms t ON t.term_id = s.term_id
            WHERE t.normalized = ? AND s.confidence >= ? AND s.status != 'deprecated'
        """
        params: List[Any] = [normalized, float(min_confidence)]
        if not include_disputed:
            query += " AND s.status != 'disputed'"
        query += " ORDER BY s.confidence DESC, s.sense_index ASC"
        rows = self.connection.execute(query, params).fetchall()
        return [
            LearnedSenseRecord(
                sense_id=int(row["sense_id"]),
                term_id=int(row["term_id"]),
                sense_index=int(row["sense_index"]),
                surface=str(row["preferred_surface"]),
                part_of_speech=str(row["part_of_speech"]),
                semantic_class=str(row["semantic_class"]),
                register=str(row["register_name"]),
                confidence=float(row["confidence"]),
                status=str(row["status"]),
                version=int(row["version"]),
                support_weight=float(row["support_weight"]),
                contradiction_weight=float(row["contradiction_weight"]),
                vector=AffectVector(**json.loads(row["vector_json"])),
                conditions=json.loads(row["conditions_json"] or "{}"),
            )
            for row in rows
        ]

    def learned_terms_summary(self) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM learned_terms ORDER BY updated_at DESC, normalized"
        ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["senses"] = [
                {
                    "sense_index": sense.sense_index,
                    "semantic_class": sense.semantic_class,
                    "confidence": sense.confidence,
                    "status": sense.status,
                    "version": sense.version,
                    "vector": sense.vector.to_dict(),
                    "conditions": sense.conditions,
                }
                for sense in self.learned_senses(str(row["normalized"]))
            ]
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Resolver observations
    # ------------------------------------------------------------------

    def record_resolver_observation(
        self,
        *,
        request_hash: str,
        resolver_name: str,
        predicate: str,
        value: Mapping[str, Any],
        source_name: str,
        certainty: int,
        observed_at: str,
        expires_at: Optional[str],
        metadata: Mapping[str, Any],
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO resolver_observations(
                    request_hash, resolver_name, predicate, value_json,
                    source_name, certainty, observed_at, expires_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_hash,
                    resolver_name,
                    predicate,
                    json.dumps(dict(value), sort_keys=True),
                    source_name,
                    max(0, min(255, int(certainty))),
                    observed_at,
                    expires_at,
                    json.dumps(dict(metadata), sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------
    # Mathematical trajectory persistence
    # ------------------------------------------------------------------

    def record_trajectory(
        self,
        *,
        input_hash: str,
        response_hash: str,
        incoming_act: str,
        response_act: str,
        context_key: str,
        input_vector: AffectVector,
        state_before: AffectVector,
        target_vector: AffectVector,
        response_vector: AffectVector,
        predicted_after: AffectVector,
        profile_id: Optional[str],
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO trajectory_turns(
                    input_hash, response_hash, incoming_act, response_act,
                    context_key, input_vector_json, state_before_json,
                    target_vector_json, response_vector_json,
                    predicted_after_json, profile_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    input_hash,
                    response_hash,
                    incoming_act,
                    response_act,
                    context_key,
                    json.dumps(input_vector.to_dict(), sort_keys=True),
                    json.dumps(state_before.to_dict(), sort_keys=True),
                    json.dumps(target_vector.to_dict(), sort_keys=True),
                    json.dumps(response_vector.to_dict(), sort_keys=True),
                    json.dumps(predicted_after.to_dict(), sort_keys=True),
                    profile_id,
                ),
            )
            return int(cursor.lastrowid)

    def pending_trajectory(self) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM trajectory_turns WHERE finalized=0 ORDER BY trajectory_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def finalize_trajectory(
        self,
        trajectory_id: int,
        *,
        observed_next: AffectVector,
        reaction_vector: AffectVector,
        residual: Mapping[str, float],
        success: float,
    ) -> None:
        with self.connection:
            row = self.connection.execute(
                "SELECT context_key FROM trajectory_turns WHERE trajectory_id=?",
                (int(trajectory_id),),
            ).fetchone()
            if not row:
                return
            context_key = str(row["context_key"])
            self.connection.execute(
                """
                UPDATE trajectory_turns SET observed_next_json=?,
                    reaction_vector_json=?, residual_json=?, finalized=1
                WHERE trajectory_id=?
                """,
                (
                    json.dumps(observed_next.to_dict(), sort_keys=True),
                    json.dumps(reaction_vector.to_dict(), sort_keys=True),
                    json.dumps({axis: float(residual.get(axis, 0.0)) for axis in ("v", "a", "d", "u", "g", "w", "i")}, sort_keys=True),
                    int(trajectory_id),
                ),
            )
            current = self.connection.execute(
                "SELECT * FROM transition_stats WHERE context_key=?",
                (context_key,),
            ).fetchone()
            if current:
                count = int(current["sample_count"])
                mean_residual = {
                    axis: float(value)
                    for axis, value in json.loads(current["mean_residual_json"]).items()
                }
                mean_reaction = AffectVector(**json.loads(current["mean_reaction_json"]))
                new_count = count + 1
                residual_mean = self._running_signed_mean(mean_residual, residual, count, new_count)
                reaction_mean = self._running_vector_mean(mean_reaction, reaction_vector, count, new_count)
                success_mean = (float(current["success_mean"]) * count + float(success)) / new_count
                self.connection.execute(
                    """
                    UPDATE transition_stats SET sample_count=?, mean_residual_json=?,
                        mean_reaction_json=?, success_mean=?, updated_at=CURRENT_TIMESTAMP
                    WHERE context_key=?
                    """,
                    (
                        new_count,
                        json.dumps(residual_mean, sort_keys=True),
                        json.dumps(reaction_mean.to_dict(), sort_keys=True),
                        success_mean,
                        context_key,
                    ),
                )
            else:
                self.connection.execute(
                    """
                    INSERT INTO transition_stats(
                        context_key, sample_count, mean_residual_json,
                        mean_reaction_json, success_mean
                    ) VALUES (?, 1, ?, ?, ?)
                    """,
                    (
                        context_key,
                        json.dumps({axis: float(residual.get(axis, 0.0)) for axis in ("v", "a", "d", "u", "g", "w", "i")}, sort_keys=True),
                        json.dumps(reaction_vector.to_dict(), sort_keys=True),
                        float(success),
                    ),
                )

    @staticmethod
    def _running_signed_mean(
        previous: Mapping[str, float],
        current: Mapping[str, float],
        previous_count: int,
        new_count: int,
    ) -> Dict[str, float]:
        return {
            axis: (float(previous.get(axis, 0.0)) * previous_count + float(current.get(axis, 0.0)))
            / max(1, new_count)
            for axis in ("v", "a", "d", "u", "g", "w", "i")
        }

    @staticmethod
    def _running_vector_mean(
        previous: AffectVector,
        current: AffectVector,
        previous_count: int,
        new_count: int,
    ) -> AffectVector:
        values = {}
        for axis in ("v", "a", "d", "u", "g", "w", "i"):
            values[axis] = round(
                (getattr(previous, axis) * previous_count + getattr(current, axis))
                / max(1, new_count)
            )
        return AffectVector(**values)

    def transition_stat(self, context_key: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM transition_stats WHERE context_key=?",
            (context_key,),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["mean_residual"] = {axis: float(value) for axis, value in json.loads(item.pop("mean_residual_json")).items()}
        item["mean_reaction"] = AffectVector(**json.loads(item.pop("mean_reaction_json")))
        return item

    # ------------------------------------------------------------------
    # Corpus trajectory profiles and fingerprints
    # ------------------------------------------------------------------

    def upsert_corpus_profile(self, profile: Mapping[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO corpus_profiles(
                    profile_id, name, source_hash, quote_count, trajectory_blob,
                    delta_blob, centroid_json, variance_json, delta_centroid_json,
                    act_distribution_json, transition_matrix_json, fingerprint, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    name=excluded.name,
                    source_hash=excluded.source_hash,
                    quote_count=excluded.quote_count,
                    trajectory_blob=excluded.trajectory_blob,
                    delta_blob=excluded.delta_blob,
                    centroid_json=excluded.centroid_json,
                    variance_json=excluded.variance_json,
                    delta_centroid_json=excluded.delta_centroid_json,
                    act_distribution_json=excluded.act_distribution_json,
                    transition_matrix_json=excluded.transition_matrix_json,
                    fingerprint=excluded.fingerprint,
                    metadata_json=excluded.metadata_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    profile["profile_id"],
                    profile["name"],
                    profile["source_hash"],
                    int(profile["quote_count"]),
                    bytes(profile["trajectory_blob"]),
                    bytes(profile["delta_blob"]),
                    json.dumps(dict(profile["centroid"]), sort_keys=True),
                    json.dumps(dict(profile["variance"]), sort_keys=True),
                    json.dumps(dict(profile["delta_centroid"]), sort_keys=True),
                    json.dumps(dict(profile["act_distribution"]), sort_keys=True),
                    json.dumps(dict(profile["transition_matrix"]), sort_keys=True),
                    profile["fingerprint"],
                    json.dumps(dict(profile.get("metadata", {})), sort_keys=True),
                ),
            )
            self.connection.execute(
                "DELETE FROM trajectory_chunks WHERE profile_id=?",
                (profile["profile_id"],),
            )
            for chunk in profile.get("chunks", []):
                self.connection.execute(
                    """
                    INSERT INTO trajectory_chunks(
                        profile_id, start_index, window_size, vector_blob,
                        delta_blob, vector_hash, delta_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile["profile_id"],
                        int(chunk["start_index"]),
                        int(chunk["window_size"]),
                        bytes(chunk["vector_blob"]),
                        bytes(chunk["delta_blob"]),
                        chunk["vector_hash"],
                        chunk["delta_hash"],
                    ),
                )

    def get_corpus_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM corpus_profiles WHERE profile_id=?",
            (profile_id,),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        for field in (
            "centroid_json",
            "variance_json",
            "delta_centroid_json",
            "act_distribution_json",
            "transition_matrix_json",
            "metadata_json",
        ):
            item[field[:-5]] = json.loads(item.pop(field) or "{}")
        return item

    def list_corpus_profiles(self) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT profile_id, name, quote_count, fingerprint, created_at, updated_at FROM corpus_profiles ORDER BY name"
        ).fetchall()
        return [dict(row) for row in rows]

    def corpus_chunks(
        self,
        profile_id: str,
        *,
        window_size: Optional[int] = None,
        limit: int = 10_000,
    ) -> List[Dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100_000:
            raise ValueError("corpus chunk limit must be an integer from 1 to 100000")
        query = "SELECT * FROM trajectory_chunks WHERE profile_id=?"
        params: List[Any] = [profile_id]
        if window_size is not None:
            query += " AND window_size=?"
            params.append(int(window_size))
        query += " ORDER BY window_size DESC, start_index ASC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    # ------------------------------------------------------------------
    # Snapshot overlay and diagnostics
    # ------------------------------------------------------------------

    def export_overlay(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"version": 1, "tables": {}}
        for table in self.OVERLAY_TABLES:
            columns = [
                str(row["name"])
                for row in self.connection.execute(f"PRAGMA table_info({table})")
            ]
            rows: List[Dict[str, Any]] = []
            for row in self.connection.execute(f"SELECT * FROM {table}"):
                item: Dict[str, Any] = {}
                for column in columns:
                    value = row[column]
                    if isinstance(value, bytes):
                        value = {"__bytes__": list(value)}
                    item[column] = value
                rows.append(item)
            result["tables"][table] = rows
        return result

    def import_overlay(self, data: Mapping[str, Any]) -> None:
        if int(data.get("version", 0)) != 1:
            raise ValueError("Unsupported language overlay version")
        tables = dict(data.get("tables", {}))
        with self.connection:
            # Delete children before parents and insert parents before children;
            # this keeps snapshot import valid even when SQLite refuses to
            # change foreign-key mode inside a transaction.
            for table in reversed(self.OVERLAY_TABLES):
                self.connection.execute(f"DELETE FROM {table}")
            for table in self.OVERLAY_TABLES:
                for raw in tables.get(table, []):
                    item = dict(raw)
                    for key, value in list(item.items()):
                        if isinstance(value, Mapping) and "__bytes__" in value:
                            item[key] = bytes(value["__bytes__"])
                    columns = list(item)
                    placeholders = ",".join("?" for _ in columns)
                    self.connection.execute(
                        f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders})",
                        [item[column] for column in columns],
                    )

    def schema_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for table in (
            "atoms",
            "grammar_rules",
            "gate_rules",
            "learned_terms",
            "learned_senses",
            "lexical_evidence",
            "resolver_observations",
            "trajectory_turns",
            "transition_stats",
            "corpus_profiles",
            "trajectory_chunks",
        ):
            row = self.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            summary[table] = int(row["n"] if row else 0)
        summary["template_tables"] = 0
        return summary
