"""Opt-in principal-bound persistence around the existing Clanker runtime.

This is not authentication, a new generator, or a shared semantic store.
Only trusted server code may supply ``principal``. Never use a principal copied
from an unauthenticated request body. The keyed envelope authenticates integrity,
not factual truth, and does NOT encrypt the user's snapshot.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any, Callable, Iterator
from urllib.parse import quote, unquote


class ScopeError(ValueError):
    """The object is invalid, foreign, stale, or incompatible with this scope."""


class ScopeBusy(TimeoutError):
    """A bounded same-compartment lock wait expired."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _text(value: Any, name: str, maximum: int) -> str:
    if (not isinstance(value, str) or not value or not value.strip()
            or len(value.encode("utf-8")) > maximum
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)):
        raise ScopeError("invalid " + name)
    # Exact opaque IDs, not lower-cased display names. No silent normalization.
    return value


def runtime_fingerprint() -> str:
    """Bind the installed source/assets, once per service construction.

    No held-out text is inspected. A different code/pack/configuration requires
    explicit migration, not relabelling a previously signed snapshot.
    """
    import clanker_lm
    root = Path(clanker_lm.__file__).resolve().parent.parent
    hashes = {}
    for folder in ("clanker_lm", "engine"):
        for path in sorted((root / folder).rglob("*")):
            if (path.is_file() and "__pycache__" not in path.parts
                    and "tests" not in path.relative_to(root).parts
                    and path.suffix in {".py", ".json", ".sqlite3", ".sql", ".js", ".css", ".html"}):
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["private_memory_service"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return digest(hashes)


class PrivateMemoryService:
    """One full, isolated snapshot per (trusted principal, compartment).

    Creates/restores a fresh in-memory runtime inside a per-compartment POSIX
    file lock. No mutable runtime, SQLite connection or learned overlay crosses
    a scope boundary. Successful writes atomically replace one authenticated
    envelope; failed turns leave the last committed envelope unchanged.

    This deliberately favors a simple correctness boundary over a live-runtime
    cache. It supports multiple local workers on the same local filesystem,
    not a distributed/NFS locking protocol. The storage directory and key must
    be accessible only to the trusted application account.
    """
    FORMAT = "clanker-private-memory-v1"

    def __init__(self, root: str | Path, *, secret: bytes,
                 configuration_id: str,
                 factory: Callable[[str], Any] | None = None,
                 restore_factory: Callable[[dict], Any] | None = None,
                 max_snapshot_bytes: int = 16 * 1024 * 1024,
                 max_turns: int = 10_000,
                 lock_timeout: float = 5.0) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ScopeError("supply a private server key of at least 32 bytes")
        if type(max_snapshot_bytes) is not int or not 4096 <= max_snapshot_bytes <= 128*1024*1024:
            raise ScopeError("invalid snapshot budget")
        if type(max_turns) is not int or not 1 <= max_turns <= 100_000:
            raise ScopeError("invalid turn budget")
        if isinstance(lock_timeout, bool) or not isinstance(lock_timeout, (float, int)) or not 0 < lock_timeout <= 60:
            raise ScopeError("invalid lock timeout")
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("this local storage adapter requires POSIX no-follow files and flock")
        self._secret = secret
        self.configuration_id = _text(configuration_id, "configuration identity", 512)
        self.runtime_id = runtime_fingerprint()
        self.max_snapshot_bytes = max_snapshot_bytes
        self.max_turns = max_turns
        self.lock_timeout = float(lock_timeout)
        self.root = Path(root).absolute()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ScopeError("storage root must be a private application-owned directory")
        from clanker_lm import ClankerLM
        self._factory = factory or (lambda namespace: ClankerLM(learning_scope=namespace))
        self._restore = restore_factory or ClankerLM.from_dict

    def _mac(self, domain: str, value: Any) -> str:
        return hmac.new(self._secret, domain.encode() + b"\x00" + canonical(value), hashlib.sha256).hexdigest()

    def namespace(self, principal: str, compartment: str = "personal") -> str:
        """Opaque routing ID; possession of this string is NOT authorization."""
        principal = _text(principal, "authenticated principal", 256)
        compartment = _text(compartment, "compartment", 128)
        return self._mac("namespace-v1", [principal, compartment])

    def _path(self, namespace: str, extension: str) -> Path:
        return self.root / (namespace + extension)

    def _check_fd(self, fd: int) -> None:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o077 or info.st_nlink != 1):
            raise ScopeError("unsafe compartment file")

    @contextmanager
    def _locked(self, namespace: str) -> Iterator[None]:
        import fcntl
        fd = os.open(self._path(namespace, ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            self._check_fd(fd)
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ScopeBusy("compartment is busy")
                    time.sleep(0.005)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _read(self, namespace: str) -> dict | None:
        try:
            fd = os.open(self._path(namespace, ".json"), os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        try:
            self._check_fd(fd)
            if os.fstat(fd).st_size > self.max_snapshot_bytes:
                raise ScopeError("snapshot budget exceeded")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(self.max_snapshot_bytes + 1)
            return self._decode(raw, namespace)
        finally:
            os.close(fd)

    def _decode(self, raw: bytes, namespace: str) -> dict:
        if not isinstance(raw, bytes) or len(raw) > self.max_snapshot_bytes:
            raise ScopeError("invalid snapshot payload")
        try:
            envelope = json.loads(raw)
            if not isinstance(envelope, dict) or set(envelope) != {
                "format", "namespace", "runtime_id", "configuration_id", "generation", "state", "mac"
            }:
                raise ScopeError("invalid private snapshot schema")
            body = {k: v for k, v in envelope.items() if k != "mac"}
            if (envelope["format"] != self.FORMAT or envelope["namespace"] != namespace
                    or envelope["runtime_id"] != self.runtime_id
                    or envelope["configuration_id"] != self.configuration_id
                    or type(envelope["generation"]) is not int or not 0 <= envelope["generation"] < 2**63
                    or not isinstance(envelope["state"], dict)
                    or not isinstance(envelope["mac"], str)
                    or not hmac.compare_digest(envelope["mac"], self._mac("snapshot-v1", body))):
                raise ScopeError("foreign, changed or incompatible snapshot")
            self._check_state(envelope["state"], namespace)
            return envelope
        except (UnicodeError, json.JSONDecodeError, TypeError, OverflowError, RecursionError) as exc:
            raise ScopeError("invalid private snapshot encoding") from exc

    def _check_state(self, state: dict, namespace: str) -> None:
        if state.get("learner", {}).get("scope_id") != namespace:
            raise ScopeError("learned sense scope does not match memory scope")
        usage = state.get("usage_learning")
        if usage is not None and usage.get("scope") != namespace:
            raise ScopeError("usage scope does not match memory scope")
        tables = state.get("language_overlay", {}).get("tables", {})
        for sense in tables.get("learned_senses", []):
            if sense.get("scope_id") != namespace:
                raise ScopeError("foreign learned sense in private store")
        turn = state.get("memory", {}).get("turn_index")
        if type(turn) is not int or not 0 <= turn <= self.max_turns:
            raise ScopeError("invalid or over-budget conversation")

    def _envelope(self, namespace: str, generation: int, state: dict) -> dict:
        self._check_state(state, namespace)
        body = dict(format=self.FORMAT, namespace=namespace, runtime_id=self.runtime_id,
                    configuration_id=self.configuration_id, generation=generation, state=state)
        return {**body, "mac": self._mac("snapshot-v1", body)}

    def _write(self, envelope: dict) -> None:
        raw = canonical(envelope)
        if len(raw) > self.max_snapshot_bytes:
            raise ScopeError("snapshot budget exceeded; no state committed")
        fd, temp = tempfile.mkstemp(prefix=".pending-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self._path(envelope["namespace"], ".json"))
            dirfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    @contextmanager
    def _runtime(self, principal: str, compartment: str, *, write: bool = False):
        namespace = self.namespace(principal, compartment)
        with self._locked(namespace):
            previous = self._read(namespace)
            runtime = self._restore(deepcopy(previous["state"])) if previous else self._factory(namespace)
            try:
                if runtime.store.path != ":memory:":
                    raise ScopeError("service requires a private in-memory runtime; shared databases are forbidden")
                state = runtime.to_dict()
                self._check_state(state, namespace)
                if not previous and (state["memory"]["turn_index"] or state["memory"].get("events")):
                    raise ScopeError("new compartments require fresh runtimes")
                generation = previous["generation"] if previous else 0
                if write and generation >= 2**63 - 2:
                    raise ScopeError("generation budget exceeded")
                yield namespace, generation, runtime
                if write:
                    self._write(self._envelope(namespace, generation + 1, runtime.to_dict()))
            finally:
                runtime.close()

    def process(self, principal: str, text: str, *, compartment: str = "personal") -> dict:
        _text(text, "message", 8192)
        with self._runtime(principal, compartment, write=True) as (namespace, generation, runtime):
            if runtime.memory.turn_index >= self.max_turns:
                raise ScopeError("conversation turn limit reached")
            result = runtime.process(text).to_dict()
            body = {"namespace": namespace, "generation": generation + 1, "turn": result,
                    "runtime_id": self.runtime_id, "configuration_id": self.configuration_id}
            response = {**body, "mac": self._mac("response-v1", body)}
        return response

    def verify_response(self, principal: str, response: dict, *, compartment: str = "personal") -> bool:
        namespace = self.namespace(principal, compartment)
        if not isinstance(response, dict) or set(response) != {"namespace", "generation", "turn", "runtime_id", "configuration_id", "mac"}:
            return False
        body = {k: v for k,v in response.items() if k != "mac"}
        return (response["namespace"] == namespace
                and response["runtime_id"] == self.runtime_id
                and response["configuration_id"] == self.configuration_id
                and isinstance(response["mac"], str)
                and hmac.compare_digest(response["mac"], self._mac("response-v1", body)))

    def export_snapshot(self, principal: str, *, compartment: str = "personal") -> bytes:
        """Private authenticated export; contains evidence and must not be public telemetry."""
        with self._runtime(principal, compartment) as (namespace, generation, runtime):
            return canonical(self._envelope(namespace, generation, runtime.to_dict()))

    def restore_snapshot(self, principal: str, payload: bytes, *, compartment: str = "personal") -> None:
        namespace = self.namespace(principal, compartment)
        incoming = self._decode(payload, namespace)  # Check BEFORE constructing any runtime.
        with self._locked(namespace):
            previous = self._read(namespace)
            if previous and (incoming["generation"] < previous["generation"]
                             or (incoming["generation"] == previous["generation"] and incoming != previous)):
                raise ScopeError("stale or conflicting restore refused")
            with self._restore(deepcopy(incoming["state"])) as runtime:
                self._check_state(runtime.to_dict(), namespace)
                if runtime.store.path != ":memory:":
                    raise ScopeError("shared database restore refused")
            self._write(incoming)

    def reset(self, principal: str, *, compartment: str = "personal") -> None:
        namespace = self.namespace(principal, compartment)
        with self._locked(namespace):
            previous = self._read(namespace)
            generation = previous["generation"] + 1 if previous else 1
            with self._factory(namespace) as runtime:
                if runtime.memory.turn_index or runtime.store.path != ":memory:":
                    raise ScopeError("reset requires a fresh private runtime")
                self._write(self._envelope(namespace, generation, runtime.to_dict()))
        # Keep the generation tombstone; reset does not erase external backups.

    def lexicon(self, principal: str, *, compartment: str = "personal") -> dict:
        with self._runtime(principal, compartment) as (namespace, generation, runtime):
            return {"namespace": namespace, "generation": generation,
                    "terms": runtime.learned_lexicon()}

    @staticmethod
    def _qualify(namespace: str, local_id: str) -> str:
        return "clanker:" + namespace + "/" + quote(local_id, safe="")

    def _local(self, namespace: str, reference: str) -> str:
        reference = _text(reference, "qualified reference", 2048)
        prefix = "clanker:" + namespace + "/"
        if not reference.startswith(prefix):
            raise ScopeError("reference is not in the caller's compartment")
        local_id = unquote(reference[len(prefix):])
        if not local_id or self._qualify(namespace, local_id) != reference:
            raise ScopeError("noncanonical private reference")
        return local_id

    def entities(self, principal: str, *, compartment: str = "personal") -> dict:
        with self._runtime(principal, compartment) as (namespace, generation, runtime):
            return {"namespace": namespace, "generation": generation,
                    "entities": [{"reference": self._qualify(namespace, "entity:" + e.entity_id),
                                  "local_id": e.entity_id, "label": e.canonical_name,
                                  "owner_reference": self._qualify(namespace, "entity:" + e.owner_id) if e.owner_id else None}
                                 for e in runtime.memory.entities.values()]}

    def entity(self, principal: str, reference: str, *, compartment: str = "personal") -> dict:
        namespace = self.namespace(principal, compartment)
        local_id = self._local(namespace, reference)
        if not local_id.startswith("entity:"):
            raise ScopeError("entity reference required")
        with self._runtime(principal, compartment) as (_, generation, runtime):
            entity = runtime.memory.get_entity(local_id[len("entity:"):])
            if entity is None:
                raise ScopeError("unknown private entity")
            return {"namespace": namespace, "generation": generation,
                    "reference": reference, "record": entity.to_dict()}

    def graph(self, principal: str, *, compartment: str = "personal", root: str | None = None,
              depth: int = 2, limit: int = 150) -> dict:
        namespace = self.namespace(principal, compartment)
        local_root = "entity:user" if root is None else self._local(namespace, root)
        with self._runtime(principal, compartment) as (_, generation, runtime):
            if getattr(runtime, "memory_web", None) is None:
                raise ScopeError("this configured runtime has no associative graph")
            graph = runtime.memory_neighborhood(local_root, depth=depth, limit=limit)
            for node in graph["nodes"]:
                node["local_id"] = node["id"]
                node["id"] = self._qualify(namespace, node["id"])
            for edge in graph["edges"]:
                for key in ("id", "source", "target"):
                    if key in edge:
                        edge[key] = self._qualify(namespace, edge[key])
            graph["root"] = self._qualify(namespace, local_root)
            if graph.get("active_episode"):
                graph["active_episode"] = self._qualify(namespace, graph["active_episode"])
            return {"namespace": namespace, "generation": generation,
                    "reference_rule": "Node/edge IDs are qualified; nested evidence IDs are local to this namespace.",
                    "graph": graph}
