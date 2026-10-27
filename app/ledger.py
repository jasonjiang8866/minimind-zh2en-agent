"""Durable source/translation pairs and atomic, source-ordered exports."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from contextlib import contextmanager

from .detector import has_chinese
from .graph import SPECIAL_TOKEN


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@contextmanager
def atomic_writer(path):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    with atomic_writer(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class Ledger:
    def __init__(self, output_path):
        self.output = Path(output_path)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.output) + ".sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY, source_json TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', translated TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, error TEXT
            );
        ''')

    def close(self):
        self.db.close()

    def prepare(self, input_path, pipeline_identity):
        """Snapshot and validate all input before making any model requests.

        The fingerprint is over raw input bytes, including blank lines. A resume
        cannot accidentally mix a changed/reordered source or model configuration.
        """
        existing = dict(self.db.execute("SELECT key, value FROM metadata"))
        if not existing and self.output.exists():
            raise ValueError("Output exists without a verified ledger; choose a new OUTPUT_PATH. Legacy append output cannot be safely resumed.")
        identity = json.dumps(pipeline_identity, sort_keys=True)
        if existing and existing.get("pipeline") != identity:
            raise ValueError("Translation configuration changed; choose a new OUTPUT_PATH")
        hasher = hashlib.sha256()
        count = 0
        with self.db:
            with open(input_path, "rb") as source:
                for line_number, raw in enumerate(source, 1):
                    hasher.update(raw)
                    if not raw.strip():
                        continue
                    try:
                        rec = json.loads(raw)
                    except (ValueError, UnicodeError) as error:
                        raise ValueError(f"Invalid JSON at input line {line_number}") from error
                    if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
                        raise ValueError(f"Input line {line_number} must be an object with a string text field")
                    serialized = json.dumps(rec, ensure_ascii=False, sort_keys=True)
                    if not existing:
                        self.db.execute("INSERT INTO records(id, source_json, source_sha256) VALUES (?, ?, ?)",
                                        (count, serialized, digest(serialized)))
                    count += 1
            fingerprint = hasher.hexdigest()
            if existing:
                if existing.get("input_sha256") != fingerprint or existing.get("total") != str(count):
                    raise ValueError("Input changed; choose a new OUTPUT_PATH")
                actual = self.db.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM records").fetchone()
                if actual != (count, 0 if count else None, count - 1 if count else None):
                    raise ValueError("Ledger record identities do not match the source count")
            else:
                self.db.executemany("INSERT INTO metadata VALUES (?, ?)",
                                    [("input_sha256", fingerprint), ("total", str(count)), ("pipeline", identity)])
        return count

    def pending_after(self, last_id, limit):
        return self.db.execute(
            "SELECT id, source_json FROM records WHERE id > ? AND status != 'success' ORDER BY id LIMIT ?",
            (last_id, limit)).fetchall()

    def record_result(self, record_id, translated=None, error=None):
        with self.db:
            self.db.execute(
                "UPDATE records SET status=?, translated=?, error=?, attempts=attempts+1 WHERE id=?",
                ("success" if error is None else "failed", translated, error, record_id))

    def report(self, status):
        counts = dict(self.db.execute("SELECT status, COUNT(*) FROM records GROUP BY status"))
        meta = dict(self.db.execute("SELECT key, value FROM metadata"))
        result = {"status": status, "input_sha256": meta["input_sha256"], "total": int(meta["total"]),
                  "success": counts.get("success", 0), "failed": counts.get("failed", 0),
                  "pending": counts.get("pending", 0)}
        write_json(str(self.output) + ".summary.json", result)
        return result

    def export(self, run_finished):
        # Always export all source rows, including explicit failures/pending rows.
        # Only a complete run publishes the training JSONL.
        count = 0
        successes = 0
        with atomic_writer(str(self.output) + ".pairs.jsonl") as pairs:
            for rid, raw, sha, status, translated, attempts, error in self.db.execute("SELECT * FROM records ORDER BY id"):
                if rid != count or digest(raw) != sha:
                    raise ValueError("Ledger source integrity check failed")
                record = json.loads(raw)
                if status == "success" and (
                    not isinstance(translated, str)
                    or (record["text"].strip() and not translated.strip())
                    or has_chinese(translated)
                    or SPECIAL_TOKEN.findall(record["text"]) != SPECIAL_TOKEN.findall(translated)
                ):
                    raise ValueError(f"Invalid successful translation for record {rid}")
                pair = {"source_id": rid, "source_sha256": sha, "source": record,
                        "text_en": translated, "status": status, "attempts": attempts, "error": error}
                pairs.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1
                successes += status == "success"
        expected = int(self.db.execute("SELECT value FROM metadata WHERE key='total'").fetchone()[0])
        if count != expected:
            raise ValueError("Source/pair count mismatch")
        complete = run_finished and successes == expected
        if complete:
            with atomic_writer(self.output) as output:
                for raw, translated in self.db.execute("SELECT source_json, translated FROM records ORDER BY id"):
                    record = json.loads(raw)
                    record["text"] = translated
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.report("complete" if complete else "incomplete")
