import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_locks_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}


class StoreError(Exception):
    pass


def _lock_for(run_id: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(run_id, threading.RLock())


@contextmanager
def locked(run_id: str):
    lock = _lock_for(run_id)
    with lock:
        yield


def _path(run_id: str) -> Path:
    if not _ID_PATTERN.fullmatch(run_id):
        raise ValueError("Некоректний ідентифікатор звернення.")
    return RUNS_DIR / f"{run_id}.json"


def save(run: Run) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    destination = _path(run.run_id)
    temporary = destination.with_name(f".{run.run_id}.{uuid.uuid4().hex}.tmp")
    run.updated_at = datetime.now().astimezone().isoformat()
    payload = json.dumps(asdict(run), ensure_ascii=False, separators=(",", ":"), default=str)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise StoreError(f"Не вдалося зберегти звернення: {exc}") from exc


def load(run_id: str) -> Run | None:
    path = _path(run_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"Не вдалося прочитати звернення {run_id}: {exc}") from exc
    try:
        payload["steps"] = [Step(**item) for item in payload.get("steps", [])]
        return Run(**payload)
    except (TypeError, KeyError) as exc:
        raise StoreError(f"Файл звернення {run_id} має некоректний формат.") from exc


def list_runs(status: str | None = None) -> list[Run]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in RUNS_DIR.glob("*.json"):
        run = load(path.stem)
        if run is not None and (status is None or run.status == status):
            runs.append(run)
    return sorted(runs, key=lambda run: run.updated_at or run.created_at or "", reverse=True)
