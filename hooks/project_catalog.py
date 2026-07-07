"""Agent-agnostic project→folder catalog.

Hermes' own ~/.hermes/projects.db is the source of truth for project
folders when Hermes Desktop is installed — but a user who only runs
Claude Code (or a future third adapter) never has it: Hermes Desktop
can't log in with a Claude subscription, so plenty of users only ever
install the Claude Code side (see ARCHITECTURE.md §7b/§8).

Any hook that resolves a project from a REAL cwd folder (source ==
"folder", never the ambient "active"/"default" signals) records the
mapping here, so a host-side background job with no cwd context of its
own — the docs/ auto-ingest watcher (scripts/ingest_watcher.py) — can
still discover every project's folder without Hermes ever being installed.

Best-effort throughout: a broken catalog write must never break a chat turn.
"""

import json
import time
from pathlib import Path

CATALOG_FILE = Path.home() / ".hermes" / "discovered_projects.json"


def record_project_folder(slug: str, path: str) -> None:
    if not slug or not path:
        return
    try:
        catalog = json.loads(CATALOG_FILE.read_text()) if CATALOG_FILE.exists() else {}
    except (OSError, json.JSONDecodeError):
        catalog = {}
    if catalog.get(slug, {}).get("path") == path:
        return  # unchanged — skip the write
    catalog[slug] = {"path": path, "last_seen": time.time()}
    try:
        CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_FILE.write_text(json.dumps(catalog, indent=2))
    except OSError:
        pass


def list_project_folders() -> list[tuple[str, str]]:
    """(slug, path) for every entry — read side, used by ingest_watcher.py."""
    try:
        catalog = json.loads(CATALOG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [(slug, e["path"]) for slug, e in catalog.items() if e.get("path")]
