"""scripts/ingest_watcher.py — project folder discovery only (the file-poll
and HTTP-upload logic needs a live filesystem + service, exercised manually
per README's "docs/ watcher" section, not here)."""

import json
import sqlite3

import ingest_watcher


def _make_hermes_db(path, rows):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY, slug TEXT, archived INTEGER DEFAULT 0,
            primary_path TEXT, created_at INTEGER DEFAULT 0
        );
        CREATE TABLE project_folders (project_id INTEGER, path TEXT);
        """
    )
    for i, (slug, folder_path) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO projects (id, slug, primary_path) VALUES (?, ?, ?)",
            (i, slug, folder_path),
        )
    conn.commit()
    conn.close()


def test_list_hermes_project_folders(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    _make_hermes_db(db, [("erp", "/work/erp")])
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", db)
    assert ingest_watcher.list_hermes_project_folders() == [("erp", "/work/erp")]


def test_list_hermes_project_folders_missing_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", tmp_path / "nope.db")
    assert ingest_watcher.list_hermes_project_folders() == []


def test_list_discovered_project_folders(tmp_path, monkeypatch):
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"myrepo": {"path": "/work/myrepo", "last_seen": 1.0}}))
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert ingest_watcher.list_discovered_project_folders() == [("myrepo", "/work/myrepo")]


def test_merge_combines_both_sources(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    _make_hermes_db(db, [("erp", "/work/erp")])
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"myrepo": {"path": "/work/myrepo"}}))
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", db)
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert set(ingest_watcher.list_project_folders()) == {
        ("erp", "/work/erp"),
        ("myrepo", "/work/myrepo"),
    }


def test_merge_hermes_wins_on_slug_collision(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    _make_hermes_db(db, [("erp", "/work/erp-hermes-anchored")])
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"erp": {"path": "/work/erp-stale-claude-guess"}}))
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", db)
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert ingest_watcher.list_project_folders() == [("erp", "/work/erp-hermes-anchored")]


def test_merge_works_with_no_hermes_at_all(tmp_path, monkeypatch):
    # The exact scenario this fallback exists for: no ~/.hermes/projects.db.
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"myrepo": {"path": "/work/myrepo"}}))
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", tmp_path / "nope.db")
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert ingest_watcher.list_project_folders() == [("myrepo", "/work/myrepo")]
