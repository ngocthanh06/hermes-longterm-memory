"""hooks/project_catalog.py — the Hermes-independent project→folder fallback
used by scripts/ingest_watcher.py for a Claude-Code-only (no Hermes
Desktop) install."""

import project_catalog


def test_record_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(project_catalog, "CATALOG_FILE", tmp_path / "discovered_projects.json")
    project_catalog.record_project_folder("myrepo", "/work/myrepo")
    assert project_catalog.list_project_folders() == [("myrepo", "/work/myrepo")]


def test_record_skips_write_when_unchanged(tmp_path, monkeypatch):
    catalog_file = tmp_path / "discovered_projects.json"
    monkeypatch.setattr(project_catalog, "CATALOG_FILE", catalog_file)
    project_catalog.record_project_folder("myrepo", "/work/myrepo")
    before = catalog_file.stat().st_mtime_ns
    project_catalog.record_project_folder("myrepo", "/work/myrepo")
    assert catalog_file.stat().st_mtime_ns == before


def test_record_updates_changed_path(tmp_path, monkeypatch):
    monkeypatch.setattr(project_catalog, "CATALOG_FILE", tmp_path / "discovered_projects.json")
    project_catalog.record_project_folder("myrepo", "/work/myrepo")
    project_catalog.record_project_folder("myrepo", "/work/myrepo-moved")
    assert project_catalog.list_project_folders() == [("myrepo", "/work/myrepo-moved")]


def test_record_ignores_empty_args(tmp_path, monkeypatch):
    catalog_file = tmp_path / "discovered_projects.json"
    monkeypatch.setattr(project_catalog, "CATALOG_FILE", catalog_file)
    project_catalog.record_project_folder("", "/work/myrepo")
    project_catalog.record_project_folder("myrepo", "")
    assert not catalog_file.exists()


def test_list_project_folders_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(project_catalog, "CATALOG_FILE", tmp_path / "nope.json")
    assert project_catalog.list_project_folders() == []


def test_list_project_folders_corrupt_file(tmp_path, monkeypatch):
    catalog_file = tmp_path / "discovered_projects.json"
    catalog_file.write_text("not json")
    monkeypatch.setattr(project_catalog, "CATALOG_FILE", catalog_file)
    assert project_catalog.list_project_folders() == []
