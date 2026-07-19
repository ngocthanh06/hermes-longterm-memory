#!/usr/bin/env python3
"""Deterministic integrity audit for a Longbrain Qdrant store.

Checks exact collection counts against a full scroll, required fact metadata,
and dangling supersession links. Optional --expect assertions reconcile a
writer's expected count with what is actually present, which catches silent
write drops; no embedding model or LLM is loaded.

Run inside the service container:
  python /repo/scripts/memory_audit.py
  python /repo/scripts/memory_audit.py --expect memories=42 --json
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llamaindex-service"))

from qdrant_client import QdrantClient  # noqa: E402

from app import config, qdrant_setup  # noqa: E402

COLLECTIONS = {
    "memories": config.MEMORIES_COLLECTION,
    "history": config.CHAT_HISTORY_COLLECTION,
    "documents": config.DOCUMENTS_COLLECTION,
}


def scroll_all(client: QdrantClient, collection: str, with_payload=False) -> list:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=with_payload, with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            return points


def scroll_count(client: QdrantClient, collection: str) -> int:
    """Count a full scroll without retaining document/history points."""
    total, offset = 0, None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=False, with_vectors=False,
        )
        total += len(batch)
        if offset is None:
            return total


def parse_expect(values: list[str]) -> dict[str, int]:
    expected = {}
    for value in values:
        try:
            name, raw_count = value.split("=", 1)
            if name not in COLLECTIONS or int(raw_count) < 0:
                raise ValueError
            expected[name] = int(raw_count)
        except ValueError:
            raise SystemExit(
                f"invalid --expect {value!r}; use memories=N, history=N or documents=N"
            )
    return expected


def audit(client: QdrantClient, expected: dict[str, int]) -> dict:
    existing = {item.name for item in client.get_collections().collections}
    report = {"collections": {}, "issues": [], "warnings": []}
    facts = []
    for alias, collection in COLLECTIONS.items():
        if collection not in existing:
            report["collections"][alias] = {"name": collection, "count": 0, "scrolled": 0}
            detail = f"; expected {expected[alias]}" if alias in expected else ""
            report["issues"].append(f"{alias}: required collection missing{detail}")
            continue
        if alias == "memories":
            facts = scroll_all(client, collection, with_payload=True)
            scrolled = len(facts)
        else:
            scrolled = scroll_count(client, collection)
        exact = client.count(collection_name=collection, exact=True).count
        report["collections"][alias] = {
            "name": collection, "count": exact, "scrolled": scrolled
        }
        if exact != scrolled:
            report["issues"].append(f"{alias}: exact count {exact} != scrolled {scrolled}")
        if alias in expected and exact != expected[alias]:
            report["issues"].append(f"{alias}: stored {exact} != expected {expected[alias]}")

    fact_ids = {str(point.id) for point in facts}
    for point in facts:
        payload = point.payload or {}
        # user_id and text are necessary to address/recall a record. Older
        # schemas legitimately omitted type/project_id; runtime deliberately
        # maps those to fact/default, so report them as migration warnings.
        missing = [field for field in ("user_id", "text") if not payload.get(field)]
        if missing:
            report["issues"].append(
                f"memory {point.id}: missing required metadata {','.join(missing)}"
            )
        legacy = [field for field in ("project_id", "type") if not payload.get(field)]
        if legacy:
            report["warnings"].append(
                f"memory {point.id}: legacy metadata defaulted for {','.join(legacy)}"
            )
        target = payload.get("superseded_by")
        if target and str(target) not in fact_ids:
            report["issues"].append(
                f"memory {point.id}: superseded_by target {target} does not exist"
            )

    meta = qdrant_setup.get_meta(client) or {}
    report["last_written_at"] = meta.get("last_written_at")
    report["ok"] = not report["issues"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", action="append", default=[], metavar="COLLECTION=N")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(QdrantClient(url=config.QDRANT_URL), parse_expect(args.expect))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for alias, item in report["collections"].items():
            print(f"{alias:10} count={item['count']} scrolled={item['scrolled']} ({item['name']})")
        print(f"last_written_at={report['last_written_at']}")
        for issue in report["issues"]:
            print(f"ISSUE: {issue}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        print("OK" if report["ok"] else f"FAILED ({len(report['issues'])} issue(s))")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
