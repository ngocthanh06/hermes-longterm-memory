#!/usr/bin/env python3
"""One-time migration for C2 hybrid recall: recreate the three memory
collections with the named BM25 sparse vector, carrying every existing
dense vector over unchanged. The embedding model did NOT change (Phase C1
closed on MiniLM), so nothing is re-embedded — the only new computation is
the cheap BM25 term-frequency vector from each point's own payload text.

Why recreate: Qdrant cannot add a new sparse vector name to an existing
collection (400 "Not existing vector name" — verified on this stack), so
the collections must be born with it. New installs get the sparse schema
from qdrant_setup.ensure_all; this script upgrades data that predates it.

Safety: dumps every collection (ids + dense vectors + payloads) to
backups/hybrid_migration_<timestamp>/<collection>.json.gz BEFORE deleting
anything. To restore a dump, upsert its rows back into a collection created
by ensure_all (vector = row["vector"], payload = row["payload"]).
Idempotent: collections that already carry the sparse vector are skipped.

Run with the service STOPPED so no hook write races the swap:

  docker compose stop llamaindex
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/migrate_hybrid_bm25.py
  docker compose up -d --build llamaindex
"""

import gzip
import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, "/repo/llamaindex-service")
from app import config, hybrid, qdrant_setup  # noqa: E402

BACKUP_ROOT = Path("/repo/backups")
BATCH = 64


def _chunk_text(payload: dict) -> str:
    try:
        return json.loads(payload.get("_node_content") or "{}").get("text", "")
    except (ValueError, TypeError):
        return ""


TEXT_OF = {
    config.CHAT_HISTORY_COLLECTION: lambda p: p.get("content", ""),
    config.MEMORIES_COLLECTION: lambda p: p.get("text", ""),
    config.DOCUMENTS_COLLECTION: _chunk_text,
}


def dump_points(client: QdrantClient, collection: str) -> list[dict]:
    rows, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=True,
        )
        for p in batch:
            dense = p.vector
            if isinstance(dense, dict):  # named form — keep only the dense one
                dense = dense.get("")
            rows.append({"id": str(p.id), "vector": dense, "payload": p.payload or {}})
        if offset is None:
            break
    return rows


def main() -> int:
    if not config.HYBRID_BM25:
        print("HYBRID_BM25=false — nothing to migrate.")
        return 0
    client = QdrantClient(url=config.QDRANT_URL)
    meta = qdrant_setup.get_meta(client)
    if not meta or not meta.get("embed_dim"):
        print("No meta / embed_dim found — is this a fresh install? Aborting.")
        return 1
    embed_dim = int(meta["embed_dim"])

    existing = {c.name for c in client.get_collections().collections}
    todo = []
    for coll in TEXT_OF:
        if coll not in existing:
            print(f"{coll}: does not exist yet — will be created with sparse schema")
        elif hybrid.collection_enabled(client, coll):
            print(f"{coll}: already has the sparse vector — skipping")
        else:
            todo.append(coll)
    if not todo:
        qdrant_setup.ensure_all(client, embed_dim)
        print("Nothing to migrate.")
        return 0

    backup_dir = BACKUP_ROOT / f"hybrid_migration_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    dumps: dict[str, list[dict]] = {}
    for coll in todo:
        rows = dump_points(client, coll)
        with gzip.open(backup_dir / f"{coll}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        dumps[coll] = rows
        print(f"{coll}: dumped {len(rows)} points -> {backup_dir / (coll + '.json.gz')}")

    for coll in todo:
        client.delete_collection(coll)
    qdrant_setup.ensure_all(client, embed_dim)  # recreates them sparse-ready

    for coll, rows in dumps.items():
        text_of, with_sparse = TEXT_OF[coll], 0
        for start in range(0, len(rows), BATCH):
            points = []
            for row in rows[start:start + BATCH]:
                sparse = hybrid.text_vector(text_of(row["payload"]))
                if sparse is not None:
                    vector = {"": row["vector"], config.BM25_VECTOR_NAME: sparse}
                    with_sparse += 1
                else:
                    vector = row["vector"]
                points.append(qmodels.PointStruct(
                    id=row["id"], vector=vector, payload=row["payload"]
                ))
            if points:
                client.upsert(collection_name=coll, points=points)
        restored = client.count(collection_name=coll, exact=True).count
        status = "OK" if restored == len(rows) else "MISMATCH"
        print(f"{coll}: restored {restored}/{len(rows)} points "
              f"({with_sparse} with sparse) [{status}]")
        if restored != len(rows):
            print(f"  !! count mismatch — restore from {backup_dir}")
            return 1

    print("\nMigration complete. Restart the service (docker compose up -d --build "
          "llamaindex) so it picks up the hybrid-aware code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
