#!/usr/bin/env python3
"""Find (and optionally repair) sessions marked fully consolidated despite
zero facts/summary ever being saved for them.

Background: consolidation._parse_extraction used to accept any parseable
JSON object even when it used the wrong top-level key (e.g. {"memories":
[...]} instead of {"facts": [...]}), silently treating it as a valid EMPTY
extraction. consolidate_session then marked the session's turns consolidated
regardless, permanently losing whatever it actually contained (the parser now
rejects a dict without a "facts" key instead). This script finds sessions
matching that exact signature so they can be given a second, correct pass.

A session showing up here is not proof of the bug: a short "hi"/chit-chat
session legitimately produces zero facts/summary too, and a session whose
facts were saved through a direct save_memories call (a synthetic session_id,
not this one) also has no linked memories. Review the printed preview before
applying.

Review-then-apply: the default (plan) run is read-only — it lists candidates
and saves them to backups/repair_stalled_consolidation_plan.json. --apply
resets consolidated=False on exactly the turns of those sessions so the next
automatic sweep (idle timeout, or the CONSOLIDATION_FORCE_TURNS backlog
trigger, or a manual POST /memory/consolidate-pending) re-consolidates them
through the current code. This script never calls an LLM itself — edit the
plan file to drop any false positives before applying.

  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo \\
      llamaindex python /repo/scripts/repair_stalled_consolidation.py
  # review the printed candidates (and/or edit the plan file), then:
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo \\
      llamaindex python /repo/scripts/repair_stalled_consolidation.py --apply
"""

import argparse
import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, "/repo/llamaindex-service")
from app import config, memory_store  # noqa: E402

PLAN_FILE = Path("/repo/backups/repair_stalled_consolidation_plan.json")
PREVIEW_CHARS = 150


def find_candidates(client: QdrantClient) -> list[dict]:
    sessions: dict[str, dict] = {}
    chat_points = memory_store._scroll_all(
        client, config.CHAT_HISTORY_COLLECTION,
        memory_store._user_filter(config.USER_ID),
        with_payload=["session_id", "consolidated", "project_id", "content", "timestamp"],
        page_size=1000,
    )
    for p in chat_points:
        sid = p.payload.get("session_id")
        e = sessions.setdefault(sid, {
            "total": 0, "done": 0, "project_id": p.payload.get("project_id"),
            "first_ts": None, "preview": "",
        })
        e["total"] += 1
        if p.payload.get("consolidated"):
            e["done"] += 1
        ts = p.payload.get("timestamp") or 0
        if e["first_ts"] is None or ts < e["first_ts"]:
            e["first_ts"] = ts
            e["preview"] = (p.payload.get("content") or "")[:PREVIEW_CHARS]

    memory_points = memory_store._scroll_all(
        client, config.MEMORIES_COLLECTION,
        memory_store._user_filter(config.USER_ID),
        with_payload=["session_id"],
        page_size=1000,
    )
    linked = {p.payload.get("session_id") for p in memory_points}

    candidates = [
        {"session_id": sid, "project_id": e["project_id"],
         "turns": e["total"], "preview": e["preview"]}
        for sid, e in sessions.items()
        if e["total"] > 0 and e["total"] == e["done"] and sid not in linked
    ]
    candidates.sort(key=lambda c: -c["turns"])
    return candidates


def apply_plan(client: QdrantClient, plan: dict) -> None:
    reset = 0
    for c in plan.get("candidates", []):
        sid = c["session_id"]
        points = memory_store._scroll_all(
            client, config.CHAT_HISTORY_COLLECTION,
            memory_store._user_filter(config.USER_ID, [
                qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=sid)),
            ]),
            with_payload=False, page_size=config.CHAT_HISTORY_MAX_MESSAGES,
        )
        ids = [str(p.id) for p in points]
        if not ids:
            continue
        client.set_payload(
            collection_name=config.CHAT_HISTORY_COLLECTION,
            payload={"consolidated": False},
            points=ids,
        )
        print(f"  reset {len(ids)} turn(s): {sid} ({c['project_id']})")
        reset += 1
    print(
        f"\n{reset} session(s) reset. Trigger the next sweep now with:\n"
        "  curl -X POST http://localhost:8800/memory/consolidate-pending"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help=f"reset consolidated=False for the sessions in {PLAN_FILE} "
             "(default: scan + save the plan, write nothing)",
    )
    args = parser.parse_args()

    client = QdrantClient(url=config.QDRANT_URL)

    if args.apply:
        if not PLAN_FILE.exists():
            print(f"No plan file at {PLAN_FILE} — run without --apply first, "
                  "review its output, then re-run with --apply.")
            return 1
        apply_plan(client, json.loads(PLAN_FILE.read_text()))
        return 0

    candidates = find_candidates(client)
    print(
        f"{len(candidates)} candidate session(s) — fully consolidated, zero linked "
        "facts/summary.\nNot all of these are bug victims: a short chit-chat session, "
        "or one whose facts were\nsaved under a different (direct:...) session_id, "
        "also matches. Review before applying.\n"
    )
    for c in candidates:
        print(f"  [{c['turns']:>3} turns] {c['project_id']:<20} {c['session_id']}  {c['preview']!r}")
    PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLAN_FILE.write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidates": candidates,
    }, ensure_ascii=False, indent=1))
    print(f"\nPlan saved to {PLAN_FILE}. Edit it to drop any false positives, "
          "then re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
