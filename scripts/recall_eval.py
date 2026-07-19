#!/usr/bin/env python3
"""Recall quality evaluation: measure what the auto-recall pipeline actually
returns against a fixed bilingual eval set, so threshold/router changes are
judged by numbers ("saved N chars, lost M expected hits") instead of feel.
Born out of a bug that survived for days precisely because nothing measured
retrieval: chunks embedded with metadata ranked near-randomly and no one saw.

Seeds a throwaway in-memory Qdrant with the corpus from
llamaindex-service/evals/recall_eval.json (facts + history + documents),
runs every case through memories.recall() with the REAL embedding model,
and scores the produced context_block:

  - include hit  : an expect_include substring appears in the block
  - violation    : an expect_exclude substring appears in the block
  - chars        : block size (proxy for injected tokens)

The committed baseline (evals/recall_baseline.json) records reality — some
cases fail on purpose until the feature they measure lands.

Exit codes: 0 = evaluated successfully and not worse; 1 = regression;
2 = incomplete/missing baseline configuration. --update-baseline rewrites
the relevant baseline after an accepted change.

Modes:
  fixture  fixed synthetic corpus; catches code/embedding/ranking regressions
  live     no seeding; catches drift of explicitly configured live canaries
  scale    fixture corpus plus deterministic distractors in an isolated store

Run inside the service container (needs app deps + the embedding model):

  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \
      llamaindex python /repo/scripts/recall_eval.py --mode fixture
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llamaindex-service"))

from llama_index.core import Settings, StorageContext, VectorStoreIndex  # noqa: E402
from llama_index.vector_stores.qdrant import QdrantVectorStore  # noqa: E402
from qdrant_client import QdrantClient, models as qmodels  # noqa: E402

from app import config, documents, hybrid, memories, memory_store, qdrant_setup  # noqa: E402
from app.providers import build_embed_model  # noqa: E402

EVAL_FILE = REPO / "llamaindex-service" / "evals" / "recall_eval.json"
BASELINE_FILE = REPO / "llamaindex-service" / "evals" / "recall_baseline.json"
LIVE_EVAL_FILE = REPO / "llamaindex-service" / "evals" / "recall_live_eval.local.json"
LIVE_BASELINE_FILE = REPO / "llamaindex-service" / "evals" / "recall_live_baseline.local.json"
SCALE_BASELINE_FILE = REPO / "llamaindex-service" / "evals" / "recall_scale_baseline.json"


def seed(client: QdrantClient, embed, index, corpus: dict) -> None:
    for f in corpus.get("facts", []):
        memories.save_facts(
            client, embed,
            [{"text": f["text"], "type": f.get("type", "fact"),
              "importance": f.get("importance", 0.5),
              "subject": f.get("subject", ""), "relation": f.get("relation", ""),
              "object": f.get("object", "")}],
            project_id=f.get("project", ""), source_agent=f.get("source_agent", ""),
        )
        if f.get("age_days"):
            # Backdate created_at AND last_seen so cases can stage the top-k
            # race between an old standing preference and fresh same-topic
            # facts — decay reads last_seen (see memories.search_memories),
            # so leaving it at seed time would make a simulated "old" fact
            # decay as if it had just been recalled. The point id is
            # deterministic, so the seeded point is addressable.
            backdated = time.time() - f["age_days"] * 86400
            client.set_payload(
                collection_name=config.MEMORIES_COLLECTION,
                payload={"created_at": backdated, "last_seen": backdated},
                points=[memories.fact_point_id(
                    config.USER_ID, f["text"],
                    f.get("project") or config.DEFAULT_PROJECT,
                )],
            )
    for turn in corpus.get("history", []):
        memory_store.add_message(
            client, embed, turn["session"], turn["role"], turn["content"],
            project_id=turn.get("project", config.DEFAULT_PROJECT),
        )
    for doc in corpus.get("documents", []):
        documents.ingest_text(
            index, client, doc["text"], {"source": doc.get("source", "")},
            project_id=doc.get("project", config.DEFAULT_PROJECT),
        )


def expected_memory_ids(spec: dict) -> dict[str, dict[str, str]]:
    """Resolve fixture expectations to deterministic point IDs before noise.

    Only expectations that uniquely identify a seeded fact participate in
    memory rank metrics. History/document expectations remain context hits.
    """
    facts = spec["corpus"].get("facts", [])
    resolved = {}
    for case in spec["cases"]:
        case_project = case.get("project", "")
        for expected in case.get("expect_include", []):
            matches = [fact for fact in facts if expected in fact["text"] and (
                not case_project
                or (fact.get("project") or config.DEFAULT_PROJECT)
                in (case_project, config.DEFAULT_PROJECT)
            )]
            if len(matches) == 1:
                fact = matches[0]
                project = fact.get("project") or config.DEFAULT_PROJECT
                resolved.setdefault(case["name"], {})[expected] = memories.fact_point_id(
                    config.USER_ID, fact["text"], project
                )
    return resolved


def configured_memory_ids(spec: dict) -> dict[str, dict[str, str]]:
    """Read explicit IDs from a private live spec (never infer live identity)."""
    configured = {}
    for case in spec["cases"]:
        ids = case.get("expect_memory_ids") or {}
        unknown = set(ids) - set(case.get("expect_include", []))
        invalid = [key for key, value in ids.items()
                   if not isinstance(value, str) or not value.strip()]
        if unknown or invalid:
            details = []
            if unknown:
                details.append("keys absent from expect_include: " + ", ".join(sorted(unknown)))
            if invalid:
                details.append("empty/non-string IDs for: " + ", ".join(sorted(invalid)))
            raise SystemExit(f"invalid expect_memory_ids in {case['name']}: " + "; ".join(details))
        if ids:
            configured[case["name"]] = ids
    return configured


def run_cases(client: QdrantClient, embed, cases: list,
              expected_ids: dict[str, dict[str, str]] | None = None) -> list:
    results = []
    for case in cases:
        recall = memories.recall(
            client, embed, case["query"],
            project=case.get("project", ""), recent_turns=0,
        )
        block = recall.get("context_block") or ""
        memory_ranks = {}
        for expected, expected_id in (expected_ids or {}).get(case["name"], {}).items():
            memory_ranks[expected] = next(
                (rank for rank, item in enumerate(recall.get("memories", []), 1)
                 if item.get("id") == expected_id), None
            )
        results.append({
            "name": case["name"],
            "hits": [s for s in case.get("expect_include", []) if s in block],
            "misses": [s for s in case.get("expect_include", []) if s not in block],
            "violations": [s for s in case.get("expect_exclude", []) if s in block],
            "chars": len(block),
            "memory_ranks": memory_ranks,
            "note": case.get("note", ""),
        })
    return results


def summarize(results: list) -> dict:
    expected = sum(len(r["hits"]) + len(r["misses"]) for r in results)
    rank_values = [rank for r in results for rank in r.get("memory_ranks", {}).values()]
    ranks = [rank for rank in rank_values
             if rank is not None]
    return {
        "include_hits": sum(len(r["hits"]) for r in results),
        "include_expected": expected,
        "violations": sum(len(r["violations"]) for r in results),
        "total_chars": sum(r["chars"] for r in results),
        "memory_hit_at_1": sum(rank <= 1 for rank in ranks),
        "memory_hit_at_3": sum(rank <= 3 for rank in ranks),
        "ranked_memory_expected": len(rank_values),
    }


def seed_distractors(client: QdrantClient, embed, count: int,
                     batch_size: int = 128) -> None:
    """Add deterministic easy, lexical and semantic hard negatives.

    This bypasses save-time dedup/supersession intentionally: scale eval is
    testing retrieval capacity, not write behaviour. It only ever receives a
    throwaway in-memory client created by this script.
    """
    templates = (
        "Synthetic unrelated memory {i}: garden sensor calibration uses channel {n}",
        "Synthetic shopmed lexical distractor {i}: PostgreSQL import job documentation draft {n}",
        "Synthetic hard negative {i}: bookinghub overlapping slots use an inclusive boundary in archive {n}",
    )
    now = time.time()
    for start in range(0, count, batch_size):
        rows = [(i, templates[i % len(templates)].format(i=i, n=i % 97))
                for i in range(start, min(start + batch_size, count))]
        texts = [text_value for _, text_value in rows]
        if hasattr(embed, "get_text_embedding_batch"):
            vectors = embed.get_text_embedding_batch(texts)
        else:
            vectors = [embed.get_text_embedding(text) for text in texts]
        points = []
        for (i, text_value), vector in zip(rows, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"longbrain-scale-distractor:{i}"))
            project = "shopmed" if i % 3 else "bookinghub"
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=hybrid.point_vector(
                    client, config.MEMORIES_COLLECTION, vector, text_value
                ),
                payload={
                    "user_id": config.USER_ID,
                    "project_id": project,
                    "type": "fact",
                    "text": text_value,
                    "importance": 0.5,
                    "created_at": now,
                    "last_seen": now,
                    "source_agent": "recall-scale-eval",
                },
            ))
        client.upsert(collection_name=config.MEMORIES_COLLECTION, points=points)


def make_fixture(embed, spec: dict) -> QdrantClient:
    dim = len(embed.get_text_embedding("dimension probe"))
    client = QdrantClient(":memory:")
    qdrant_setup.ensure_all(client, dim)
    vector_store = QdrantVectorStore(
        client=client, collection_name=config.DOCUMENTS_COLLECTION
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
    )
    seed(client, embed, index, spec["corpus"])
    return client


def compare_summary(summary: dict, base: dict) -> list[str]:
    regressed = []
    if summary["include_hits"] < base["include_hits"]:
        regressed.append(f"hits {base['include_hits']} -> {summary['include_hits']}")
    if summary["violations"] > base["violations"]:
        regressed.append(f"violations {base['violations']} -> {summary['violations']}")
    if summary["total_chars"] > base["total_chars"] * 1.2:
        regressed.append(f"chars {base['total_chars']} -> {summary['total_chars']} (+20%)")
    for metric, label in (("memory_hit_at_1", "memory@1"),
                          ("memory_hit_at_3", "memory@3")):
        if metric in base and summary.get(metric, 0) < base[metric]:
            regressed.append(f"{label} {base[metric]} -> {summary.get(metric, 0)}")
    return regressed


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fixture", "live", "scale"), default="fixture")
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--live-file", type=Path, default=LIVE_EVAL_FILE)
    p.add_argument("--scale-tiers", default="0,100,1000",
                   help="comma-separated distractor counts (default: 0,100,1000)")
    return p


def main() -> int:
    args = parser().parse_args()
    spec_file = args.live_file if args.mode == "live" else EVAL_FILE
    if args.mode == "live" and not spec_file.exists():
        raise SystemExit(
            f"live eval file not found: {spec_file}\n"
            "copy evals/recall_live_eval.example.json to "
            "evals/recall_live_eval.local.json and add private canaries"
        )
    spec = json.loads(spec_file.read_text())

    embed = build_embed_model()
    Settings.embed_model = embed
    if args.mode == "live":
        client = QdrantClient(url=config.QDRANT_URL)
        # A verifier must not change the decay state it is measuring. Recall
        # production refreshes last_seen, but live canaries are read-only.
        refresh = config.LAST_SEEN_REFRESH
        try:
            config.LAST_SEEN_REFRESH = False
            live_results = run_cases(
                client, embed, spec["cases"], configured_memory_ids(spec)
            )
        finally:
            config.LAST_SEEN_REFRESH = refresh
        runs = [("live", live_results)]
        baseline_file = LIVE_BASELINE_FILE
    elif args.mode == "scale":
        tiers = sorted({int(x) for x in args.scale_tiers.split(",") if x.strip()})
        if not tiers or tiers[0] < 0:
            raise SystemExit("--scale-tiers requires non-negative integers")
        runs = []
        fixture_ids = expected_memory_ids(spec)
        for tier in tiers:
            # Every tier starts from an identical fixture. recall() refreshes
            # last_seen, so sharing a client would contaminate later tiers.
            client = make_fixture(embed, spec)
            seed_distractors(client, embed, tier)
            started = time.perf_counter()
            tier_results = run_cases(client, embed, spec["cases"], fixture_ids)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            runs.append((str(tier), tier_results, elapsed_ms))
        baseline_file = SCALE_BASELINE_FILE
    else:
        client = make_fixture(embed, spec)
        runs = [("fixture", run_cases(
            client, embed, spec["cases"], expected_memory_ids(spec)
        ))]
        baseline_file = BASELINE_FILE

    if args.mode == "scale":
        scale_output = {}
        for tier, tier_results, elapsed_ms in runs:
            summary = summarize(tier_results)
            summary["elapsed_ms"] = elapsed_ms
            scale_output[tier] = {"summary": summary, "cases": tier_results}
            print(f"tier={tier:>8} distractors  hits={summary['include_hits']}/"
                  f"{summary['include_expected']} violations={summary['violations']} "
                  f"memory@1={summary['memory_hit_at_1']} memory@3={summary['memory_hit_at_3']} "
                  f"elapsed={elapsed_ms}ms")
        if args.update_baseline:
            baseline_file.write_text(json.dumps({"tiers": scale_output}, ensure_ascii=False, indent=1) + "\n")
            print(f"baseline written: {baseline_file}")
            return 0
        if not baseline_file.exists():
            print("no scale baseline yet — run with --update-baseline to record one")
            return 0
        base_tiers = json.loads(baseline_file.read_text())["tiers"]
        regressed = []
        missing_baselines = []
        for tier, output in scale_output.items():
            if tier not in base_tiers:
                missing_baselines.append(tier)
                continue
            regressed += [f"tier {tier}: {item}" for item in
                          compare_summary(output["summary"], base_tiers[tier]["summary"])]
        if missing_baselines:
            print("\n✗ scale baseline incomplete; missing tier(s): "
                  + ", ".join(missing_baselines)
                  + ". Run with --update-baseline after reviewing the results.")
        if regressed:
            print("\n✗ WORSE than scale baseline: " + "; ".join(regressed)
                  + ". Review regressions before updating the baseline.")
            return 1
        if missing_baselines:
            return 2
        improved = any(
            output["summary"][metric] > base_tiers[tier]["summary"].get(metric, 0)
            for tier, output in scale_output.items()
            for metric in ("include_hits", "memory_hit_at_1", "memory_hit_at_3")
        ) or any(
            output["summary"]["violations"]
            < base_tiers[tier]["summary"]["violations"]
            for tier, output in scale_output.items()
        )
        print("\n✓ not worse than scale baseline"
              + (" (better — consider --update-baseline)" if improved else ""))
        return 0

    _, results = runs[0]

    print(f"{'case':34} {'hits':>6} {'viol':>5} {'chars':>6}")
    for r in results:
        total = len(r["hits"]) + len(r["misses"])
        flag = "" if not r["misses"] and not r["violations"] else "  <-"
        print(f"{r['name']:34} {len(r['hits'])}/{total:<4} {len(r['violations']):>5} {r['chars']:>6}{flag}")
        for m in r["misses"]:
            print(f"    miss: {m!r}" + (f"  ({r['note']})" if r["note"] else ""))
        for v in r["violations"]:
            print(f"    VIOLATION: {v!r}")

    summary = summarize(results)
    print(f"\nsummary: {summary['include_hits']}/{summary['include_expected']} expected hits, "
          f"{summary['violations']} violations, {summary['total_chars']} chars injected")

    if args.update_baseline:
        baseline_file.write_text(json.dumps({"summary": summary, "cases": results},
                                            ensure_ascii=False, indent=1) + "\n")
        print(f"baseline written: {baseline_file}")
        return 0

    if not baseline_file.exists():
        print("no baseline yet — run with --update-baseline to record one")
        return 0

    base = json.loads(baseline_file.read_text())["summary"]
    regressed = compare_summary(summary, base)
    if regressed:
        print("\n✗ WORSE than baseline: " + "; ".join(regressed))
        return 1
    print("\n✓ not worse than baseline"
          + (" (better — consider --update-baseline)"
             if summary["include_hits"] > base["include_hits"]
             or summary["violations"] < base["violations"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
