#!/usr/bin/env python3
"""Phase C decision-point benchmark (see UPGRADE_PLAN.md): does
intfloat/multilingual-e5-large recall real Vietnamese memories/documents
better than the current paraphrase-multilingual-MiniLM-L12-v2? A one-off
analysis tool, not part of the running service.

Method: pull real Vietnamese user chat questions as queries and real
fact/document texts as the candidate corpus, embed the corpus with both
models, rank by cosine similarity per query per model, then ask the
configured LLM to judge each model's top-3 for relevance — blind, it
doesn't know which model produced which list. Report a win/tie/loss tally.

Run inside the container (needs fastembed's e5-large + the real Qdrant):
  docker compose run --rm -v "$PWD:/repo" -w /repo/llamaindex-service llamaindex \
    sh -c "pip install -q fastembed==0.8.0 && python3 /repo/scripts/benchmark_embeddings.py"
"""

import random
import sys

import numpy as np
from qdrant_client import QdrantClient

sys.path.insert(0, "/repo/llamaindex-service")
from app import config  # noqa: E402

CURRENT_MODEL = config.EMBED_MODEL
CANDIDATE_MODEL = "intfloat/multilingual-e5-large"
N_QUERIES = 15
TOP_K = 3
random.seed(42)


def fetch_texts(client, collection, field, limit=500, **filters):
    must = [{"key": k, "match": {"value": v}} for k, v in filters.items()]
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter={"must": must} if must else None,
        limit=limit,
        with_payload=True,
    )
    return [p.payload[field].strip() for p in points if (p.payload or {}).get(field, "").strip()]


def cosine_topk(query_vec, corpus_vecs, k):
    q = np.array(query_vec)
    c = np.array(corpus_vecs)
    sims = c @ q / (np.linalg.norm(c, axis=1) * np.linalg.norm(q) + 1e-9)
    idx = np.argsort(-sims)[:k]
    return idx


def main():
    client = QdrantClient(url=config.QDRANT_URL)
    queries = fetch_texts(client, config.CHAT_HISTORY_COLLECTION, "content", role="user")
    facts = fetch_texts(client, config.MEMORIES_COLLECTION, "text")

    corpus = list(dict.fromkeys(facts))  # facts are short + self-contained, best corpus for this test
    queries = list(dict.fromkeys(queries))
    if len(queries) > N_QUERIES:
        queries = random.sample(queries, N_QUERIES)
    print(f"corpus: {len(corpus)} facts | queries: {len(queries)}")
    if len(corpus) < 5 or len(queries) < 3:
        print("Not enough real data for a meaningful benchmark. Aborting.")
        return

    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    print(f"loading current model: {CURRENT_MODEL}")
    current = FastEmbedEmbedding(model_name=CURRENT_MODEL)
    print(f"loading candidate model: {CANDIDATE_MODEL}")
    candidate = FastEmbedEmbedding(model_name=CANDIDATE_MODEL)

    print("embedding corpus with both models...")
    corpus_current = [current.get_text_embedding(t) for t in corpus]
    corpus_candidate = [candidate.get_text_embedding(t) for t in corpus]

    llm = None
    if config.LLM_PROVIDER != "none":
        from app import providers
        llm = providers.build_llm()

    results = []
    for q in queries:
        qv_current = current.get_text_embedding(q)
        qv_candidate = candidate.get_text_embedding(q)
        top_current = [corpus[i] for i in cosine_topk(qv_current, corpus_current, TOP_K)]
        top_candidate = [corpus[i] for i in cosine_topk(qv_candidate, corpus_candidate, TOP_K)]
        results.append({"query": q, "current": top_current, "candidate": top_candidate})

    print(f"\n{'='*70}\nRESULTS ({len(results)} queries, top-{TOP_K})\n{'='*70}")
    for r in results:
        print(f"\nQ: {r['query']}")
        print(f"  [current/{CURRENT_MODEL.split('/')[-1]}]")
        for t in r["current"]:
            print(f"    - {t[:90]}")
        print(f"  [candidate/{CANDIDATE_MODEL.split('/')[-1]}]")
        for t in r["candidate"]:
            print(f"    - {t[:90]}")

    if not llm:
        print("\n(LLM_PROVIDER=none — skipping automated relevance judging; "
              "review the printed top-k lists manually above.)")
        return

    print(f"\n{'='*70}\nLLM-JUDGED RELEVANCE (blind — sides randomized per query)\n{'='*70}")
    wins = {"current": 0, "candidate": 0, "tie": 0}
    for r in results:
        sides = [("current", r["current"]), ("candidate", r["candidate"])]
        random.shuffle(sides)
        (label_a, list_a), (label_b, list_b) = sides
        prompt = (
            "Câu hỏi của người dùng: \"" + r["query"] + "\"\n\n"
            "Danh sách A:\n" + "\n".join(f"- {t}" for t in list_a) + "\n\n"
            "Danh sách B:\n" + "\n".join(f"- {t}" for t in list_b) + "\n\n"
            "Danh sách nào liên quan hơn tới câu hỏi? Trả lời DUY NHẤT một từ: "
            "A, B, hoặc TIE nếu ngang nhau."
        )
        verdict = llm.complete(prompt).text.strip().upper()
        winner = {"A": label_a, "B": label_b}.get(verdict[:1], "tie")
        wins[winner if winner in wins else "tie"] += 1
        print(f"Q: {r['query'][:60]!r} -> {winner}")

    print(f"\nTally: current={wins['current']} candidate={wins['candidate']} tie={wins['tie']}")
    if wins["candidate"] > wins["current"] * 1.3:
        print("RECOMMENDATION: candidate model shows a clear improvement — worth migrating (C2/C3).")
    elif wins["current"] > wins["candidate"] * 1.3:
        print("RECOMMENDATION: current model is at least as good — do NOT migrate.")
    else:
        print("RECOMMENDATION: no clear winner — migration cost not justified by this sample.")


if __name__ == "__main__":
    main()
