"""Evaluate local retrieval against a user-labelled JSONL question set.

Each line: {"question": "...", "relevant_chunk_ids": ["doc-id:chunk-index"]}
Run from project root: python scripts/evaluate_retrieval.py path/to/eval.jsonl
No network services are used; Ollama/ChromaDB must already be running locally.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from app.retrieval.store import search
from app.retrieval.pipeline import rank_candidates


def evaluate(rows, owner_id="local", k=10):
    recalls, precisions, reciprocal_ranks = [], [], []
    details = []
    for row in rows:
        expected = set(row.get("relevant_chunk_ids") or [])
        if not expected:
            continue
        candidates = search(row["question"], owner_id=owner_id)
        ranked = rank_candidates(row["question"], candidates)
        ids = [item.get("id") for item in ranked[:k]]
        found = expected.intersection(ids)
        recalls.append(len(found) / len(expected))
        precisions.append(len(found) / max(1, k))
        first = next((i for i, chunk_id in enumerate(ids, 1) if chunk_id in expected), None)
        reciprocal_ranks.append(1 / first if first else 0.0)
        details.append({
            "question": row["question"], "expected": sorted(expected),
            "retrieved": ids, "relevant_found": sorted(found),
            "recall_at_k": recalls[-1], "precision_at_k": precisions[-1],
            "reciprocal_rank": reciprocal_ranks[-1],
        })
    n = len(details)
    return {
        "evaluated_questions": n,
        "k": k,
        "recall_at_k": sum(recalls) / n if n else None,
        "precision_at_k": sum(precisions) / n if n else None,
        "mrr": sum(reciprocal_ranks) / n if n else None,
        "details": details,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/evaluate_retrieval.py eval.jsonl [owner_id] [k]")
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    owner = sys.argv[2] if len(sys.argv) > 2 else "local"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print(json.dumps(evaluate(rows, owner_id=owner, k=k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
