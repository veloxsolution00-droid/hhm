"""Deterministic scoring aid for the live end-to-end runs.

Produces a compact per-case dossier (answer, cited sources, stage timings) and a
mechanical metrics table. Correctness/completeness are adjudicated by reading
the dossiers; this script only computes the parts that can be computed exactly
(citation validity, citation page accuracy, abstention, latency).

Usage: python evaluation/audit/score_live.py
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
QUESTION_BANK = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["questions"]
BY_ID = {q["id"]: q for q in QUESTION_BANK}

CONFIGS = [
    ("cfgA_3b_programmatic", "live_cfgA_3b_programmatic.json"),
    ("cfgB_3b_3b", "live_cfgB_3b_3b.json"),
    ("cfgC_3b_7b", "live_cfgC_3b_7b.json"),
]

DOC_KEY = {
    "dracula": "dracula",
    "franklin": "franklin",
    "enchiridion": "enchiridion",
    "science": "science",
    "hill": "hill",
}


def doc_of(filename: str) -> str:
    name = (filename or "").lower()
    for key, value in DOC_KEY.items():
        if key in name:
            return value
    return name


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main():
    for label, filename in CONFIGS:
        path = HERE / filename
        if not path.exists():
            print(f"== {label}: missing {filename} ==")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["results"]
        print("=" * 110)
        print(f"CONFIG {label}  ({len(rows)} cases)")
        print("=" * 110)
        for row in rows:
            item = BY_ID.get(row["id"], {})
            expected_pages = set(item.get("pages") or [])
            expected_docs = set(item.get("docs") or [item.get("doc")])
            sources = row.get("sources") or []
            refs = row.get("citation_refs") or []
            valid_refs = [r for r in refs if 1 <= r <= len(sources)]
            invalid_refs = [r for r in refs if not (1 <= r <= len(sources))]
            cited = [(r, sources[r - 1]) for r in valid_refs]
            cited_pages = {str(s.get("page")) for _r, s in cited}
            cited_docs = {doc_of(s.get("filename")) for _r, s in cited}
            page_hits = len(cited_pages & expected_pages)
            answer = row.get("answer") or ""
            retrieval = row.get("retrieval") or {}
            print(f'\n--- {row["id"]} [{row["category"]}] {row["question"]}')
            if item.get("history"):
                print(f'    history={len(item["history"])} turns')
            print(f'    status={row.get("status")} abstained={row.get("abstained")} '
                  f'total={row.get("elapsed_seconds")}s')
            print(f'    stages: retrieval={retrieval.get("retrieval_seconds")} '
                  f'generation={retrieval.get("generation_seconds")} '
                  f'verification={retrieval.get("verification_seconds")} '
                  f'ranking={retrieval.get("ranking_seconds")} '
                  f'selection={retrieval.get("selection_seconds")}')
            print(f'    flags: passes={retrieval.get("retrieval_passes")} '
                  f'variants={retrieval.get("query_variants_searched")} '
                  f'cache={retrieval.get("cache_hit")} '
                  f'recovery={retrieval.get("answer_recovery_attempted")} '
                  f'verify_retry={retrieval.get("verification_retry_attempted")} '
                  f'extractive={retrieval.get("extractive_recovery_attempted")} '
                  f'scope={retrieval.get("document_scope")}')
            print(f'    sources({len(sources)}): ' +
                  ", ".join(f'{doc_of(s.get("filename"))}:{s.get("page")}' for s in sources))
            print(f'    cited refs={refs} invalid={invalid_refs} '
                  f'citation_pages={sorted(cited_pages)} expected_pages={sorted(expected_pages)} '
                  f'page_hits={page_hits}/{len(expected_pages) if expected_pages else "-"}')
            print(f'    expected_docs={sorted(expected_docs)} cited_docs={sorted(cited_docs)}')
            print(f'    citation_validation={json.dumps(row.get("citation_validation"))[:300]}')
            print(f'    confidence={row.get("confidence")} coverage={row.get("coverage")}')
            print(f'    expected: {item.get("expected_answer")}')
            print(f'    ANSWER: {" ".join(answer.split())}')
        totals = [r.get("elapsed_seconds") for r in rows if r.get("elapsed_seconds")]
        gens = [r["retrieval"].get("generation_seconds") for r in rows
                if (r.get("retrieval") or {}).get("generation_seconds")]
        vers = [r["retrieval"].get("verification_seconds") for r in rows
                if (r.get("retrieval") or {}).get("verification_seconds")]
        print(f"\n{label} LATENCY: n={len(totals)} mean={statistics.mean(totals):.1f}s "
              f"p50={statistics.median(totals):.1f}s p90={percentile(totals,0.9):.1f}s "
              f"max={max(totals):.1f}s min={min(totals):.1f}s")
        if gens:
            print(f"  generation: mean={statistics.mean(gens):.1f}s max={max(gens):.1f}s")
        if vers:
            print(f"  verification: n={len(vers)} mean={statistics.mean(vers):.1f}s max={max(vers):.1f}s")
        statuses = {}
        for row in rows:
            statuses[row.get("status")] = statuses.get(row.get("status"), 0) + 1
        print(f"  statuses: {statuses}")
        mem = [r["memory_after"].get("process_rss_mb") for r in rows
               if (r.get("memory_after") or {}).get("process_rss_mb")]
        if mem:
            print(f"  rss MB: min={min(mem):.0f} max={max(mem):.0f}")
        models = [r["memory_after"].get("ollama_models") for r in rows
                  if r.get("memory_after")]
        if models:
            print(f"  models loaded at end: {models[-1]}")


if __name__ == "__main__":
    main()
