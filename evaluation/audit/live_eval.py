"""Live end-to-end evaluation against the running API.

Records per question: status, answer, sources (file+page), citation refs,
stage timings from the response diagnostics, and process/model memory.

Usage:
  python evaluation/audit/live_eval.py --label cfgA --ids A01,H05,J01 --out cfgA.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

QUESTIONS = Path(__file__).with_name("questions.json")
API = "http://127.0.0.1:8000"


def post(url, payload, timeout=1800):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url, timeout=30):
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def memory_snapshot():
    snapshot = {"ollama_models": None, "process_rss_mb": None}
    try:
        ps = get("http://127.0.0.1:11434/api/ps")
        snapshot["ollama_models"] = [
            {"name": m.get("name"), "size_mb": round(m.get("size", 0) / 1048576, 1),
             "size_vram_mb": round(m.get("size_vram", 0) / 1048576, 1)}
            for m in ps.get("models", [])
        ]
    except Exception as exc:  # noqa: BLE001
        snapshot["ollama_models"] = f"error: {exc}"
    try:
        import psutil

        total = 0
        for name in ("ollama.exe", "ollama app.exe", "python.exe"):
            for proc in psutil.process_iter(["name", "memory_info"]):
                if (proc.info.get("name") or "").lower() == name:
                    total += proc.info["memory_info"].rss
        snapshot["process_rss_mb"] = round(total / 1048576, 1)
    except Exception:
        pass
    return snapshot


def refs_in(text):
    refs = set()
    for group in re.findall(r"\[([\d,\s]+)\]", text or ""):
        for value in group.split(","):
            value = value.strip()
            if value.isdigit():
                refs.add(int(value))
    return sorted(refs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--api", default=API)
    args = parser.parse_args()

    payload = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in payload["questions"]}
    ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    results = []
    for case_id in ids:
        item = by_id[case_id]
        before = memory_snapshot()
        started = time.perf_counter()
        try:
            response = post(args.api + "/chat", {
                "question": item["question"],
                "history": item.get("history") or [],
            })
            elapsed = time.perf_counter() - started
            error = None
        except Exception as exc:  # noqa: BLE001
            response = {}
            elapsed = time.perf_counter() - started
            error = f"{type(exc).__name__}: {exc}"
        after = memory_snapshot()
        retrieval = response.get("retrieval") or {}
        sources = response.get("sources") or []
        answer = response.get("answer") or ""
        expected_pages = set(item.get("pages") or [])
        source_pages = {str(s.get("source_location")) for s in sources}
        record = {
            "id": case_id,
            "category": item["category"],
            "question": item["question"],
            "expected_pages": sorted(expected_pages),
            "expected_answer": item.get("expected_answer"),
            "status": response.get("status"),
            "abstained": response.get("abstained"),
            "answer": answer,
            "sources": [{"filename": s.get("filename"), "page": s.get("source_location"),
                         "rank": s.get("rank")} for s in sources],
            "citation_refs": refs_in(answer),
            "citation_validation": response.get("citation_validation"),
            "question_type": response.get("question_type"),
            "question_resolved": response.get("question_resolved"),
            "confidence": response.get("confidence"),
            "coverage": response.get("coverage"),
            "retrieval": {
                key: retrieval.get(key)
                for key in ("retrieval_seconds", "generation_seconds", "verification_seconds",
                            "ranking_seconds", "selection_seconds", "candidate_chunks",
                            "selected_chunks", "best_similarity", "top_k", "min_similarity",
                            "retrieval_passes", "query_variants_searched", "spelling_correction",
                            "cache_hit", "cache_type", "document_scope", "failure_stage",
                            "verification_retry_attempted", "answer_recovery_attempted",
                            "extractive_recovery_attempted", "selected_context_chars")
                if key in retrieval
            },
            "page_match": bool(expected_pages & source_pages) if expected_pages else None,
            "teacher_forcing_note": item.get("inference"),
            "elapsed_seconds": round(elapsed, 3),
            "response_seconds": response.get("response_seconds"),
            "error": error,
            "memory_before": before,
            "memory_after": after,
        }
        results.append(record)
        print(f'{case_id:<4} {record["status"]:<28} '
              f'total={record["elapsed_seconds"]:8.1f}s '
              f'retr={record["retrieval"].get("retrieval_seconds")} '
              f'gen={record["retrieval"].get("generation_seconds")} '
              f'ver={record["retrieval"].get("verification_seconds")} '
              f'page_match={record["page_match"]}', flush=True)

    Path(args.out).write_text(json.dumps({"label": args.label, "results": results}, indent=2), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
