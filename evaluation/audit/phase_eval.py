"""Phase runner for the frozen retrieval harness.

Reuses `retrieval_eval.run_case` unchanged (same production components, same
metric formulas) but writes to a NEW output path so the frozen
`retrieval_results.json` is never overwritten.

Usage:
    python evaluation/audit/phase_eval.py <label>

Writes evaluation/audit/retrieval_results_<label>.json and prints the summary.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import retrieval_eval as R  # noqa: E402
from app.core.config import settings  # noqa: E402


def summarise(results: list[dict]) -> dict:
    answerable = [r for r in results if "error" not in r and r["metrics"]["expected_pages"]]
    unanswerable = [r for r in results if "error" not in r and not r["metrics"]["expected_pages"]]
    out: dict = {"cases": len(results), "answerable": len(answerable),
                 "unanswerable": len(unanswerable)}
    for key in ("recall@1", "recall@3", "recall@4", "recall@8", "recall@20", "recall_selected"):
        out[key] = round(statistics.mean(r["metrics"][key] for r in answerable), 4) if answerable else None
    out["mrr"] = round(statistics.mean(r["metrics"]["mrr"] for r in answerable), 4) if answerable else None
    out["precision_selected"] = round(
        statistics.mean(r["metrics"]["precision_selected"] for r in answerable), 4) if answerable else None
    cov = [r["metrics"]["page_coverage_selected"] for r in answerable
           if r["metrics"]["page_coverage_selected"] is not None]
    out["page_coverage_selected"] = round(statistics.mean(cov), 4) if cov else None
    tc = [r["metrics"]["term_coverage_selected"] for r in answerable
          if r["metrics"]["term_coverage_selected"] is not None]
    out["term_coverage_selected"] = round(statistics.mean(tc), 4) if tc else None
    secs = sorted(r["retrieval_seconds"] for r in answerable)
    if secs:
        out["retrieval_seconds"] = {"mean": round(statistics.mean(secs), 2),
                                    "p50": round(statistics.median(secs), 2),
                                    "max": round(secs[-1], 2)}
    out["empty_after_gate"] = sum(1 for r in answerable if r["metrics"]["no_evidence_after_gate"])
    by_cat = defaultdict(list)
    for r in answerable:
        by_cat[r["category"]].append(r["metrics"]["recall_selected"])
    out["recall_selected_by_category"] = {c: round(statistics.mean(v), 4)
                                          for c, v in sorted(by_cat.items())}
    return out


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    payload = json.loads(R.QUESTIONS.read_text(encoding="utf-8"))
    results = []
    for item in payload["questions"]:
        try:
            result = R.run_case(item)
        except Exception as exc:  # noqa: BLE001
            result = {"id": item["id"], "category": item["category"],
                      "question": item["question"], "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        metrics = result.get("metrics") or {}
        print(f'{result["id"]:<4} {result["category"]:<26} '
              f'rec@sel={metrics.get("recall_selected")} mrr={metrics.get("mrr")} '
              f'prec={metrics.get("precision_selected")} '
              f'cov={metrics.get("page_coverage_selected")} '
              f'sim={result.get("best_similarity_before_gate")} '
              f't={result.get("retrieval_seconds")}s', flush=True)

    summary = summarise(results)
    out = HERE / f"retrieval_results_{label}.json"
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "config": {"top_k": settings.top_k, "min_similarity": settings.min_similarity,
                   "lexical_weight": settings.hybrid_lexical_weight,
                   "embedding_model": settings.embedding_model},
        "summary": summary,
        "results": results,
    }, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
