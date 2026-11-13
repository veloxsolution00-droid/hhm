"""Compact review of retrieval results: failures plus selected evidence."""
import json
from pathlib import Path

HERE = Path(__file__).parent
data = json.loads((HERE / "retrieval_results.json").read_text(encoding="utf-8"))
rows = data["results"]

fails = [r for r in rows if r.get("metrics", {}).get("expected_pages") and not r["metrics"]["recall_selected"]]
lines = [f"cases with no selected match: {len(fails)}", ""]
for r in fails:
    lines.append("=" * 100)
    lines.append(f'{r["id"]} [{r["category"]}] {r["question"]}')
    lines.append(f'  expected {r["metrics"]["expected_pages"]} | terms {r["metrics"]["terms_expected"]} '
                 f'| term_cov={r["metrics"]["term_coverage_selected"]} | sim={r["best_similarity_before_gate"]}')
    lines.append(f'  normalized: {r["normalized"]!r} | correction={r["spelling_correction"]}')
    lines.append("  selected:")
    for hit in r["selected"]:
        lines.append(f'    - {hit["doc"]:<12} {hit["page"]:<9} sim={hit["similarity"]} :: {hit["snippet"]}')
    lines.append("  top ranked:")
    for hit in r["ranked"][:6]:
        lines.append(f'    {hit["rank"]:<2} {hit["doc"]:<12} {hit["page"]:<9} sim={hit["similarity"]} '
                     f'sem={hit["semantic"]} lex={hit["lexical"]} :: {hit["snippet"][:120]}')
(HERE / "failures.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(fails)} failures written")
