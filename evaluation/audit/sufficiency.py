"""Evidence-sufficiency scoring from the retrieval results."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
results = json.loads((HERE / "retrieval_results.json").read_text(encoding="utf-8"))["results"]
requirements = json.loads((HERE / "evidence_requirements.json").read_text(encoding="utf-8"))["requirements"]

rows = []
for result in results:
    case_id = result["id"]
    groups = requirements.get(case_id)
    if not groups:
        continue
    text = " ".join(hit.get("text", "") for hit in result.get("selected", [])).lower()
    missing = [group for group in groups if not any(alt.lower() in text for alt in group)]
    rows.append({
        "id": case_id,
        "category": result["category"],
        "expected_groups": len(groups),
        "satisfied": len(groups) - len(missing),
        "sufficient": not missing,
        "partial": 0 < len(groups) - len(missing) < len(groups),
        "missing": missing,
        "page_recall_selected": result["metrics"]["recall_selected"],
        "pages": result["metrics"]["expected_pages"],
    })

answerable = [row for row in rows]
sufficient = [row for row in answerable if row["sufficient"]]
partial = [row for row in answerable if row["partial"]]
print(f"adjudicated cases: {len(answerable)}")
print(f"fully sufficient evidence: {len(sufficient)} ({len(sufficient)/len(answerable):.3f})")
print(f"partially sufficient:      {len(partial)} ({len(partial)/len(answerable):.3f})")
print(f"insufficient:              {sum(1 for r in answerable if not r['satisfied'])}")
print("\nby category (fully sufficient / n):")
by_cat = defaultdict(list)
for row in answerable:
    by_cat[row["category"]].append(row)
for category, items in sorted(by_cat.items()):
    ok = sum(1 for item in items if item["sufficient"])
    print(f"  {category:<28} {ok}/{len(items)}")

print("\nnot fully sufficient:")
for row in answerable:
    if row["sufficient"]:
        continue
    state = "PARTIAL" if row["partial"] else "MISS"
    print(f'  {row["id"]:<4} {state:<7} {row["satisfied"]}/{row["expected_groups"]} missing={row["missing"]}')

print("\npage-recall false negatives recovered by evidence check:")
recovered = [row["id"] for row in answerable if row["sufficient"] and not row["page_recall_selected"]]
print("  " + ", ".join(recovered) if recovered else "  none")
print("\npage-recall true negatives confirmed:")
confirmed = [row["id"] for row in answerable if not row["sufficient"] and not row["page_recall_selected"]]
print("  " + ", ".join(confirmed) if confirmed else "  none")

(HERE / "sufficiency.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
