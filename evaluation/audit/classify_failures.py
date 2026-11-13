"""Consolidated per-case failure classification from the retrieval run.

Case definitions (retrieval-stage only; generation-stage cases need live runs):
  C1_not_retrieved      no expected page anywhere in the ranked candidate pool
  C2_retrieved_unused   expected page was ranked but not selected as evidence
  C5_incomplete         evidence selected but missing required facts
  OK                    fully sufficient evidence selected
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
results = json.loads((HERE / "retrieval_results.json").read_text(encoding="utf-8"))["results"]
requirements = json.loads((HERE / "evidence_requirements.json").read_text(encoding="utf-8"))["requirements"]

rows = []
for result in results:
    case_id = result["id"]
    if "error" in result:
        rows.append({"id": case_id, "category": result["category"], "case": "ERROR",
                     "detail": result["error"]})
        continue
    groups = requirements.get(case_id)
    if not groups:
        rows.append({"id": case_id, "category": result["category"], "case": "UNANSWERABLE",
                     "case_class": "-", "detail": "abstention expected",
                     "best_sim": result["best_similarity_before_gate"]})
        continue
    text = " ".join(hit.get("text", "") for hit in result.get("selected", [])).lower()
    missing = [group for group in groups if not any(alt.lower() in text for alt in group)]
    metrics = result["metrics"]
    if not missing:
        case = "OK"
    elif metrics["recall@20"] and not metrics["recall_selected"]:
        case = "C2_retrieved_unused"
    elif not metrics["recall@20"]:
        case = "C1_not_retrieved"
    else:
        case = "C5_incomplete"
    rows.append({
        "id": case_id,
        "category": result["category"],
        "case": case,
        "satisfied": f'{len(groups) - len(missing)}/{len(groups)}',
        "missing": missing,
        "recall@20": metrics["recall@20"],
        "recall_selected": metrics["recall_selected"],
        "best_sim": result["best_similarity_before_gate"],
        "eligible": result["eligible_after_min_similarity"],
        "passes": result["retrieval_passes"],
        "top_k": result["top_k"],
    })

counts = Counter(row["case"] for row in rows)
print("case distribution:", dict(counts))

answerable = [row for row in rows if row["case"] not in ("UNANSWERABLE", "ERROR")]
print(f"\nanswerable adjudicated: {len(answerable)}")
for case in ("C1_not_retrieved", "C2_retrieved_unused", "C5_incomplete", "OK"):
    subset = [row for row in answerable if row["case"] == case]
    print(f"  {case:<22} {len(subset):>2}  ({len(subset)/len(answerable):.1%})")

print("\nby category:")
by_cat = defaultdict(Counter)
for row in answerable:
    by_cat[row["category"]][row["case"]] += 1
for category in sorted(by_cat):
    counts_row = by_cat[category]
    print(f"  {category:<28} OK={counts_row['OK']:<2} "
          f"C1={counts_row['C1_not_retrieved']:<2} C2={counts_row['C2_retrieved_unused']:<2} "
          f"C5={counts_row['C5_incomplete']:<2}")

print("\nC1 (never retrieved):")
for row in answerable:
    if row["case"] == "C1_not_retrieved":
        print(f'  {row["id"]:<4} best_sim={row["best_sim"]} eligible={row["eligible"]:<3} '
              f'passes={row["passes"]} missing={row["missing"]}')

print("\nC2 (retrieved but not selected):")
for row in answerable:
    if row["case"] == "C2_retrieved_unused":
        print(f'  {row["id"]:<4} best_sim={row["best_sim"]} eligible={row["eligible"]:<3} '
              f'passes={row["passes"]} top_k={row["top_k"]} missing={row["missing"]}')

print("\nC5 (incomplete evidence):")
for row in answerable:
    if row["case"] == "C5_incomplete":
        print(f'  {row["id"]:<4} satisfied={row["satisfied"]} missing={row["missing"]}')

(HERE / "case_classification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
