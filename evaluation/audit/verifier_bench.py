"""Isolated verifier benchmark: identical claim/evidence pairs, 3B vs 7B.

Calls app.rag.verification.verify_claims with the production prompt so the only
variable is the model. Measures latency and verdict accuracy against manually
labelled support decisions derived from the real passages.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.rag.verification import verify_claims  # noqa: E402
from app.retrieval.ollama import generate  # noqa: E402

OUT = Path(__file__).with_name("verifier_results.json")

HILL_FEARS = (
    "[1] think-and-grow-rich-napoleon-hill.pdf (Page 432):\n"
    "The fear of POVERTY, The fear of CRITICISM, the fear of ILL HEALTH (these three at the bottom "
    "of most of one's worries) The fear of LOSS OF LOVE OF SOMEONE The fear of OLD AGE The fear of "
    "DEATH All other fears are of minor importance, they can be grouped under these six headings."
)
FRANK_JUNTO = (
    "[1] the-autobiography-of-benjamin-franklin-benjamin-franklin.pdf (Page 37):\n"
    "In the autumn of the preceding year, I had form'd most of my ingenious acquaintance into a club "
    "of mutual improvement, which we called the JUNTO; we met on Friday evenings."
)
ENCH_51 = (
    "[1] the-enchiridion-epictetus.pdf (Page 5):\n"
    "The Enchiridion is made up of 51 short chapters, numbered and without titles, and each chapter "
    "stands on its own, so you can read them in order or open the book at any chapter."
)
SCI_GRATITUDE = (
    "[1] the-science-of-getting-rich.pdf (Page 74):\n"
    "Gratitude unifies the mind of man with the intelligence of Substance, so that man's thoughts are "
    "received by the Formless."
)
DRAC_VAN_HELSING = (
    "[1] dracula-bram-stoker.pdf (Page 218):\n"
    "I have written to my old friend and master, Professor Van Helsing, of Amsterdam, who knows as "
    "much about obscure diseases as any one in the world."
)
HILL_STEPS = (
    "[1] think-and-grow-rich-napoleon-hill.pdf (Page 69):\n"
    "Second. Determine exactly what you intend to give in return for the money you desire. "
    "Third. Establish a definite date when you intend to possess the money you desire."
)

CASES = [
    {
        "id": "V01",
        "question": "What are the six basic fears according to Napoleon Hill?",
        "evidence": HILL_FEARS,
        "draft": "Hill lists poverty, criticism, ill health, loss of the love of someone, old age and death as the six basic fears [1].",
        "supported": True,
        "kind": "supported_list",
    },
    {
        "id": "V02",
        "question": "Who founded the Junto?",
        "evidence": FRANK_JUNTO,
        "draft": "Benjamin Franklin founded the Junto, a club of mutual improvement that met on Friday evenings [1].",
        "supported": True,
        "kind": "supported_fact",
    },
    {
        "id": "V03",
        "question": "How many chapters are in the Enchiridion?",
        "evidence": ENCH_51,
        "draft": "The Enchiridion has 51 short chapters, numbered and without titles [1].",
        "supported": True,
        "kind": "supported_number",
    },
    {
        "id": "V04",
        "question": "What does Wattles say gratitude does?",
        "evidence": SCI_GRATITUDE,
        "draft": "Gratitude unifies the mind with the intelligence of Substance so that a person's thoughts are received by the Formless [1].",
        "supported": True,
        "kind": "supported_paraphrase",
    },
    {
        "id": "V05",
        "question": "How many chapters are in the Enchiridion?",
        "evidence": ENCH_51,
        "draft": "The Enchiridion has 52 short chapters [1].",
        "supported": False,
        "kind": "wrong_number",
    },
    {
        "id": "V06",
        "question": "Who is Professor Van Helsing?",
        "evidence": DRAC_VAN_HELSING,
        "draft": "Van Helsing is a professor from Amsterdam who hunts vampires with a rifle [1].",
        "supported": False,
        "kind": "unsupported_addition",
    },
    {
        "id": "V07",
        "question": "What are the six basic fears according to Napoleon Hill?",
        "evidence": SCI_GRATITUDE,
        "draft": "The six basic fears are poverty, criticism, ill health, loss of love, old age and death [1].",
        "supported": False,
        "kind": "wrong_passage",
    },
    {
        "id": "V08",
        "question": "What are the six basic fears according to Napoleon Hill?",
        "evidence": HILL_FEARS,
        "draft": "Hill lists poverty and criticism among the six basic fears [1].",
        "supported": False,
        "kind": "incomplete_set",
    },
    {
        "id": "V09",
        "question": "What is the second of Hill's six steps?",
        "evidence": HILL_STEPS,
        "draft": "The second step is to determine exactly what you intend to give in return for the money [1].",
        "supported": True,
        "kind": "supported_step",
    },
]


def main():
    models = sys.argv[1:] or ["qwen2.5:3b", "qwen2.5:7b"]
    output = {"cases": len(CASES), "runs": {}}
    for model in models:
        rows = []
        print(f"\n=== {model} ===", flush=True)
        for case in CASES:
            diagnostics = {}
            started = time.perf_counter()
            try:
                verdict = verify_claims(
                    case["question"], case["evidence"], case["draft"], diagnostics,
                    generate_fn=lambda prompt, num_predict=None: generate(
                        prompt, num_predict=num_predict, model=model),
                    enabled=True, max_tokens=128,
                )
                error = None
            except Exception as exc:  # noqa: BLE001
                verdict = None
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            correct = (verdict == case["supported"]) if verdict is not None else False
            rows.append({
                "id": case["id"], "kind": case["kind"], "expected": case["supported"],
                "verdict": verdict, "correct": correct,
                "verdict_text": diagnostics.get("verdict"),
                "claims_checked": diagnostics.get("claims_checked"),
                "seconds": round(elapsed, 2), "error": error,
            })
            print(f'{case["id"]:<4} {case["kind"]:<24} expected={case["supported"]!s:<5} '
                  f'got={verdict!s:<5} correct={correct!s:<5} {elapsed:6.1f}s '
                  f'raw={str(diagnostics.get("verdict"))[:40]!r}', flush=True)
        correct_count = sum(1 for row in rows if row["correct"])
        seconds = [row["seconds"] for row in rows]
        output["runs"][model] = {
            "rows": rows,
            "accuracy": round(correct_count / len(rows), 3),
            "mean_seconds": round(sum(seconds) / len(seconds), 2),
            "max_seconds": max(seconds),
        }
        print(f'{model}: accuracy={correct_count}/{len(rows)} mean={sum(seconds)/len(seconds):.1f}s', flush=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
