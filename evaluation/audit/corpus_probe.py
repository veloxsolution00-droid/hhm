"""Ad-hoc read-only corpus probe used to build the evaluation dataset.

Runs against the extracted JSONL artifacts only; never touches the index.
Usage: python evaluation/audit/corpus_probe.py "pattern" [pattern ...]
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.core.config import settings  # noqa: E402


def load_docs():
    con = sqlite3.connect(f"file:{settings.sqlite_path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    docs = {}
    for row in con.execute("select filename, extracted_path from documents order by filename"):
        blocks = []
        for line in open(row["extracted_path"], encoding="utf-8"):
            if line.strip():
                blocks.append(json.loads(line))
        docs[row["filename"]] = blocks
    con.close()
    return docs


def main():
    patterns = [re.compile(p, re.I) for p in sys.argv[1:]]
    docs = load_docs()
    out = []
    for filename, blocks in docs.items():
        for block in blocks:
            text = " ".join(str(block.get("text", "")).split())
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                start = max(0, match.start() - 260)
                out.append(
                    f'{filename}\t{block.get("source_location")}\t...{text[start:match.end() + 420]}...'
                )
                break
    Path(r"C:\Users\Admin\AppData\Local\Temp\opencode\probe_out.txt").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print(f"{len(out)} matching blocks -> probe_out.txt")


if __name__ == "__main__":
    main()
