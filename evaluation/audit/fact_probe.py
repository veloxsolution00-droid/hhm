"""Compact read-only fact probe: one representative hit per document per pattern."""
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.core.config import settings  # noqa: E402

OUT = Path(r"C:\Users\Admin\AppData\Local\Temp\opencode\probe_compact.txt")


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


def short_name(filename):
    stem = re.sub(r"\.pdf$", "", filename)
    return {
        "dracula-bram-stoker": "DRAC",
        "the-autobiography-of-benjamin-franklin-benjamin-franklin": "FRANK",
        "the-enchiridion-epictetus": "ENCH",
        "the-science-of-getting-rich": "SCI",
        "think-and-grow-rich-napoleon-hill": "HILL",
    }.get(stem, stem[:8].upper())


def main():
    patterns = sys.argv[1:]
    docs = load_docs()
    lines = []
    for pattern in patterns:
        rx = re.compile(pattern, re.I)
        lines.append(f"\n### {pattern}")
        for filename, blocks in docs.items():
            hits = []
            for block in blocks:
                text = " ".join(str(block.get("text", "")).split())
                for match in rx.finditer(text):
                    hits.append((block.get("source_location"), text, match))
                    break
            lines.append(f"  [{short_name(filename)}] hits={len(hits)}")
            for location, text, match in hits[:2]:
                start = max(0, match.start() - 130)
                snippet = text[start:match.end() + 190]
                lines.append(f"    {location:<9} ...{snippet}...")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
