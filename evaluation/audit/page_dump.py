"""Dump exact page text from the extracted artifacts (read-only)."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.core.config import settings  # noqa: E402


def main():
    wanted = {}
    for arg in sys.argv[1:]:
        filename_part, _, page = arg.partition("|")
        wanted.setdefault(filename_part.lower(), []).append(page)

    con = sqlite3.connect(f"file:{settings.sqlite_path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = []
    for row in con.execute("select filename, extracted_path from documents order by filename"):
        for key, pages in wanted.items():
            if key not in row["filename"].lower():
                continue
            blocks = [json.loads(line) for line in open(row["extracted_path"], encoding="utf-8") if line.strip()]
            for page in pages:
                text = "\n\n".join(
                    " ".join(str(b.get("text", "")).split())
                    for b in blocks
                    if b.get("source_location") == page
                )
                out.append(f"\n===== {row['filename']} :: {page} =====\n{text}")
    Path(r"C:\Users\Admin\AppData\Local\Temp\opencode\pages_out.txt").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print("written pages_out.txt")


if __name__ == "__main__":
    main()
