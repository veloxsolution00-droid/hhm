"""Isolated extraction test across supported formats (never touches the index).

Creates small synthetic files in a temp directory and calls
app.ingestion.extractor.extract(), which only reads a file and returns labelled
text blocks. No production document, chunk, vector, or FTS row is written.

Usage: python evaluation/audit/format_probe.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.ingestion.extractor import SUPPORTED, extract  # noqa: E402


def make_txt(path: Path):
    path.write_text("First paragraph about the Junto.\n\nSecond paragraph about the library.\n", encoding="utf-8")


def make_csv_file(path: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "Amount", "Date"])
        writer.writerow(["Alpha", "1200", "2024-01-05"])
        writer.writerow(["Beta", "3400", "2024-02-11"])


def make_docx(path: Path):
    from docx import Document

    document = Document()
    document.add_heading("Quarterly Report", level=1)
    document.add_paragraph("Revenue rose by 12 percent in the second quarter.")
    table = document.add_table(rows=3, cols=2)
    rows = [["Region", "Revenue"], ["North", "100"], ["South", "200"]]
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    document.save(str(path))


def make_xlsx(path: Path):
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Budget"
    for row in [["Item", "Cost"], ["Hosting", 250], ["Support", 800]]:
        sheet.append(row)
    workbook.save(str(path))


def make_html(path: Path):
    path.write_text(
        "<html><head><style>.x{color:red}</style><script>var a=1;</script></head>"
        "<body><h1>Title</h1><p>Visible paragraph text.</p></body></html>",
        encoding="utf-8",
    )


def make_md(path: Path):
    path.write_text("# Heading\n\nMarkdown paragraph with **bold** text.\n", encoding="utf-8")


def make_json_file(path: Path):
    path.write_text(json.dumps({"alpha": 1, "nested": {"beta": [1, 2]}}), encoding="utf-8")


def make_png(path: Path):
    """Render simple text to a PNG so OCR has something deterministic to read."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 60), "JUNTO MEETS FRIDAY EVENING", fill="black", font=font)
    image.save(str(path))


def main():
    cases = [
        ("txt", make_txt), ("csv", make_csv_file), ("docx", make_docx),
        ("xlsx", make_xlsx), ("html", make_html), ("md", make_md),
        ("json", make_json_file), ("png", make_png),
    ]
    unsupported_probe = {"pptx", "rtf", "odt", "ods", "eml", "xml", "yaml", "log", "tiff", "bmp", "webp"}
    print(f"declared SUPPORTED extensions ({len(SUPPORTED)}): {sorted(SUPPORTED)}")
    print(f"declared but not exercised here: {sorted(unsupported_probe & SUPPORTED)}")

    results = []
    with tempfile.TemporaryDirectory(prefix="rnd2_fmt_") as temp:
        temp_path = Path(temp)
        for suffix, builder in cases:
            target = temp_path / f"sample.{suffix}"
            try:
                builder(target)
            except Exception as exc:  # noqa: BLE001
                print(f"{suffix:<6} SETUP FAILED {type(exc).__name__}: {exc}")
                results.append({"format": suffix, "status": "setup_failed", "error": str(exc)})
                continue
            started = time.perf_counter()
            try:
                blocks = extract(target)
                elapsed = time.perf_counter() - started
                text = " ".join(value for _loc, value in blocks)
                locations = sorted({loc for loc, _v in blocks})
                print(f"{suffix:<6} OK   blocks={len(blocks):<3} chars={len(text):<6} "
                      f"{elapsed*1000:7.0f}ms locations={locations[:4]} "
                      f"sample={text[:90]!r}")
                results.append({"format": suffix, "status": "ok", "blocks": len(blocks),
                                "chars": len(text), "ms": round(elapsed * 1000, 1),
                                "locations": locations, "sample": text[:200]})
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - started
                print(f"{suffix:<6} FAIL {type(exc).__name__}: {exc} ({elapsed*1000:.0f}ms)")
                results.append({"format": suffix, "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "ms": round(elapsed * 1000, 1)})

    out = Path(__file__).with_name("format_probe.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
