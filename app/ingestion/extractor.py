"""Local, multi-format document text extraction.

Extraction is deterministic and never downloads models or contacts remote services.
"""
from __future__ import annotations
import csv, io, json, logging, re, zipfile, os, hashlib, itertools
from pathlib import Path
from xml.etree import ElementTree

from app.core.config import settings

logger = logging.getLogger(__name__)
SUPPORTED = {'.xml','.xlsm','.jpg','.webp','.bmp','.csv','.htm','.pptx','.md','.txt','.html','.tiff','.rtf','.yml','.png','.tif','.log','.eml','.json','.ods','.odt','.docx','.jpeg','.pdf','.xlsx','.yaml'}

class NoReadableTextError(ValueError):
    """Raised when a supported file contains no extractable text."""


def _block(text: str, location: str, **extra):
    value = re.sub(r'\x00', '', str(text or '')).strip()
    return (location, value) if value else None


def _clean_cell(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", value)


def _looks_tabular(text):
    lines=[line for line in str(text or "").splitlines() if line.strip()]
    if len(lines)<2: return False
    return sum(bool(re.search(r"\S\s{2,}\S", line) or line.count(" | ")>=2) for line in lines)/len(lines) >= 0.5


def _table_blocks(rows, title="Table", rows_per_block=1000):
    """Render tabular rows without dropping blank-header columns or source row 1."""
    iterator = iter(rows)
    try:
        first = [_clean_cell(c) for c in next(iterator)]
    except StopIteration:
        return
    width = max(len(first), 1)
    labels = " | ".join(
        first[i] if i < len(first) and first[i] else f"Column {i + 1}"
        for i in range(width)
    )
    first_values = " | ".join(first[i] if i < len(first) else "" for i in range(width))
    batch_size = max(1, int(rows_per_block))
    pending = []
    for row_number, row in enumerate(iterator, 2):
        pending.append((row_number, [_clean_cell(c) for c in row]))
        if len(pending) >= batch_size:
            parts = [f"Columns: {labels}", f"First source row (header candidate): {first_values}"]
            for offset, values in pending:
                padded = values + [""] * max(0, width - len(values))
                parts.append(f"Row {offset}: " + " | ".join(padded))
            yield (title, "\n".join(parts))
            pending = []
    if pending:
        parts = [f"Columns: {labels}", f"First source row (header candidate): {first_values}"]
        for offset, values in pending:
            padded = values + [""] * max(0, width - len(values))
            parts.append(f"Row {offset}: " + " | ".join(padded))
        yield (title, "\n".join(parts))
    else:
        yield (title, f"Columns: {labels}\nFirst source row (header candidate): {first_values}")


def _iter_ods_tables(source):
    """Yield (sheet_name, row_iterator) pairs from ODS content.xml incrementally."""
    table_ns = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    office_ns = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    row_tag = f"{{{table_ns}}}table-row"
    table_tag = f"{{{table_ns}}}table"
    cell_tags = {f"{{{table_ns}}}table-cell", f"{{{table_ns}}}covered-table-cell"}
    name_attr = f"{{{table_ns}}}name"
    repeat_attr = f"{{{table_ns}}}number-columns-repeated"
    repeat_rows_attr = f"{{{table_ns}}}number-rows-repeated"
    value_attrs = (f"{{{office_ns}}}value", f"{{{office_ns}}}date-value",
                   f"{{{office_ns}}}string-value")
    archive = zipfile.ZipFile(source)
    xml_stream = archive.open("content.xml")
    events = iter(ElementTree.iterparse(xml_stream, events=("start", "end")))
    stack = []
    try:
        for event, element in events:
            if event == "start":
                stack.append(element)
                if element.tag != table_tag:
                    continue
                sheet_name = element.attrib.get(name_attr, "Sheet")
                table_element = element

                def rows_for_table():
                    row_count = 0
                    for row_event, row_element in events:
                        if row_event == "start":
                            stack.append(row_element)
                            continue
                        if row_element.tag == row_tag:
                            repeated_rows = int(row_element.attrib.get(repeat_rows_attr, "1") or "1")
                            if repeated_rows < 1 or repeated_rows > 1000:
                                raise ValueError("ODS row repetition exceeds the configured safety limit.")
                            row_count += repeated_rows
                            if row_count > 1_000_000:
                                raise ValueError("ODS sheet exceeds the configured row safety limit.")
                            row = []
                            for cell in list(row_element):
                                if cell.tag not in cell_tags:
                                    continue
                                value = " ".join(part.strip() for part in cell.itertext() if part.strip())
                                if not value:
                                    value = next((cell.attrib[attr] for attr in value_attrs if attr in cell.attrib), "")
                                repeated = int(cell.attrib.get(repeat_attr, "1") or "1")
                                if repeated < 1 or repeated > 1000:
                                    raise ValueError("ODS cell repetition exceeds the configured safety limit.")
                                row.extend([value] * repeated)
                            parent = stack[-2] if len(stack) > 1 else None
                            if parent is not None:
                                parent.remove(row_element)
                            row_element.clear()
                            if stack and stack[-1] is row_element:
                                stack.pop()
                            for _ in range(repeated_rows):
                                yield row
                        elif row_element is table_element:
                            parent = stack[-2] if len(stack) > 1 else None
                            if parent is not None:
                                parent.remove(row_element)
                            row_element.clear()
                            if stack and stack[-1] is row_element:
                                stack.pop()
                            return
                        elif stack and stack[-1] is row_element:
                            stack.pop()

                yield sheet_name, rows_for_table()
            elif stack and stack[-1] is element:
                stack.pop()
    finally:
        xml_stream.close()
        archive.close()


def iter_extract(path: str | Path, content: bytes | None = None, progress_callback=None):
    path = Path(path)
    suffix = path.suffix.lower()
    raw = content
    source = io.BytesIO(raw) if raw is not None else path
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file extension: {suffix or '(none)'}")
    if suffix == '.pdf':
        import fitz
        with (fitz.open(stream=raw, filetype='pdf') if raw is not None else fitz.open(path)) as doc:
            if len(doc) > settings.max_pdf_pages:
                raise ValueError(f"PDF exceeds configured page limit ({settings.max_pdf_pages})")
            for i, page in enumerate(doc, 1):
                if progress_callback:
                    progress_callback("Extraction", 10 + int(25 * (i - 1) / max(1, len(doc))), f"Reading PDF page {i}/{len(doc)}")
                text = page.get_text("text") or ""
                if len(text.strip()) < 20 and page.get_images() and not settings.ocr_enabled:
                    raise NoReadableTextError(
                        f"PDF page {i} in {path.name} appears image-based. Enable local OCR to avoid incomplete extraction."
                    )
                if len(text.strip()) < 20 and settings.ocr_enabled:
                    try:
                        import pytesseract
                        from PIL import Image
                        rect = page.rect
                        requested_scale = 1.5
                        estimated_pixels = max(1.0, rect.width * requested_scale) * max(1.0, rect.height * requested_scale)
                        if estimated_pixels > settings.max_image_pixels:
                            requested_scale *= (settings.max_image_pixels / estimated_pixels) ** 0.5
                        pix = page.get_pixmap(matrix=fitz.Matrix(requested_scale, requested_scale), alpha=False)
                        image = Image.open(io.BytesIO(pix.tobytes("png")))
                        if image.width * image.height > settings.max_image_pixels:
                            raise ValueError("Rendered PDF page exceeds image pixel limit")
                        if settings.tesseract_cmd:
                            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
                        text = pytesseract.image_to_string(image, timeout=settings.ocr_timeout_seconds)
                    except Exception as exc:
                        logger.exception("OCR failed on %s page %s", path.name, i)
                        raise RuntimeError(f"OCR failed on {path.name}, page {i}: {exc}") from exc
                b = _block(text, f"Page {i}")
                if b: yield b
                if progress_callback:
                    progress_callback("Extraction", 10 + int(25 * i / max(1, len(doc))), f"Read PDF page {i}/{len(doc)}")
        return
    if suffix == '.docx':
        from docx import Document
        doc = Document(io.BytesIO(raw) if raw is not None else str(path))
        from docx.text.paragraph import Paragraph
        from docx.table import Table
        pidx=tidx=0
        paragraph_buffer=[]
        def flush_paragraphs():
            nonlocal pidx, paragraph_buffer
            if paragraph_buffer:
                pidx += 1
                b=_block("\n".join(paragraph_buffer), "document" if pidx == 1 else f"document part {pidx}")
                paragraph_buffer=[]
                if b: return b
            return None
        child_count = max(1, len(list(doc.element.body.iterchildren())))
        for child_index, child in enumerate(doc.element.body.iterchildren(), 1):
            if progress_callback:
                progress_callback("Extraction", 10 + int(25 * child_index / child_count), f"Reading DOCX element {child_index}/{child_count}")
            if child.tag.endswith('}p'):
                para=Paragraph(child, doc)
                if para.text.strip(): paragraph_buffer.append(para.text.strip())
            elif child.tag.endswith('}tbl'):
                b=flush_paragraphs()
                if b: yield b
                tidx += 1
                table=Table(child, doc)
                rows=[[cell.text for cell in row.cells] for row in table.rows]
                yield from _table_blocks(rows, f"Table {tidx}", settings.extraction_block_rows)
        b=flush_paragraphs()
        if b: yield b
        return
    if suffix == '.pptx':
        from pptx import Presentation
        prs = Presentation(io.BytesIO(raw) if raw is not None else str(path))
        for si, slide in enumerate(prs.slides, 1):
            texts=[]
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text.strip(): texts.append(shape.text)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows: texts.append(" | ".join(c.text for c in row.cells))
            b = _block("\n".join(texts), f"Slide {si}")
            if b: yield b
        return
    if suffix in {'.xlsx', '.xlsm'}:
        # openpyxl read-only mode streams worksheet rows rather than materializing
        # every cell in a Pandas DataFrame.
        from openpyxl import load_workbook
        workbook = load_workbook(source, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                yield from _table_blocks(sheet.iter_rows(values_only=True),
                                         f"Sheet {sheet.title}", settings.extraction_block_rows)
        finally:
            workbook.close()
        return
    if suffix == '.ods':
        for name, rows in _iter_ods_tables(source):
            yield from _table_blocks(rows, f"Sheet {name}", settings.extraction_block_rows)
        return
    if suffix == '.csv':
        if raw is not None:
            try:
                text = raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = raw.decode('cp1252', errors='replace')
            yield from _table_blocks(csv.reader(io.StringIO(text)), "CSV", settings.extraction_block_rows)
        else:
            encoding = "utf-8-sig"
            with path.open("rb") as probe:
                sample = probe.read(128 * 1024)
            try:
                sample.decode("utf-8-sig")
            except UnicodeDecodeError:
                encoding = "cp1252"
            with path.open("r", encoding=encoding, errors="replace", newline="") as stream:
                yield from _table_blocks(csv.reader(stream), "CSV", settings.extraction_block_rows)
        return
    if suffix in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}:
        from PIL import Image
        image = Image.open(io.BytesIO(raw) if raw is not None else path)
        if image.width * image.height > settings.max_image_pixels: raise ValueError("Image exceeds configured pixel limit")
        if not settings.ocr_enabled: raise NoReadableTextError("Image has no directly extractable text; enable local Tesseract OCR to process images.")
        import pytesseract
        if settings.tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        b = _block(pytesseract.image_to_string(image, timeout=settings.ocr_timeout_seconds), "Image OCR")
        if b: yield b
        return
    if suffix in {'.txt', '.md', '.log', '.yaml', '.yml'} and raw is None:
        paragraph = []
        section = 0
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                if line.strip():
                    paragraph.append(line.rstrip("\r\n"))
                elif paragraph:
                    section += 1
                    block = _block("\n".join(paragraph), f"Section {section}")
                    paragraph = []
                    if block:
                        yield block
            if paragraph:
                section += 1
                block = _block("\n".join(paragraph), f"Section {section}")
                if block:
                    yield block
        return
    if suffix == '.eml':
        from email import policy
        from email.parser import BytesParser
        message=BytesParser(policy=policy.default).parsebytes(raw if raw is not None else path.read_bytes())
        body=[]
        for part in message.walk() if message.is_multipart() else [message]:
            filename=part.get_filename()
            if filename:
                payload=part.get_payload(decode=True) or b''
                ext=Path(filename).suffix.lower()
                if ext in {'.txt','.md','.csv','.json','.yaml','.yml'}:
                    b=_block(payload.decode('utf-8',errors='replace'), f"email attachment: {filename}")
                    if b: yield b
            elif part.get_content_type() in ('text/plain','text/html'):
                try:
                    value=part.get_content()
                    if part.get_content_type() == 'text/html':
                        from html.parser import HTMLParser
                        class _MailHTML(HTMLParser):
                            def __init__(self): super().__init__(); self.parts=[]
                            def handle_data(self,data): self.parts.append(data)
                        parser=_MailHTML(); parser.feed(value); value='\n'.join(parser.parts)
                    if value: body.append(str(value))
                except Exception: pass
        b=_block(('Subject: '+str(message.get('Subject'))+'\n' if message.get('Subject') else '')+'\n'.join(body), 'email body')
        if b: yield b
        return
    if suffix == '.rtf':
        from striprtf.striprtf import rtf_to_text
        text = rtf_to_text((raw if raw is not None else path.read_bytes()).decode('utf-8', errors='replace'))
    elif suffix in {'.html','.htm'}:
        from html.parser import HTMLParser
        import codecs
        class TextParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self.pending_chars = 0
                self.skip_depth = 0
            def handle_starttag(self, tag, attrs):
                if tag.lower() in ("script", "style", "noscript"):
                    self.skip_depth += 1
            def handle_endtag(self, tag):
                if tag.lower() in ("script", "style", "noscript") and self.skip_depth:
                    self.skip_depth -= 1
            def handle_data(self, data):
                if not self.skip_depth:
                    self.parts.append(data)
                    self.pending_chars += len(data)
            def take(self):
                value = "\n".join(self.parts)
                self.parts = []
                self.pending_chars = 0
                return value
        parser = TextParser()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        section = 0
        if raw is not None:
            chunks = (raw[i:i + 65536] for i in range(0, len(raw), 65536))
            for chunk in chunks:
                parser.feed(decoder.decode(chunk))
                if parser.pending_chars >= 65536:
                    value = _block(parser.take(), f"HTML section {section + 1}")
                    if value:
                        section += 1
                        yield value
        else:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    parser.feed(decoder.decode(chunk))
                    if parser.pending_chars >= 65536:
                        value = _block(parser.take(), f"HTML section {section + 1}")
                        if value:
                            section += 1
                            yield value
        parser.feed(decoder.decode(b"", final=True))
        parser.close()
        final_text = _block(parser.take(), f"HTML section {section + 1}")
        if final_text:
            yield final_text
        return
    elif suffix in {'.xml'}:
        root=ElementTree.fromstring(raw) if raw is not None else ElementTree.parse(path).getroot(); text='\n'.join(x.strip() for x in root.itertext() if x.strip())
    elif suffix in {'.json'}:
        obj=json.loads((raw if raw is not None else path.read_bytes()).decode('utf-8-sig')); text=json.dumps(obj, ensure_ascii=False, indent=2)
    elif suffix in {'.yaml','.yml'}:
        text=(raw if raw is not None else path.read_bytes()).decode('utf-8-sig',errors='replace')
    elif suffix == '.odt':
        with zipfile.ZipFile(io.BytesIO(raw) if raw is not None else path) as zf:
            root=ElementTree.fromstring(zf.read('content.xml'))
        text='\n'.join(x.strip() for x in root.itertext() if x.strip())
    elif suffix in {'.txt','.md','.log','.eml'}:
        text=(raw if raw is not None else path.read_bytes()).decode('utf-8-sig',errors='replace')
    else:
        text=(raw if raw is not None else path.read_bytes()).decode('utf-8-sig',errors='replace')
    for i, part in enumerate(re.split(r'\n\s*\n', text), 1):
        b=_block(part, f"Section {i}")
        if b: yield b


def extract(path: str | Path, content: bytes | None = None, progress_callback=None) -> list[tuple[str, str]]:
    blocks = list(iter_extract(path, content, progress_callback=progress_callback))
    if not any(text.strip() for _location, text in blocks):
        raise NoReadableTextError(f"No readable text found in {Path(path).name}. If this is a scanned document or image, configure local Tesseract OCR.")
    return blocks


def _copy_file_durable(source: Path, destination: Path, block_size: int = 1024 * 1024) -> None:
    """Copy a staged upload to permanent storage and flush it before DB commit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        for block in iter(lambda: src.read(block_size), b""):
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_archive_limits(path: Path, suffix: str) -> None:
    """Reject archive-based office files whose expanded size is unsafe."""
    if suffix not in {".docx", ".pptx", ".xlsx", ".xlsm", ".ods", ".odt"}:
        return
    max_bytes = int(settings.max_archive_uncompressed_mb) * 1024 * 1024
    max_ratio = float(settings.max_archive_compression_ratio)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > int(settings.max_archive_entries):
                raise ValueError(
                    f"Office archive contains more than {settings.max_archive_entries} entries."
                )
            total = 0
            compressed = 0
            for info in infos:
                if info.is_dir():
                    continue
                total += int(info.file_size)
                compressed += int(info.compress_size)
                if total > max_bytes:
                    raise ValueError(
                        f"Office archive expands beyond the configured limit "
                        f"({settings.max_archive_uncompressed_mb} MB)."
                    )
                if info.file_size > 0 and info.compress_size == 0:
                    raise ValueError("Office archive contains an invalid zero-sized compressed entry.")
            ratio = total / max(1, compressed)
            if ratio > max_ratio:
                raise ValueError(
                    f"Office archive compression ratio {ratio:.1f} exceeds the configured limit ({max_ratio:.1f})."
                )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid {suffix} archive: {exc}") from exc


def ingest(path: str, content: bytes | None = None, owner_id: str = "local", progress_callback=None,
           filename: str | None = None, job_id: str | None = None) -> dict:
    """Persist, extract and index a document using bounded extraction/index batches.

    ``content`` remains supported for compatibility and tests. The background
    worker passes None so large uploads are copied and read from disk incrementally.
    """
    import uuid
    from datetime import datetime, timezone
    from app.storage.database import save_document, update_document
    from app.retrieval.chunker import chunk_text
    from app.retrieval.store import add
    from app.storage.files import atomic_write

    staged = Path(path)
    filename = Path(filename or staged.name).name
    suffix = staged.suffix.lower().replace('.upload', '')
    if suffix not in SUPPORTED:
        suffix = staged.suffixes[-2].lower() if len(staged.suffixes) >= 2 else staged.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}")

    document_id = str(uuid.uuid4())
    now_stamp = datetime.now(timezone.utc).isoformat()
    permanent = Path(settings.uploads_path) / f"{document_id}{suffix}"
    extracted_path = Path(settings.extracted_path) / f"{document_id}.jsonl"
    spool_path = Path(settings.extracted_path) / f"{document_id}.chunks.jsonl"
    permanent.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.parent.mkdir(parents=True, exist_ok=True)

    if content is not None:
        atomic_write(permanent, content)
        digest = hashlib.sha256(content).hexdigest()
    else:
        if not staged.is_file():
            raise FileNotFoundError(f"Staged upload is missing: {staged}")
        _copy_file_durable(staged, permanent)
        digest = _sha256_file(permanent)

    from app.storage.database import get_by_hash
    try:
        existing = get_by_hash(digest, owner_id)
    except Exception:
        permanent.unlink(missing_ok=True)
        raise
    if existing and existing.get("processing_status") == "indexed" and existing.get("indexing_status") == "complete":
        permanent.unlink(missing_ok=True)
        if job_id:
            from app.storage.database import update_job
            update_job(job_id, document_id=existing["document_id"], document_created=0, updated_at=now_stamp)
        return {"document_id": existing["document_id"], "filename": filename,
                "document_type": (existing.get("source_metadata") or {}).get("document_type", "generic"),
                "blocks": 0, "chunks": existing.get("chunk_count", 0), "status": "indexed", "duplicate": True}

    doc = {"document_id": document_id, "owner_id": owner_id, "filename": filename,
           "stored_filename": permanent.name, "original_path": str(permanent), "content_hash": digest,
           "file_format": suffix.lstrip('.'), "uploaded_at": now_stamp, "updated_at": now_stamp,
           "processing_status": "processing", "extraction_status": "pending", "indexing_status": "pending",
           "chunk_count": 0, "source_metadata": {"document_type": "generic"},
           "extracted_path": str(extracted_path), "processing_error": None, "last_reconciled_at": None}
    block_count = 0
    chunk_count = 0
    sample_parts = []
    sample_chars = 0
    document_saved = False
    ingestion_started = False
    try:
        save_document(doc)
        document_saved = True
        if job_id:
            from app.storage.database import update_job
            update_job(job_id, document_id=document_id, updated_at=now_stamp)
        ingestion_started = True
        _check_archive_limits(permanent, suffix)
        if progress_callback:
            progress_callback("Extraction", 10, "Extracting document text")
        # Spool extracted blocks and chunks to disk; never retain the full corpus
        # or all chunk records in RAM.
        with extracted_path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8", newline="\n") as extracted_out, \
             spool_path.open("w", encoding="utf-8", newline="\n") as chunks_out:
            for location, block_text in iter_extract(permanent, content, progress_callback=progress_callback):
                block_count += 1
                extracted_out.write(json.dumps({"source_location": location, "text": block_text}, ensure_ascii=False) + "\n")
                if sample_chars < 100_000:
                    sample = block_text[:100_000 - sample_chars]
                    sample_parts.append(sample)
                    sample_chars += len(sample)
                for chunk in chunk_text(block_text, settings.chunk_size, settings.chunk_overlap):
                    chunks_out.write(json.dumps({"text": chunk, "source_location": location}, ensure_ascii=False) + "\n")
                    chunk_count += 1
            extracted_out.flush()
            os.fsync(extracted_out.fileno())
            chunks_out.flush()
            os.fsync(chunks_out.fileno())
        if block_count == 0 or chunk_count == 0:
            raise NoReadableTextError(
                f"No readable text found in {filename}. If this is scanned, enable local Tesseract OCR."
            )
        extracted_tmp = extracted_path.with_suffix(".jsonl.tmp")
        os.replace(extracted_tmp, extracted_path)

        from app.ingestion.classify import detect_document_type
        classification = detect_document_type("\n".join(sample_parts), filename)
        update_document(document_id, extraction_status="complete",
                        source_metadata={"document_type": classification["type"], "classification": classification})
        if progress_callback:
            progress_callback("Indexing", 45, f"Prepared {chunk_count} chunks")

        batch_size = max(1, int(settings.embedding_batch_size))
        records = []
        next_index = 0
        with spool_path.open("r", encoding="utf-8") as chunks_in:
            for line in chunks_in:
                item = json.loads(line)
                records.append({
                    "id": f"{document_id}:{next_index}",
                    "text": item["text"],
                    "metadata": {"document_id": document_id, "owner_id": owner_id,
                                 "filename": filename, "source_location": item["source_location"],
                                 "chunk_index": next_index, "document_type": classification["type"]},
                })
                next_index += 1
                if len(records) >= batch_size:
                    add(records, progress_callback=progress_callback)
                    records = []
        if records:
            add(records, progress_callback=progress_callback)
        spool_path.unlink(missing_ok=True)
        update_document(document_id, processing_status="indexed", extraction_status="complete",
                        indexing_status="complete", chunk_count=chunk_count,
                        updated_at=datetime.now(timezone.utc).isoformat(), processing_error=None)
        if progress_callback:
            progress_callback("Complete", 100, "Document indexed and searchable")
        return {"document_id": document_id, "filename": filename,
                "document_type": classification["type"], "blocks": block_count,
                "chunks": chunk_count, "status": "indexed"}
    except Exception as exc:
        spool_path.unlink(missing_ok=True)
        extracted_path.with_suffix(".jsonl.tmp").unlink(missing_ok=True)
        try:
            from app.retrieval.store import delete
            delete(document_id)
        except Exception:
            logger.exception("Could not clean partially indexed chunks for %s", document_id)
        if document_saved and ingestion_started:
            update_document(document_id, processing_status="failed", indexing_status="failed",
                            chunk_count=0, processing_error=str(exc),
                            updated_at=datetime.now(timezone.utc).isoformat())
        else:
            if document_saved:
                try:
                    from app.storage.database import remove_document_record
                    remove_document_record(document_id)
                except Exception:
                    logger.exception("Could not roll back document registry after startup failure: %s", document_id)
            permanent.unlink(missing_ok=True)
        raise
