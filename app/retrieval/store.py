from pathlib import Path
import logging
import math
import time
import re
import chromadb
from app.core.config import settings
from app.storage import database as db
from app.retrieval.ollama import embed, embed_query, OllamaUnavailableError

_client = None
_collection = None
logger = logging.getLogger(__name__)

_LEXICAL_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "with",
}

def _query_terms(question):
    return {term for term in re.findall(r"[\w-]+", question.lower(), flags=re.UNICODE)
            if len(term) > 1 and term not in _LEXICAL_STOP_WORDS}

def _document_terms(text):
    return set(re.findall(r"[\w-]+", (text or "").lower(), flags=re.UNICODE))


def _lexical_similarity(question_terms, text):
    """Measure exact query-term coverage for names, dates, IDs, and table values.

    Kept for callers that expect an unweighted ratio; the retrieval path uses
    `_weighted_lexical_similarity` so rare terms outweigh ubiquitous ones.
    """
    if not question_terms:
        return 0.0
    document_terms = _document_terms(text)
    return len(set(question_terms) & document_terms) / len(set(question_terms))


def _query_term_stats(question_terms):
    """Return IDF weights, document frequencies and the indexed chunk count.

    A term that occurs in nearly every chunk (for example a repeatedly named
    person) carries little discriminating power, so it is down-weighted. A rare
    term — an identifier, a distinctive name, a numeric value — carries more.
    When index statistics are unavailable the weights are uniform, which
    reproduces the previous unweighted behaviour exactly.
    """
    terms = {term for term in question_terms if term}
    if not terms:
        return {}, {}, 0
    try:
        total, frequencies = db.indexed_lexical_statistics(terms)
    except Exception:  # noqa: BLE001 - index statistics must never break retrieval
        return {term: 1.0 for term in terms}, {}, 0
    if total <= 0:
        return {term: 1.0 for term in terms}, frequencies, total
    weights = {}
    for term in terms:
        df = max(0, int(frequencies.get(term, 0)))
        if df <= 0:
            # A term no chunk contains cannot discriminate between candidates,
            # so it is treated as maximally common (the lowest weight) instead
            # of inflating the denominator for every candidate.
            df = total
        weights[term] = math.log(1.0 + (total + 1) / (df + 1))
    return weights, frequencies, total


def _weighted_lexical_similarity(weights, text):
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    document_terms = _document_terms(text)
    matched = sum(weight for term, weight in weights.items() if term in document_terms)
    return matched / total_weight


def _quote_fts_term(term):
    # Quote individual terms so punctuation cannot become FTS query syntax.
    return f'"{term.replace(chr(34), "")}"'


def _fts_query(question_terms):
    return " OR ".join(_quote_fts_term(term) for term in sorted(question_terms))


def _fts_rare_query(terms):
    """OR over only the rarest query terms.

    Restricting the route to discriminating terms keeps a chunk that matches a
    distinctive term at the top even when ubiquitous words dominate BM25 over
    the full query. The full OR route is still run alongside this one, so no
    candidate is lost.
    """
    if len(terms) < 2:
        return ""
    return " OR ".join(_quote_fts_term(term) for term in terms)


def _fts_phrase_query(question):
    """One quoted adjacent-word-pair query for exact entity and identifier phrases.

    Expressed as a single FTS query (``"a b" OR "b c" OR ...``) so phrase
    matching costs one index round-trip instead of one per word pair.
    """
    tokens = [t for t in re.findall(r"[\w-]+", (question or "").lower(), flags=re.UNICODE)
              if len(t) > 1]
    if len(tokens) < 2:
        return ""
    phrases = []
    seen = set()
    for left, right in zip(tokens, tokens[1:]):
        phrase = f'"{left} {right}"'
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return " OR ".join(phrases)


def _rarest_terms(weights, frequencies, maximum=3):
    """Return the rarest indexed query terms (absent terms excluded).

    Sorted by IDF descending, so terms appearing in the fewest chunks come
    first. Terms with no indexed document frequency are dropped: they cannot
    match, so including them would only crowd the route.
    """
    candidates = [(term, int(frequencies.get(term, 0))) for term in weights]
    candidates = [(term, df) for term, df in candidates if df > 0]
    candidates.sort(key=lambda item: (item[1], item[0]))
    return [term for term, _ in candidates[:maximum]]


def collection():
    global _client, _collection
    if _collection is None:
        # Imported lazily so a minimal test double for the top-level chromadb
        # module (used by unit tests) does not need to provide submodules.
        from chromadb.config import Settings as ChromaSettings
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        # Telemetry is disabled explicitly: this application is local-only and
        # the bundled telemetry client raises noisy errors on some Windows
        # installs. Disabling it also avoids any outbound analytics attempt.
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Never delete or recreate an existing collection at startup.
        _collection = _client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def add(records, progress_callback=None):
    """Embed and persist chunks in bounded batches, logging stage timings."""
    c = collection()
    batch_size = max(1, int(settings.embedding_batch_size))
    total_embed_seconds = 0.0
    total_write_seconds = 0.0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        if progress_callback:
            # Pre-batch checkpoint: do not begin another embedding request after
            # a cancellation request has been observed.
            progress_callback("Embedding", 55 + int(40 * i / max(1, len(records))),
                              f"Preparing batch {i // batch_size + 1}")
        started = time.perf_counter()
        vectors = embed([x["text"] for x in batch])
        total_embed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        c.upsert(ids=[x["id"] for x in batch],
                 documents=[x["text"] for x in batch],
                 embeddings=vectors,
                 metadatas=[x["metadata"] for x in batch])
        db.index_chunks(batch)
        total_write_seconds += time.perf_counter() - started
        if progress_callback:
            completed = min(i + len(batch), len(records))
            progress_callback("Embedding and saving chunks", 55 + int(40 * completed / max(1, len(records))),
                              f"Embedded and saved {completed}/{len(records)} chunks")
    logger.info(
        "Index stages: chunks=%d batch_size=%d embedding_seconds=%.2f chroma_write_seconds=%.2f",
        len(records), batch_size, total_embed_seconds, total_write_seconds
    )

def search(question, owner_id="local"):
    """Retrieve from dense Chroma and FTS5 routes, degrading safely if Ollama is down.

    FTS5 is an independent local retrieval route. A temporary embedding-model
    failure must not disable exact lexical retrieval for already indexed chunks.
    The fallback is still owner-scoped and bounded by LEXICAL_CANDIDATE_LIMIT.
    """
    c = collection()
    collection_count = c.count()
    if not collection_count:
        return []

    question_terms = _query_terms(question)
    lexical_weight = max(0.0, min(1.0, float(settings.hybrid_lexical_weight)))
    weights, frequencies, _total_chunks = _query_term_stats(question_terms)
    q_vector = None
    try:
        q_vector = embed_query(question)
    except OllamaUnavailableError as exc:
        logger.warning(
            "Dense retrieval unavailable; continuing with FTS5: owner=%s reason=%s",
            owner_id, str(exc)[:240],
        )

    results = []
    seen_chunk_ids = set()
    candidate_count = 0
    if q_vector is not None:
        multiplier = max(1, int(settings.retrieval_candidate_multiplier))
        wanted = max(1, int(settings.top_k))
        # Do not fetch every owner chunk just to count it. Chroma safely returns
        # fewer filtered matches when this owner has fewer than requested.
        candidate_count = min(max(wanted, wanted * multiplier), collection_count)
        response = c.query(
            query_embeddings=[q_vector], n_results=candidate_count,
            where={"owner_id": owner_id},
            include=["documents", "metadatas", "distances"],
        )
        docs = (response.get("documents") or [[]])[0] or []
        metas = (response.get("metadatas") or [[]])[0] or []
        distances = (response.get("distances") or [[]])[0] or []
        ids = (response.get("ids") or [[]])[0] or []
        for chunk_id, document, metadata, distance in zip(ids, docs, metas, distances):
            if not document or distance is None:
                continue
            semantic_similarity = 1 - float(distance)
            lexical_similarity = _weighted_lexical_similarity(weights, document)
            hybrid_similarity = (
                (1 - lexical_weight) * semantic_similarity
                + lexical_weight * lexical_similarity
            )
            results.append({
                "id": chunk_id, "text": document, "metadata": metadata or {},
                "similarity": hybrid_similarity,
                "semantic_similarity": semantic_similarity,
                "lexical_similarity": lexical_similarity,
            })
            seen_chunk_ids.add(chunk_id)

    # FTS brings in exact hits that semantic retrieval may not rank highly,
    # particularly identifiers, dates, codes, and values from tables. It runs as
    # a few bounded routes over the SAME index: the full OR query (always), an
    # OR over only the rarest terms, and one combined quoted-phrase query. The
    # full OR route always runs, so the original query never loses candidates.
    lexical_limit = max(1, int(settings.lexical_candidate_limit))
    if question_terms and lexical_weight > 0:
        for fts_hit in _fts_candidates(question, question_terms, weights, frequencies,
                                       owner_id, lexical_limit):
            if fts_hit["chunk_id"] in seen_chunk_ids:
                continue
            lexical_similarity = _weighted_lexical_similarity(weights, fts_hit["text"])
            results.append({
                "id": fts_hit["chunk_id"], "text": fts_hit["text"],
                "metadata": {
                    "document_id": fts_hit["document_id"], "owner_id": owner_id,
                    "filename": fts_hit["filename"],
                    "source_location": fts_hit["source_location"],
                },
                # An FTS-only hit has no semantic component, so it contributes
                # only the lexical term of the dense hybrid score.
                "similarity": lexical_weight * lexical_similarity,
                "semantic_similarity": None,
                "lexical_similarity": lexical_similarity,
            })

    logger.info(
        "Retrieval: owner=%s dense_candidates=%d lexical_limit=%d returned=%d "
        "dense_available=%s",
        owner_id, candidate_count, lexical_limit, len(results), q_vector is not None,
    )
    return sorted(results, key=lambda item: item["similarity"], reverse=True)


def _fts_candidates(question, question_terms, weights, frequencies, owner_id, limit):
    """Merge the bounded FTS routes into one deduplicated candidate list.

    Each route is capped at `limit`, so candidate generation stays bounded. A
    chunk surfaced by several routes keeps the best (lowest) BM25 rank.
    """
    queries = []
    or_query = _fts_query(question_terms)
    if or_query:
        queries.append(or_query)
    rare_terms = _rarest_terms(weights, frequencies)
    rare_query = _fts_rare_query(rare_terms)
    if rare_query and rare_query != or_query and len(rare_terms) < len(question_terms):
        queries.append(rare_query)
    phrase_query = _fts_phrase_query(question)
    if phrase_query and phrase_query not in queries:
        queries.append(phrase_query)

    merged = {}
    for query in queries:
        for hit in db.search_indexed_chunks(query, owner_id, limit):
            chunk_id = hit["chunk_id"]
            previous = merged.get(chunk_id)
            if previous is None or _bm25_rank(hit) < _bm25_rank(previous):
                merged[chunk_id] = hit
    return list(merged.values())


def _bm25_rank(hit):
    value = hit.get("rank")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def delete(document_id):
    # Explicit deletion/recovery only; never called on normal initialization.
    collection().delete(where={"document_id": document_id})
    db.remove_indexed_chunks(document_id)

def document_chunk_count(document_id):
    c = collection()
    return len(c.get(where={"document_id": document_id}, include=["metadatas"])["ids"])

def rebuild_fts_for_document(document_id, batch_size=500):
    """Rebuild one document's FTS rows from persisted ChromaDB records."""
    c = collection()
    offset = 0
    while True:
        result = c.get(where={"document_id": document_id}, limit=batch_size, offset=offset,
                       include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids:
            break
        records = []
        for chunk_id, text, metadata in zip(ids, result.get("documents") or [],
                                             result.get("metadatas") or []):
            if text is not None and metadata:
                records.append({"id": chunk_id, "text": text, "metadata": metadata})
        db.index_chunks(records)
        offset += len(ids)
        if len(ids) < batch_size:
            break


def list_chunk_document_ids():
    c = collection()
    ids = set()
    total = c.count()
    batch_size = 500
    for offset in range(0, total, batch_size):
        result = c.get(limit=batch_size, offset=offset, include=["metadatas"])
        for meta in result.get("metadatas") or []:
            if meta and meta.get("document_id"):
                ids.add(meta["document_id"])
    return ids


def backfill_owner_metadata(documents):
    """Add ownership metadata to legacy chunks without deleting/recreating the collection."""
    c = collection()
    for doc in documents:
        docid = doc.get("document_id")
        if not docid:
            continue
        offset = 0
        while True:
            result = c.get(where={"document_id": docid}, limit=500, offset=offset,
                           include=["metadatas"])
            ids = result.get("ids", [])
            metas = result.get("metadatas", [])
            if not ids:
                break
            batch_metas = []
            for meta in metas:
                metadata = dict(meta or {})
                metadata["owner_id"] = "local"
                batch_metas.append(metadata)
            c.update(ids=ids, metadatas=batch_metas)
            offset += len(ids)
            if len(ids) < 500:
                break


def backfill_text_search():
    """Populate FTS for pre-existing ChromaDB chunks after an application update."""
    c = collection()
    total = c.count()
    if not total or db.indexed_chunk_count() >= total:
        return
    batch_size = 500
    for offset in range(0, total, batch_size):
        result = c.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        records = []
        for chunk_id, text, metadata in zip(result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or []):
            if text and metadata:
                records.append({"id": chunk_id, "text": text, "metadata": metadata})
        db.index_chunks(records)
