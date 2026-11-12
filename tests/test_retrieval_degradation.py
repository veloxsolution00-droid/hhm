"""Regression coverage for retrieval degradation and hybrid score consistency."""

import app.retrieval.store as store


class _Collection:
    def __init__(self, query_result=None):
        self.query_result = query_result or {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        self.query_calls = 0

    def count(self):
        return 5

    def query(self, **kwargs):
        self.query_calls += 1
        return self.query_result


def _fts_hit():
    return {
        "chunk_id": "fts-1",
        "text": "Alpha beta appear together in this source passage.",
        "document_id": "doc-1",
        "filename": "source.pdf",
        "source_location": "Page 2",
    }


def _configure(monkeypatch, collection, fts_hits=None):
    monkeypatch.setattr(store, "collection", lambda: collection)
    monkeypatch.setattr(
        store.db, "search_indexed_chunks",
        lambda query, owner_id, limit: list(fts_hits or []),
    )
    monkeypatch.setattr(store.settings, "hybrid_lexical_weight", 0.35)
    monkeypatch.setattr(store.settings, "lexical_candidate_limit", 7)
    monkeypatch.setattr(store.settings, "retrieval_candidate_multiplier", 4)
    monkeypatch.setattr(store.settings, "top_k", 5)


def test_fts_retrieval_survives_embedding_model_unavailability(monkeypatch):
    collection = _Collection()
    _configure(monkeypatch, collection, [_fts_hit()])

    def unavailable(_question):
        raise store.OllamaUnavailableError("local embedding model unavailable")

    monkeypatch.setattr(store, "embed_query", unavailable)

    hits = store.search("alpha beta", owner_id="local")

    assert collection.query_calls == 0
    assert len(hits) == 1
    assert hits[0]["id"] == "fts-1"
    assert hits[0]["semantic_similarity"] is None
    assert hits[0]["lexical_similarity"] == 1.0
    assert hits[0]["similarity"] == 0.35


def test_fts_only_candidate_uses_lexical_hybrid_weight(monkeypatch):
    collection = _Collection()
    _configure(monkeypatch, collection, [_fts_hit()])
    monkeypatch.setattr(store, "embed_query", lambda _question: [0.1, 0.2])

    hits = store.search("alpha beta", owner_id="local")

    assert collection.query_calls == 1
    assert len(hits) == 1
    # An FTS-only hit contributes w * lexical_score, not (1 - w).
    assert hits[0]["similarity"] == 0.35
