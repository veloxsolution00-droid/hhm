# RND2 Retrieval Evaluation Dataset

Use a manually labelled JSONL dataset. Each line must contain:
- `question`: the exact user question.
- `relevant_chunk_ids`: one or more chunk IDs known to contain relevant evidence.

Chunk IDs use the existing ChromaDB ID convention (`<document_id>:<chunk_index>`).
Do not label a chunk relevant merely because it mentions the same entity; it must contain evidence that helps answer the question.

Run:

```bash
python scripts/evaluate_retrieval.py evaluation.jsonl local 10
```

For an end-to-end answer, citation and latency run against the current indexed
corpus, start the local API and run:

```powershell
python tests/eval_live_corpus.py --api http://127.0.0.1:8000 --strict-document-coverage
```

The report lists every indexed document and identifies documents without a
targeted question case. When a new document is uploaded, add at least one
answerable question and its expected source file/page to `tests/eval_live_corpus.py`;
the strict option then fails until every indexed document is represented. The
matrix records answer status, expected answer terms, cited source/page, citation
validator and answer verifier outcomes, and response latency (mean, p50, p95,
maximum). A passing citation format alone is not counted as accurate if the
answer verifier or expected cited source/page check fails.

Keep the dataset stable across baseline and upgraded builds so Recall@K, Precision@K and MRR comparisons are meaningful. Keep a separate held-out set for final validation.
