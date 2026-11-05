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

Keep the dataset stable across baseline and upgraded builds so Recall@K, Precision@K and MRR comparisons are meaningful. Keep a separate held-out set for final validation.
