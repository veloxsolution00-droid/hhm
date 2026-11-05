# Balanced TOP_K Configuration

## Active defaults in this build

- Base `TOP_K`: **4**
- `RETRIEVAL_CANDIDATE_MULTIPLIER`: **5**
- Dense ChromaDB candidate pool: **20** (`4 × 5`, bounded by collection size)
- `LEXICAL_CANDIDATE_LIMIT`: **40** (unchanged)
- `HYBRID_LEXICAL_WEIGHT`: **0.35** (unchanged)

## Question-type evidence limits

| Question type | Final evidence chunk limit |
|---|---:|
| Factual | 4 |
| Explanation | 6 |
| Calculation | 5 |
| Summary | 7 |
| Comparison | 7 |

Complex question types can use more than the base four chunks. The maximum remains bounded by the existing application cap of 10.

## Important deployment step

The project archive intentionally does not include a machine-specific `.env`. If you already have an `.env` file, edit its existing setting:

```env
TOP_K=4
```

An existing `.env` value takes precedence over the Python default and `.env.example`. Restart the FastAPI backend after changing it.

## Expected performance effect

The dense candidate request decreases from 40 to 20 with the default multiplier, which reduces the amount of dense candidate data passed into ranking. Factual answers also pass at most four selected chunks to the LLM. Explanations, calculations, summaries, and comparisons retain larger evidence budgets.

This does **not** guarantee a 50% reduction in end-to-end response time: query embedding, keyword retrieval, Ollama generation, verification, and disk/database overhead remain. A real latency/quality benchmark requires the target machine, its Ollama models, and the actual indexed corpus.

## Validation performed

- Regression test asserts the five question-type limits above.
- Regression test asserts the existing hard maximum remains enforced.
- Full test suite: **116 passed, 1 skipped**.
- Python compile check: passed.
