# Examples

All programs run without credentials or network calls after installation.

```bash
uv sync --locked --extra providers --extra memory --extra qdrant --extra chroma --extra faiss --extra processing --extra tables
uv run python examples/offline.py
uv run python examples/async_team.py
uv run python examples/local_qdrant.py
uv run python examples/graph.py
uv run python examples/evaluation.py
# Requires the processing extra and the Tesseract executable:
uv run python examples/processing.py
```

`offline.py` exercises a real graph-based tool loop, retrieval/generation, a linear
pipeline, parallel agents and supervisor delegation with scripted model responses.
`async_team.py` demonstrates async workflows and RAG. `local_qdrant.py` uses the
actual Qdrant client in local mode and hybrid retrieval. `graph.py` demonstrates a
native conditional cycle.

The scripted model is not an LLM. Hash embeddings are lexical test vectors. Replace
them with a configured live model and real semantic embeddings in an application.

`evaluation.py` runs answer, retrieval and scripted-judge evaluations with regression
gates and content-free trace exports under `.truffles/evaluation/`. See the
[evaluation guide](../docs/evaluation.md) and [observability guide](../docs/observability.md).

`processing.py` uses real local Tesseract and PDFium on generated documents, with
scripted agent and speech adapters. [Processing setup and limits](../docs/processing.md).

`tables.py` generates a sideways page and scanned PDF, corrects orientation and
extracts a ruled table into CSV/JSON using real local engines. Requires `tables` and
Tesseract with English/OSD data: `uv run python examples/tables.py`. Results require review.
