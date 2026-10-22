# Validation report

Validation date: 2026-10-19. No live credentials or paid model calls were used.

## Results

- Python 3.12.13 on macOS: **112 tests passed**, **99.61% combined statement/branch coverage**.
- Python 3.11.15 on macOS: **112 tests passed**.
- Python 3.14.4: clean base-wheel installation and agent/async/RAG/workflow smoke checks passed.
- Ruff lint and formatting: passed.
- Documentation: 19 Markdown files checked for local links; 42 Python snippets syntax-checked.
- Four offline example programs: executed successfully.
- Real local vector stores: InMemoryVectorStore, Qdrant, Chroma and FAISS.
- Provider contracts: actual native SDK serialization/transport and response parsing,
  plus explicit current Perplexity search endpoints, with intercepted HTTP.
- The test suite blocks outbound network sockets and uses synthetic credentials.
- CI is configured for Linux Python 3.11/3.12/3.13; those remote CI jobs have not
  been run from this local checkout.

One upstream deprecation warning remains: the optional FAISS integration imports
`langchain-community`, whose maintainers are sunsetting that package. The warning
is not hidden and does not fail the tests. See [compatibility](compatibility.md).

## Example output

Commands: `.venv/bin/python examples/offline.py`, `examples/async_team.py`,
`examples/local_qdrant.py`, and `examples/graph.py`.

```text
ReAct: 7 × 6 = 42.
RAG: LangChain and LangGraph [1].
Sources: ['Truffles wraps LangChain and LangGraph.']
Pipeline: An edited draft.
Parallel: {'research': 'Evidence.', 'critique': 'Limitations.'}
Supervisor: Summary of the evidence.
{'writer': 'Draft', 'reviewer': 'Review'}
asyncio [1].
Qdrant stores vectors.
Python is a language.
{'count': 3}
```

The hybrid Qdrant example and regression tests verify that backend-injected
metadata does not cause duplicate chunks in fused results.

## Local benchmark sample

Command: `.venv/bin/python scripts/benchmark.py`. Scripted models and a synthetic
corpus only; numbers are hardware-dependent and do not measure live inference.

```json
{
  "python": "3.12.13",
  "truffles_import_ms_median": 4.636,
  "wrapped_agent_ms_per_call_median": 0.506,
  "native_same_graph_ms_per_call_median": 0.491,
  "bm25_10000_docs_ms_per_query": 0.046
}
```

## Resolved integration versions

The complete resolution, including platform/Python markers, is in `uv.lock`.

| Package | Tested version |
| --- | --- |
| langchain | 1.4.6 |
| langchain-core | 1.6.12 |
| langgraph | 1.2.16 |
| langchain-openai | 1.7.2 |
| openai | 3.35.1 |
| langchain-anthropic | 1.7.6 |
| anthropic | 1.15.0 |
| langchain-google-genai | 4.4.2 |
| google-genai | 2.31.0 |
| langchain-groq | 1.1.3 |
| groq | 0.37.1 |
| langchain-xai | 1.3.0 |
| langchain-perplexity | 1.4.1 |
| perplexityai | 0.44.0 |
| langchain-qdrant | 1.1.0 |
| qdrant-client | 1.19.1 |
| langchain-chroma | 1.1.0 |
| chromadb | 1.5.9 |
| faiss-cpu | 1.15.1 |

## Full test logs

<details>
<summary>Python 3.12 — pytest --cov=truffles --cov-fail-under=90 -q --tb=short</summary>

```text
........................................................................ [ 64%]
........................................                                 [100%]
=============================== warnings summary ===============================
tests/test_retrieval.py::test_additional_local_stores[similarity-faiss]
  /Users/tanmayarora/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
    return _bootstrap._gcd_import(name[level:], package, level)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================ tests coverage ================================
______________ coverage: platform darwin, python 3.12.13-final-0 _______________

Name                        Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------
src/truffles/__init__.py       13      0      2      0   100%
src/truffles/agents.py         74      0     26      0   100%
src/truffles/knowledge.py      98      0     36      0   100%
src/truffles/providers.py      58      1     24      1    98%   36
src/truffles/results.py        35      0     10      0   100%
src/truffles/retrieval.py     137      0     36      0   100%
src/truffles/search.py         68      0     24      1    99%   91->93
src/truffles/stores.py         21      0      8      0   100%
src/truffles/testing.py        46      0      6      0   100%
src/truffles/workflows.py      43      0     14      0   100%
-----------------------------------------------------------------------
TOTAL                         593      1    186      2    99%
Required test coverage of 90% reached. Total coverage: 99.61%
112 passed, 1 warning in 6.34s

```

</details>

<details>
<summary>Python 3.11 — python -m pytest -q --tb=short</summary>

```text
........................................................................ [ 64%]
........................................                                 [100%]
=============================== warnings summary ===============================
tests/test_retrieval.py::test_additional_local_stores[similarity-faiss]
  /Users/tanmayarora/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/lib/python3.11/importlib/__init__.py:126: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
    return _bootstrap._gcd_import(name[level:], package, level)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
112 passed, 1 warning in 11.91s

```

</details>


## Build and isolated wheel installation

`python -m build` produced `dist/truffles-0.1.0-py3-none-any.whl` and
`dist/truffles-0.1.0.tar.gz`. The source archive includes documentation, examples,
tests and lockfile; it contains no virtual environments, caches or credential files.
The wheel includes the `py.typed` marker and only the package/distribution files.

The wheel was installed into a fresh Python 3.14 environment with **base
dependencies only**, then imported from `/tmp` using isolated Python (`-I`).
This prevents the editable checkout from accidentally satisfying the smoke test.
The OpenAI extra was deliberately absent to verify the installation hint.

```text
Python: 3.14.4
Imported installed wheel: /private/tmp/truffles-wheel-smoke/lib/python3.14/site-packages/truffles/__init__.py
Agent, async agent, RAG, workflow and missing-extra checks passed.
```

Full provider/vector-store tests were run on Python 3.11 and 3.12. The Python 3.14
check is a base-package smoke test, not the full provider suite.

## Scope

Stub tests establish the exercised wire formats and local behavior. They do not
verify account access, service availability, model quality, real search freshness,
remote vector database deployments or every backend capability. The remaining
remote factories are native adapters, not live-verified services.

## PFM extension validation

The IB3 reference application and speech adapter add a new validation layer. The current
combined run passes 183 tests with coverage rounded to 96%; the original results above are
retained as historical harness validation. See [PFM evidence](../apps/pfm/docs/testing.md)
for the current full test logs, live synthetic model/audio checks, browser recording,
standalone locked installation and container smoke checks.
