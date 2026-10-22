# Quickstart

## Install this checkout

Python 3.11 or later is required. Python 3.12 is the primary development environment.

```bash
python -m pip install -e .
# Choose only the integrations you need:
python -m pip install -e '.[openai,qdrant]'
# Or install all six named chat providers:
python -m pip install -e '.[providers]'
```

Use single quotes around extras in shells such as zsh. The distribution is named
`truffles`, but this checkout has not been published. Use the local path or the
built wheel, rather than assuming a package with that name on PyPI is this project.

For the complete development environment:

```bash
uv sync --locked --extra providers --extra memory --extra qdrant --extra chroma --extra faiss
uv run pytest
uv run python examples/offline.py
```

## First run without an API key

```python
from langchain_core.messages import AIMessage
from truffles import agent
from truffles.testing import ScriptedChatModel

model = ScriptedChatModel(responses=[AIMessage(content="Hello from Truffles!")])
print(agent(model).run("Say hello").text)
```

The script defines exactly what the fake model returns. It does not perform AI
inference. Exhausting its responses raises an error instead of looping silently.

## Use a live provider

Set the provider's environment variable in your own shell. Truffles does not read
`.env` files automatically or save keys.

```bash
export OPENAI_API_KEY='your-key'
```

```python
from truffles import agent
bot = agent("openai:gpt-4.1-mini", instructions="Answer briefly.")
print(bot.run("What is RAG?").text)
```

Model identifiers in examples illustrate the API. Choose an available model in
your account; no model is substituted automatically.

## Add a tool

```python
from truffles import agent, tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

bot = agent("openai:gpt-4.1-mini", tools=[multiply], max_model_calls=5)
result = bot.run("What is 17 times 23?")
print(result.text)
```

Use type annotations and a useful docstring. Native LangChain tools and ordinary
annotated functions also work. The model selects tool arguments; LangChain
validates and executes them and feeds their results back into the loop.

## Add knowledge

```python
from truffles import rag
knowledge = rag(["The office opens at 09:00.", "Support is available on weekdays."])
print(knowledge.retrieve("office opens")[0].page_content)
```

This retrieval call is fully local. To generate an answer:

```python
answer = knowledge.ask("When does the office open?", model="openai:gpt-4.1-mini")
print(answer.text)
print(answer.sources)
```

For an agent that chooses when to retrieve, pass `knowledge=knowledge` to `agent`.
For guaranteed retrieval before generation, use `knowledge.ask` as above.

## Compose agents

```python
from truffles import agent, sequence
writer = agent("openai:gpt-4.1-mini", instructions="Write a concise draft.")
editor = agent("openai:gpt-4.1-mini", instructions="Edit the supplied draft.")
pipeline = sequence(writer, editor)
print(pipeline.invoke("Explain embeddings to a beginner."))
```

Workflows use `invoke`, `ainvoke`, `batch`, `abatch`, `stream` and `astream` from
LangChain. Agents additionally offer `run` and `arun`, which return Truffles
`Result` objects. See [the API reference](api.md) for this distinction.
