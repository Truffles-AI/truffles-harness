"""Real local Qdrant, deterministic test embeddings, no credentials."""

from truffles import rag
from truffles.testing import HashEmbeddings

knowledge = rag(
    ["Qdrant stores vectors.", "Truffles orchestrates agents.", "Python is a language."],
    embedding=HashEmbeddings(),
    backend="qdrant",
    algorithm="hybrid",
    store_options={"location": ":memory:", "collection_name": "truffles_demo"},
    k=2,
)
try:
    for doc in knowledge.retrieve("Qdrant vectors"):
        print(doc.page_content)
finally:
    knowledge.store.client.close()
