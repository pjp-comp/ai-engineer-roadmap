"""
PHASE 2 — A real vector database (FAISS), with metadata and filtering.

What's new vs Phase 1:
    - We store METADATA next to each item (here: a category like "animal").
    - We use a real search index (FAISS) instead of a hand-written loop.
    - We can FILTER results by metadata (e.g. only show "animal" matches).

A real vector DB stores three things per item:
    vector (the meaning)  +  original data (the text)  +  metadata (tags).

Run (from the project root, with the venv active):
    python examples/phase2/example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class Item:
    """One record in our store: the text plus a metadata tag."""

    text: str
    category: str


model = load_model()

# 1. Items now carry metadata (a category), not just bare text.
items = [
    Item("cat", "animal"),
    Item("dog", "animal"),
    Item("puppy", "animal"),
    Item("apple", "fruit"),
    Item("orange", "fruit"),
    Item("car", "vehicle"),
    Item("bus", "vehicle"),
    Item("train", "vehicle"),
]
texts = [item.text for item in items]

# 2. Turn the text into vectors. float32 is the format FAISS expects.
vectors = model.encode(texts, normalize_embeddings=True).astype("float32")
dim = vectors.shape[1]  # 384

# 3. Build a real FAISS index and add our vectors to it.
#    IndexFlatIP = inner product; on normalized vectors that equals cosine similarity.
index = faiss.IndexFlatIP(dim)
index.add(vectors)
print(f"Stored {index.ntotal} items in a FAISS index (each {dim} numbers).\n")

# 4. Search.
query = "kitten"
query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
k = len(items)  # ask for all, so we can also demo filtering
scores, ids = index.search(query_vector, k)

print(f'Searching for: "{query}"\n')
print("All matches (with metadata):")
for score, idx in zip(scores[0], ids[0]):
    item = items[idx]
    print(f"  {item.text:>7}  [{item.category:<7}]  similarity={score:0.3f}")

# 5. Metadata filtering: keep only the "animal" results.
print('\nNow filtered to category == "animal" only:')
for score, idx in zip(scores[0], ids[0]):
    item = items[idx]
    if item.category == "animal":
        print(f"  {item.text:>7}  similarity={score:0.3f}")

print(
    "\nThis is the shape of a real vector DB:"
    "\n  store (vector + text + metadata) -> index -> similarity search -> filter."
)
