"""
VECTOR STORE — hold the vectors and find the nearest ones.

A thin wrapper around a FAISS index. It stores the item vectors and, given a
query vector, returns the closest items by cosine similarity.

We use IndexFlatIP (inner product). Because every vector is normalized to
length 1, inner product == cosine similarity.
"""

import faiss
import numpy as np


class VectorStore:
    """A tiny FAISS-backed store: add a matrix of vectors, then search."""

    def __init__(self, index):
        self.index = index

    @classmethod
    def from_matrix(cls, matrix):
        """Build a fresh index from a matrix of (normalized) vectors."""
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index)

    @property
    def size(self):
        return self.index.ntotal

    def search(self, query_vector, k=4):
        """Return (score, item_index) pairs for the k nearest vectors."""
        scores, idxs = self.index.search(np.array([query_vector]), k)
        return [(float(s), int(i)) for s, i in zip(scores[0], idxs[0]) if i >= 0]

    def save(self, path):
        """Write the FAISS index to disk."""
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path):
        """Read a FAISS index back from disk."""
        return cls(faiss.read_index(path))
