"""
COMMON — tiny shared helpers for the embeddings-101 lesson.

Keeps the three step scripts short. Nothing magic here:
    - read_chunks()  : read my_text.txt and split it into paragraphs (our "chunks")
    - VEC_PATH/...   : where step 1 saves the vectors + chunks for steps 2 and 3
"""

import os
import sys

# Make the project's shared helpers importable (examples/shared.py).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # examples/  (has shared.py)

TEXT_PATH = os.path.join(HERE, "my_text.txt")
VEC_PATH = os.path.join(HERE, "vectors.npy")     # the embeddings (a matrix)
CHUNKS_PATH = os.path.join(HERE, "chunks.txt")   # the chunks, one per line


def read_chunks(path=TEXT_PATH):
    """Read the text file and split it into chunks (one per paragraph).

    We chunk by paragraph here because it's the easiest split to *see*: each
    paragraph is one idea, so each becomes one searchable vector.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # split on blank lines -> paragraphs; collapse inner newlines/spaces
    chunks = [" ".join(p.split()) for p in raw.split("\n\n") if p.strip()]
    return chunks
