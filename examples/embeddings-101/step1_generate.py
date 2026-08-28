"""
ZERO-TO-HERO · STEP 1 — Generate embedding vectors from your text file.

What this does:
    1. Reads my_text.txt
    2. Splits it into chunks (one per paragraph)
    3. Turns each chunk into a VECTOR (a list of 384 numbers) with an AI model
    4. Saves the vectors + chunks so steps 2 and 3 can use them

An "embedding" is just a list of numbers that captures the MEANING of the text.
Similar meaning -> similar numbers. That's the whole idea; everything else builds
on it.

Want the theory — the different WAYS to make embeddings, and what real AI apps use?
    See HOW_EMBEDDINGS_WORK.md next to this file.

Run (from project root, venv active):
    python examples/embeddings-101/step1_generate.py
"""

import numpy as np

from common import CHUNKS_PATH, VEC_PATH, read_chunks
from shared import load_model  # examples/shared.py — one shared embedding model


def main():
    # 1) read + chunk
    chunks = read_chunks()
    print(f"Read {len(chunks)} chunks (paragraphs) from my_text.txt:\n")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c[:70]}{'…' if len(c) > 70 else ''}")

    # 2) load the model and embed every chunk
    model = load_model()
    vectors = model.encode(chunks, normalize_embeddings=True)
    #   normalize=True -> each vector has length 1, which makes comparing them
    #   with cosine similarity clean and simple (see step 2).

    print(f"\nEach chunk is now a vector of {vectors.shape[1]} numbers.")
    print("Here are the first 8 numbers of chunk [1]'s vector:")
    print("  ", np.round(vectors[0][:8], 4))

    # 3) save vectors (a matrix) + chunks (one per line) for the next steps
    np.save(VEC_PATH, vectors)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))

    print(f"\nSaved {vectors.shape[0]} vectors to {VEC_PATH.split('/')[-1]} "
          f"(shape {vectors.shape}).")
    print("Next: run step2_test.py to CHECK the embeddings are good.")


if __name__ == "__main__":
    main()
