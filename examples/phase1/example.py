"""
PHASE 1 — The core idea: words become numbers, and we find the nearest meaning.

Beginner goal:
    See that "kitten" lands closest to "cat" / "puppy" and far from "train",
    even though the word "kitten" was never in our list.

There is NO database here on purpose. It is just a Python list in memory, so you
can see the raw idea with nothing hidden.

Run (from the project root, with the venv active):
    python examples/phase1/example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers.util import cos_sim
from shared import load_model  # model name + loading live in examples/shared.py

# 1. Load a small AI model that turns text into a list of numbers (an embedding).
#    The first run downloads it (~90 MB); after that it is cached and fast.
model = load_model()

# 2. Our tiny "knowledge": just 8 words.
words = ["cat", "dog", "puppy", "apple", "orange", "car", "bus", "train"]

# 3. Turn every word into numbers. Each word becomes 384 numbers (a "vector").
#    normalize_embeddings=True scales them so comparing is fair and simple.
word_vectors = model.encode(words, normalize_embeddings=True)
print(f"Each word is now a list of {len(word_vectors[0])} numbers.\n")

# 4. The thing we are searching for.
query = "kitten"
query_vector = model.encode(query, normalize_embeddings=True)

# 5. Compare the query to every word using cosine similarity.
#    Higher score (closer to 1.0) = more similar in meaning.
scores = cos_sim(query_vector, word_vectors)[0]

# 6. Pair each word with its score and sort highest-first.
ranked = sorted(zip(words, scores.tolist()), key=lambda pair: pair[1], reverse=True)

print(f'Searching for: "{query}"\n')
print("How close is each word in MEANING (1.0 = identical, 0 = unrelated):")
for word, score in ranked:
    bar = "#" * int(score * 40)  # a little visual bar
    print(f"  {word:>7}  {score:0.3f}  {bar}")

print(
    "\nNotice: 'cat' and 'puppy' are nearest, even though we never stored 'kitten'."
    "\nThat is semantic search — matching by meaning, not spelling."
)
