# Phase 1 — The core idea

**Big idea:** turn words into numbers that capture meaning, then find the nearest numbers.

This example has **no database** — just a Python list — so nothing is hidden.

## Run
```bash
# from the project root, with the venv active (see top-level README)
python examples/phase1/example.py
```

## What you'll see
"kitten" is closest to **cat** and **puppy**, and far from **train** — even though
"kitten" was never in our list. That's *semantic search*: matching by meaning.

## Words to know
- **Embedding** — the list of numbers that represents a word's meaning.
- **Vector** — another name for that list of numbers.
- **Cosine similarity** — a score (0–1) for how close two meanings are.

## See the vectors (companion demo)
Want to actually *see* a vector instead of just reading about it?
```bash
python examples/phase1/visualize.py
```
It prints the real 384-number vector for "kitten" and writes two files next to
the script:
- **`kitten_vector.txt`** — all 384 numbers, one per line (open in any editor).
- **`vectors_map.svg`** — the 8 words drawn as colored dots on a 2-D map
  (open in a browser). You'll see animals, fruit, and vehicles land in three
  separate clusters — that closeness *is* the "meaning" the model learned.

(The 384-D vectors are squeezed to 2-D with PCA just for drawing. Both output
files are git-ignored since they're regenerated on each run. No matplotlib
needed — it writes a plain SVG.)

→ Full explanation: [learning_notes.md](../../learning_notes.md#phase-1--the-core-idea-in-memory-demo)
