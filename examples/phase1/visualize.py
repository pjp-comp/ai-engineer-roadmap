"""
PHASE 1 (companion) — SEE the vectors: print one, and draw them on a map.

Beginner goal:
    Make the abstract "a word is 384 numbers" concrete. We do two things:
      1. Print the actual embedding for one word (all 384 numbers).
      2. Draw every word as a dot on a 2-D map, so you can SEE that
         similar meanings land close together (cat/dog/puppy in one cluster,
         car/bus/train in another, apple/orange in another).

Why a 2-D map when the vectors are 384-D?
    You can't picture 384 dimensions. PCA is a standard trick that "squeezes"
    the 384 numbers down to just 2 (an x and a y) while keeping the overall
    shape as much as possible. The 2-D picture is an approximation — but the
    clustering you see is real.

No matplotlib needed: we write a plain .svg file you can open in any browser
or in VS Code. Everything here uses packages already in requirements.txt.

Run (from the project root, with the venv active):
    python examples/phase1/visualize.py

Outputs (written next to this script):
    kitten_vector.txt   the 384 numbers for the word "kitten", one per line
    vectors_map.svg     the words drawn as colored dots on a 2-D map
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Same 8 words as example.py, so the two demos line up.
words = ["cat", "dog", "puppy", "apple", "orange", "car", "bus", "train"]

# Where to write the output files (this script's own folder).
HERE = os.path.dirname(os.path.abspath(__file__))

print("Loading the embedding model (first time downloads it)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Turn every word into its vector (384 numbers). normalize=True gives each
# vector length 1, which is what makes cosine-similarity comparisons clean.
vectors = model.encode(words, normalize_embeddings=True)  # shape: (8, 384)

# ---- 1. Print + save one real vector so you can look at the numbers -------
kitten = model.encode("kitten", normalize_embeddings=True)
print(f'\nThe word "kitten" as a vector: {kitten.shape[0]} numbers.')
print("First 12 of them:")
print(np.round(kitten[:12], 4))

vector_file = os.path.join(HERE, "kitten_vector.txt")
np.savetxt(vector_file, kitten, fmt="%.6f",
           header="'kitten' embedding: 384 numbers, one per line")
print(f"Saved all 384 numbers to {os.path.basename(vector_file)}")

# ---- 2. Squeeze 384-D -> 2-D and draw the words as dots -------------------
xy = PCA(n_components=2).fit_transform(vectors)

# Rescale the points into a nice drawing area.
xy = xy - xy.min(axis=0)
xy = xy / xy.max(axis=0)
W, H, PAD = 640, 460, 60
sx = PAD + xy[:, 0] * (W - 2 * PAD)
sy = PAD + (1 - xy[:, 1]) * (H - 2 * PAD)  # flip y: SVG's y grows downward

# Color by rough category so the clusters are obvious at a glance.
color = {"cat": "#e05a5a", "dog": "#e05a5a", "puppy": "#e05a5a",
         "apple": "#3fa34d", "orange": "#3fa34d",
         "car": "#3b7dd8", "bus": "#3b7dd8", "train": "#3b7dd8"}

marks = []
for x, y, w in zip(sx, sy, words):
    marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color[w]}"/>')
    marks.append(f'<text x="{x + 11:.1f}" y="{y + 4:.1f}" '
                 f'font-family="sans-serif" font-size="15" fill="#222">{w}</text>')

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">\n'
    f'<rect width="{W}" height="{H}" fill="#fafafa"/>\n'
    f'<text x="{W // 2}" y="30" text-anchor="middle" font-family="sans-serif" '
    f'font-size="17" fill="#111">Words as vectors (384-D squeezed to 2-D) — '
    f'closer = more similar meaning</text>\n'
    + "\n".join(marks) + "\n</svg>\n"
)

svg_file = os.path.join(HERE, "vectors_map.svg")
with open(svg_file, "w") as f:
    f.write(svg)

print(f"Saved the 2-D map to {os.path.basename(svg_file)} — open it in a browser.")
print("  red = animals · green = fruit · blue = vehicles")
print("\nTakeaway: the numbers aren't random. Words with similar meaning get "
      "similar\nvectors, so they land near each other on the map. That closeness "
      "is exactly\nwhat semantic search measures.")
