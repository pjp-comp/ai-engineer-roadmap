"""
ZERO-TO-HERO · STEP 2 — Test whether the embeddings are actually good.

"How do I know the embeddings worked?" You can't read 384 numbers and tell. So we
check the ONE property embeddings promise: SIMILAR MEANING -> SIMILAR NUMBERS.
We verify that three ways:

    A. SIMILARITY MATRIX  — score every chunk against every other. A chunk must be
       most similar to ITSELF (score ~1.0), and related topics should score higher
       than unrelated ones.

    B. 2-D MAP (picture)  — squeeze the 384-D vectors down to 2-D and draw them.
       Related chunks should land NEAR each other. Saved as similarity_map.svg.

    C. KNOWN-ANSWER PROBE — ask a question whose answer you KNOW, and check the
       right chunk comes out on top. If it does, retrieval (step 3) will work.

Run (after step 1):
    python examples/embeddings-101/step2_test.py
"""

import os

import numpy as np
from sklearn.decomposition import PCA

from common import CHUNKS_PATH, HERE, VEC_PATH
from shared import load_model


def load():
    vectors = np.load(VEC_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = f.read().splitlines()
    return vectors, chunks


# ---- A. similarity matrix -------------------------------------------------
def similarity_matrix(vectors, chunks):
    # vectors are normalized, so dot product == cosine similarity
    sim = vectors @ vectors.T
    print("A. SIMILARITY MATRIX (1.00 = identical meaning, 0 = unrelated)\n")
    header = "        " + "  ".join(f"C{i+1}" for i in range(len(chunks)))
    print(header)
    for i, row in enumerate(sim):
        cells = "  ".join(f"{v:.2f}" for v in row)
        print(f"  C{i+1}  {cells}")
    # sanity checks
    diag_ok = all(abs(sim[i, i] - 1.0) < 1e-3 for i in range(len(chunks)))
    print(f"\n  ✓ every chunk is most similar to itself (diagonal ≈ 1.00): {diag_ok}")
    # show the most-related pair (excluding self)
    best, pair = -1, None
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if sim[i, j] > best:
                best, pair = sim[i, j], (i, j)
    print(f"  ➜ most-related pair: C{pair[0]+1} & C{pair[1]+1} (score {best:.2f})")
    print(f"      C{pair[0]+1}: {chunks[pair[0]][:60]}…")
    print(f"      C{pair[1]+1}: {chunks[pair[1]][:60]}…\n")


# ---- B. 2-D map (SVG picture) ---------------------------------------------
def draw_map(vectors, chunks, out=os.path.join(HERE, "similarity_map.svg")):
    xy = PCA(n_components=2).fit_transform(vectors)
    xy = xy - xy.min(0)
    xy = xy / (xy.max(0) + 1e-9)
    W, H, PAD = 720, 500, 70
    px = PAD + xy[:, 0] * (W - 2 * PAD)
    py = PAD + (1 - xy[:, 1]) * (H - 2 * PAD)
    dots = []
    for i, (x, y) in enumerate(zip(px, py)):
        label = f"C{i+1}"
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#3b7dd8"/>')
        dots.append(f'<text x="{x+11:.1f}" y="{y+4:.1f}" font-family="sans-serif" '
                    f'font-size="14" fill="#222">{label}: {chunks[i][:34]}…</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#fafafa"/>'
           f'<text x="{W//2}" y="34" text-anchor="middle" font-family="sans-serif" '
           f'font-size="16" fill="#111">Your chunks as points (384-D → 2-D). '
           f'Closer = more similar meaning.</text>'
           + "".join(dots) + "</svg>\n")
    with open(out, "w") as f:
        f.write(svg)
    print(f"B. 2-D MAP saved to {os.path.basename(out)} — open it in a browser.")
    print("   Chunks about related topics should sit closer together.\n")


# ---- C. known-answer probe ------------------------------------------------
def known_answer_probe(vectors, chunks):
    model = load_model()
    # We KNOW the minutes cover revenue/profit, so this should hit that chunk —
    # note the question says "earn", not "revenue", so it's a MEANING match.
    question = "How much did the company earn last quarter?"
    q = model.encode(question, normalize_embeddings=True)
    scores = vectors @ q
    best = int(np.argmax(scores))
    print("C. KNOWN-ANSWER PROBE")
    print(f'   Question: "{question}"')
    print(f"   Top chunk (score {scores[best]:.2f}): {chunks[best][:70]}…")
    hit = "revenue" in chunks[best].lower() or "profit" in chunks[best].lower()
    print(f"   ✓ retrieved the expected (financial) chunk: {hit}\n")


def main():
    if not os.path.exists(VEC_PATH):
        raise SystemExit("No vectors found — run step1_generate.py first.")
    vectors, chunks = load()
    similarity_matrix(vectors, chunks)
    draw_map(vectors, chunks)
    known_answer_probe(vectors, chunks)
    print("If the diagonal is ~1.00, related topics cluster on the map, and the")
    print("probe found the right chunk — your embeddings are good. On to step 3!")


if __name__ == "__main__":
    main()
