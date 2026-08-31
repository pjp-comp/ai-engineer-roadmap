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

    D. MRR & NDCG — the metrics real teams use. Give a few questions whose correct
       chunk you know, then measure HOW HIGH the right chunk ranks on average.
       MRR (Mean Reciprocal Rank) and NDCG@k are the standard retrieval scores.

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


# ---- D. MRR & NDCG (the metrics real teams use) ---------------------------
#
# To MEASURE retrieval quality you need a labeled set: (question -> the ONE chunk
# that should answer it). Real teams write these questions by hand. To keep this
# demo working on ANY document you paste in, we AUTO-BUILD the labels: for a few
# chunks we take a short slice of the chunk's OWN text as the "question", and the
# correct answer is that same chunk. (A hand-written paraphrase would be a harder,
# more realistic test — see the note printed below.)
#
# Two standard scores, both 0–1, higher = better:
#   • MRR  (Mean Reciprocal Rank): per question, score = 1 / (rank of the correct
#     chunk). Right chunk at #1 -> 1.0, at #2 -> 0.5, at #3 -> 0.33 … averaged.
#   • NDCG@k: rewards putting the correct chunk near the TOP of the first k hits.
#     With one correct answer: 1/log2(rank+1), so rank #1 = 1.0, and 0 past k.


def build_labels(chunks, n=5):
    """Auto-build (question, correct_chunk_id) pairs spread across the document."""
    if len(chunks) <= n:
        ids = list(range(len(chunks)))
    else:
        step = len(chunks) / n
        ids = sorted({int(i * step) for i in range(n)})
    labels = []
    for cid in ids:
        words = chunks[cid].split()
        # a middle slice of the chunk, so it's not just the opening boilerplate
        query = " ".join(words[3:16]) or chunks[cid]
        labels.append((query, cid))
    return labels


def retrieval_metrics(vectors, chunks, k=5):
    model = load_model()
    print("D. RETRIEVAL METRICS (MRR & NDCG) — how high does the right chunk rank?\n")

    labels = build_labels(chunks)
    rrs, ndcgs = [], []
    for question, correct in labels:
        q = model.encode(question, normalize_embeddings=True)
        order = np.argsort(-(vectors @ q))          # chunk indices, best first
        rank = int(np.where(order == correct)[0][0]) + 1  # 1-based rank of correct

        rr = 1.0 / rank                              # reciprocal rank
        ndcg = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0  # NDCG@k (1 relevant)
        rrs.append(rr)
        ndcgs.append(ndcg)
        print(f'   chunk {correct}: found at rank {rank}  (RR {rr:.2f}, NDCG {ndcg:.2f})')

    n = len(labels)
    print(f"\n   MRR      = {sum(rrs)/n:.3f}   (avg of 1/rank; 1.0 = always #1)")
    print(f"   NDCG@{k}  = {sum(ndcgs)/n:.3f}   (1.0 = right chunk always at top)")
    print("   Near 1.0 here just confirms the plumbing works (each query is a slice")
    print("   of its own chunk). For a REAL test, replace build_labels() with your")
    print("   own hand-written questions — that's how teams compare models on their")
    print("   data instead of guessing.\n")


def main():
    if not os.path.exists(VEC_PATH):
        raise SystemExit("No vectors found — run step1_generate.py first.")
    vectors, chunks = load()
    similarity_matrix(vectors, chunks)
    draw_map(vectors, chunks)
    known_answer_probe(vectors, chunks)
    retrieval_metrics(vectors, chunks)
    print("If the diagonal is ~1.00, related topics cluster on the map, the probe")
    print("found the right chunk, and MRR/NDCG are high — your embeddings are good!")


if __name__ == "__main__":
    main()
