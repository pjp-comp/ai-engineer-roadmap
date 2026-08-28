"""
ZERO-TO-HERO · STEP 3 — Ask anything, get the meaningful content back.

This is SEMANTIC SEARCH. You type a question; we:
    1. turn your question into a vector (same model, same space as the chunks)
    2. compare it to every chunk's vector with cosine similarity
    3. return the closest chunks — ranked by MEANING, not keyword matching

Notice you can ask using DIFFERENT words than the text and still get the right
chunk back — because it matches on meaning. That's the whole point of embeddings.

We also draw where YOUR QUESTION lands among the chunks (query_map.svg).

Run (after step 1):
    python examples/embeddings-101/step3_search.py                 # demo questions
    python examples/embeddings-101/step3_search.py -q "your question here"
"""

import argparse
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


def search(question, vectors, chunks, model, k=3):
    q = model.encode(question, normalize_embeddings=True)
    scores = vectors @ q  # cosine (vectors are normalized)
    order = np.argsort(-scores)[:k]
    print(f'\n🔎 "{question}"')
    print("─" * 66)
    for rank, i in enumerate(order, 1):
        bar = "█" * int(scores[i] * 20)
        print(f"  {rank}. (score {scores[i]:.3f}) {bar}")
        print(f"     {chunks[i]}")
    return q, order


def draw_query_map(query_vec, order, vectors, chunks,
                   out=os.path.join(HERE, "query_map.svg")):
    """Plot all chunks + the query together, so you can SEE what it matched."""
    allv = np.vstack([vectors, query_vec])           # query is the last point
    xy = PCA(n_components=2).fit_transform(allv)
    xy = xy - xy.min(0)
    xy = xy / (xy.max(0) + 1e-9)
    W, H, PAD = 720, 500, 70
    px = PAD + xy[:, 0] * (W - 2 * PAD)
    py = PAD + (1 - xy[:, 1]) * (H - 2 * PAD)
    top = set(int(i) for i in order)
    marks = []
    for i in range(len(chunks)):
        color = "#3fa34d" if i in top else "#9aa5b1"  # green = retrieved
        marks.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="7" fill="{color}"/>')
        marks.append(f'<text x="{px[i]+10:.1f}" y="{py[i]+4:.1f}" '
                     f'font-family="sans-serif" font-size="13" fill="#333">'
                     f'C{i+1}</text>')
    # the query, as a red star-ish marker
    qx, qy = px[-1], py[-1]
    marks.append(f'<circle cx="{qx:.1f}" cy="{qy:.1f}" r="9" fill="#e0342b"/>')
    marks.append(f'<text x="{qx+12:.1f}" y="{qy+4:.1f}" font-family="sans-serif" '
                 f'font-size="14" fill="#e0342b">YOUR QUESTION</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#fafafa"/>'
           f'<text x="{W//2}" y="34" text-anchor="middle" font-family="sans-serif" '
           f'font-size="16" fill="#111">Red = your question. Green = retrieved '
           f'chunks (nearest). Grey = the rest.</text>'
           + "".join(marks) + "</svg>\n")
    with open(out, "w") as f:
        f.write(svg)
    print(f"\n🖼️  Saved {os.path.basename(out)} — the red dot is your question; the")
    print("   green dots (nearest) are what got retrieved. Open it in a browser.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", action="append", help="a question (repeatable)")
    ap.add_argument("-k", type=int, default=3, help="how many chunks to return")
    args = ap.parse_args()

    if not os.path.exists(VEC_PATH):
        raise SystemExit("No vectors found — run step1_generate.py first.")
    vectors, chunks = load()
    model = load_model()

    # These demo questions use DIFFERENT words than the meeting minutes on
    # purpose, to show that matching is by meaning, not keywords.
    queries = args.query or [
        "How much money did the company make?",       # -> financial review chunk
        "How many people were hired?",                 # -> hiring/people chunk
        "What did they decide about the Berlin office?",  # -> decisions chunk
    ]
    last = None
    for qtext in queries:
        last = (search(qtext, vectors, chunks, model, k=args.k), qtext)

    # draw the map for the LAST query so the picture matches the final search
    (query_vec, order), _ = last
    draw_query_map(query_vec, order, vectors, chunks)

    print("\nTakeaway: your question never had to use the same words as the text —")
    print("it matched by MEANING. That's semantic search, the heart of RAG.")


if __name__ == "__main__":
    main()
