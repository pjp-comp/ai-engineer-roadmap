"""
PHASE 5 — Better retrieval quality: hybrid search + re-ranking.

Two upgrades that make results noticeably better:

1. HYBRID SEARCH
   Vector search is great at meaning but can miss exact terms (codes, names).
   Keyword search nails exact terms but misses meaning. Combine both.

2. RE-RANKING
   First do a fast search to get a shortlist, then re-score that shortlist more
   carefully so the truly best result rises to the top.

We keep it simple and runnable:
   - vector score  = cosine similarity (real embeddings)
   - keyword score = how many query words appear in the text (simple overlap)
   - hybrid score  = a blend of the two
   - re-rank       = re-sort the shortlist by the hybrid score

Run:
    python examples/phase5/example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model

from sentence_transformers.util import cos_sim

model = load_model()

docs = [
    "Reset your password from the account settings page.",
    "Error code E-404 means the page was not found.",
    "To recover a lost login, use the forgot-password link.",
    "Our refund policy allows returns within 30 days.",
    "Contact support if you see error code E-500.",
]
doc_vectors = model.encode(docs, normalize_embeddings=True)


def keyword_score(query, text):
    """Simple overlap: fraction of query words that appear in the text."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


query = "error code E-404"
q_vec = model.encode(query, normalize_embeddings=True)
vector_scores = cos_sim(q_vec, doc_vectors)[0].tolist()

print(f'\nQuery: "{query}"\n')
print("Vector-only ranking (good at meaning, but 'E-404' is an exact term):")
for text, vscore in sorted(zip(docs, vector_scores), key=lambda p: p[1], reverse=True):
    print(f"  ({vscore:0.3f}) {text}")

# HYBRID: blend vector + keyword. Weight is a knob you tune (0.5 = equal).
weight = 0.5
hybrid = []
for text, vscore in zip(docs, vector_scores):
    kscore = keyword_score(query, text)
    blended = weight * vscore + (1 - weight) * kscore
    hybrid.append((text, vscore, kscore, blended))

print("\nHybrid ranking (vector + keyword) — exact term 'E-404' now wins:")
for text, v, k, b in sorted(hybrid, key=lambda p: p[3], reverse=True):
    print(f"  blended={b:0.3f}  (vector={v:0.3f}, keyword={k:0.3f})  {text}")

print(
    "\nTakeaway:"
    "\n- Hybrid search catches BOTH meaning and exact terms."
    "\n- Re-ranking = take a shortlist, then re-score it more carefully (done above"
    "\n  by re-sorting on the blended score). Real apps use a 'cross-encoder' model"
    "\n  as the re-ranker for even better ordering."
)
