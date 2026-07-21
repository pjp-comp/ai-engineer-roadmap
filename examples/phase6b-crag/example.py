"""
PHASE 6b — Corrective RAG (CRAG): grade the retrieved chunks BEFORE trusting them.

Plain RAG retrieves the top chunk and answers from it — even when that chunk is
a bad match. CRAG adds a safety net:
    retrieve -> GRADE the result -> if weak, FALL BACK (rewrite + retry, or admit
    we don't know) instead of answering from bad context.

This is how you cut hallucinations when the knowledge base has gaps: the model
refuses to make something up from an irrelevant chunk.

About the "grader brain":
    A real CRAG system uses an LLM to score "is this chunk actually relevant?"
    To run with NO API key, we fake that grader with a cosine-score threshold.
    The grade -> correct -> answer loop is the real lesson.

Run:
    python examples/phase6b-crag/example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model

from sentence_transformers.util import cos_sim

model = load_model()

# A small knowledge base that deliberately has a GAP: nothing about shipping.
knowledge = [
    "The annual subscription can be refunded within 30 days of purchase.",
    "Monthly subscriptions are billed each month and are non-refundable.",
    "You can upgrade or downgrade your plan at any time.",
    "Gift cards are non-refundable once redeemed.",
]
kb_vectors = model.encode(knowledge, normalize_embeddings=True)

RELEVANT = 0.55  # at/above this, the chunk is trusted as relevant
AMBIGUOUS = 0.40  # between AMBIGUOUS and RELEVANT, try to correct; below, give up


def retrieve(query):
    q_vec = model.encode(query, normalize_embeddings=True)
    scores = cos_sim(q_vec, kb_vectors)[0].tolist()
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return knowledge[best_idx], scores[best_idx]


def grade(score):
    """FAKE 'grader brain'. A real CRAG system asks an LLM: is this relevant?"""
    if score >= RELEVANT:
        return "CORRECT"      # trust it, answer
    if score >= AMBIGUOUS:
        return "AMBIGUOUS"    # try to correct (rewrite the query)
    return "INCORRECT"        # too weak — fall back, don't hallucinate


def rewrite(query):
    """FAKE query correction. A real system uses an LLM to rephrase."""
    if "money back" in query:
        return "refund policy for subscriptions"
    return query


def answer(query):
    print(f'\nQuestion: "{query}"')
    for attempt in range(1, 3):
        chunk, score = retrieve(query)
        verdict = grade(score)
        print(f"  attempt {attempt}: best chunk (score {score:0.3f}) -> {verdict}")
        print(f"             {chunk}")

        if verdict == "CORRECT":
            return f"Answer: {chunk}"
        if verdict == "AMBIGUOUS":
            new_query = rewrite(query)
            if new_query != query:
                print(f"  -> grade weak; correcting query to '{new_query}'")
                query = new_query
                continue
        # INCORRECT, or no better rewrite available: refuse instead of guessing.
        return "Answer: I don't have reliable information on that. (Refusing to guess.)"
    return "Answer: I don't have reliable information on that. (Refusing to guess.)"


# Case A: a vague-but-answerable question -> grader rejects the weak first hit,
# correction rewrites it, second retrieval is relevant.
print(answer("can I get my money back"))

# Case B: a question the knowledge base simply can't answer (no shipping info) ->
# grader marks it INCORRECT and CRAG refuses rather than hallucinating.
print(answer("how long does international shipping take"))

print(
    "\nTakeaway: CRAG doesn't blindly answer from the top chunk. It GRADES the"
    "\nretrieval first, CORRECTS a weak query when it can, and REFUSES when the"
    "\nknowledge base has no good match — which is how you avoid confident wrong"
    "\nanswers. Real systems use an LLM for the grading and rewriting."
)
