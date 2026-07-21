"""
PHASE 6 — Agentic retrieval: search, judge, refine, search again.

Plain RAG searches once. An AGENT can loop:
    search -> "is this good enough?" -> if not, rewrite the query -> search again.

This mimics how a person researches: you try a search, see it's off, rephrase,
and try again until you find what you need.

About the "agent brain":
    A real agent uses an LLM to decide "good enough?" and to rewrite the query.
    To run with NO API key, we fake that decision with a simple rule (score
    threshold) and a fixed rewrite. The retrieval loop is the real lesson.

Run:
    python examples/phase6/example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model

from sentence_transformers.util import cos_sim

model = load_model()

knowledge = [
    "The annual subscription can be refunded within 30 days of purchase.",
    "Monthly subscriptions are billed each month and are non-refundable.",
    "You can upgrade or downgrade your plan at any time.",
    "Gift cards are non-refundable once redeemed.",
]
kb_vectors = model.encode(knowledge, normalize_embeddings=True)

GOOD_ENOUGH = 0.55  # if best score is below this, the agent tries again


def search(query):
    q_vec = model.encode(query, normalize_embeddings=True)
    scores = cos_sim(q_vec, kb_vectors)[0].tolist()
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return knowledge[best_idx], scores[best_idx]


def agent_rewrite(query):
    """FAKE 'agent brain'. A real agent would ask an LLM to rewrite the query."""
    # The agent notices the vague phrase 'money back' and replaces the whole
    # query with a clearer, more specific one about refunds.
    if "money back" in query:
        return "refund policy for subscriptions"
    return query


# A deliberately vague question that vector search handles poorly at first.
query = "can I get my money back"
print(f'\nStart question: "{query}"\n')

for attempt in range(1, 4):
    result, score = search(query)
    print(f"Attempt {attempt}: searched '{query}'")
    print(f"   best match (score {score:0.3f}): {result}")

    if score >= GOOD_ENOUGH:
        print("   -> good enough, stopping.\n")
        break

    new_query = agent_rewrite(query)
    if new_query == query:
        print("   -> can't improve the query further, stopping.\n")
        break
    print(f"   -> not confident; agent rewrites the query to '{new_query}'\n")
    query = new_query

print(f"Final answer: {result}")
print(
    "\nTakeaway: an agent doesn't accept a weak first result — it judges quality"
    "\nand retries with a better query. Real agents use an LLM for the judging and"
    "\nrewriting steps (and may pick which data source to search)."
)
