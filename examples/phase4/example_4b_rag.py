"""
PHASE 4b — RAG (Retrieval-Augmented Generation): the full loop.

The big pattern behind most AI products today:
    1. Split your documents into chunks (Phase 4a).
    2. Embed and store the chunks (Phases 1-3).
    3. When a question comes in, RETRIEVE the most relevant chunks.
    4. Feed those chunks to an LLM as context, and it ANSWERS using them.

About the "LLM" here:
    A real app would call an LLM (e.g. Claude) in step 4. To keep this runnable
    with NO API key, we use a tiny FAKE answer-builder that just stitches the
    retrieved chunks into a reply. The retrieval part is 100% real. Where the
    real LLM call goes is clearly marked below.

Run:
    python examples/phase4/example_4b_rag.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import load_model

from sentence_transformers.util import cos_sim

model = load_model()

# Our "knowledge base": a handful of facts (each one acts like a chunk).
knowledge = [
    "Our refund window for annual plans is 30 days.",
    "Monthly plans can be cancelled any time but are not refundable.",
    "Support is available 24/7 via chat and email.",
    "The free trial lasts 14 days and needs no credit card.",
    "Enterprise customers get a dedicated account manager.",
]

# Step 1-2: embed and "store" the knowledge (in memory for the demo).
kb_vectors = model.encode(knowledge, normalize_embeddings=True)


def retrieve(question, top_k=2):
    """Step 3: find the most relevant chunks for the question."""
    q_vec = model.encode(question, normalize_embeddings=True)
    scores = cos_sim(q_vec, kb_vectors)[0]
    ranked = sorted(zip(knowledge, scores.tolist()), key=lambda p: p[1], reverse=True)
    return ranked[:top_k]


def fake_llm_answer(question, context_chunks):
    """Step 4 (FAKE). A real app sends this prompt to an LLM instead.

    Real version would be roughly:
        client.messages.create(
            model="claude-...",
            messages=[{"role": "user", "content": prompt}],
        )
    Here we just echo the most relevant chunk so the demo needs no API key.
    """
    best = context_chunks[0][0]
    return f"Based on our docs: {best}"


question = "How long do I have to get a refund on a yearly subscription?"
print(f'\nQuestion: "{question}"\n')

retrieved = retrieve(question)
print("Retrieved context (most relevant chunks):")
for text, score in retrieved:
    print(f"  ({score:0.3f}) {text}")

answer = fake_llm_answer(question, retrieved)
print(f"\nAnswer: {answer}")

print(
    "\nNotice: the question said 'yearly subscription', the doc said 'annual plans'."
    "\nDifferent words, same meaning — retrieval found it. That is RAG's superpower."
    "\nSwap fake_llm_answer() for a real LLM call and you have a real RAG app."
)
