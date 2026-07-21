"""
PDF RAG — answer questions about a real PDF using embeddings (classic RAG).

This ties Phases 1-4 together on an actual file:
    1. Read the text out of a PDF.
    2. Split it into chunks.
    3. Embed the chunks and store them.
    4. For a question, retrieve the most relevant chunks.
    5. (Fake) LLM builds an answer from those chunks.

The LLM step is faked (no API key needed); the comment shows where a real LLM
call would go. Everything else is real.

Run (from project root, venv active):
    python examples/pdf-rag/example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers.util import cos_sim
from shared import chunk_text, load_model, read_pdf_text

HERE = os.path.dirname(__file__)
PDF_PATH = os.path.abspath(os.path.join(HERE, "..", "handbook.pdf"))


def ensure_pdf():
    if not os.path.exists(PDF_PATH):
        print("Sample PDF not found — creating it...")
        from make_sample_pdf import build_pdf  # local helper next to this file
        build_pdf()


def fake_llm_answer(question, chunks):
    """Where a REAL app would call an LLM, e.g.:
        client.messages.create(model="claude-...", messages=[...])
    Here we just return the top chunk so it runs with no API key.
    """
    return f"Based on the handbook: {chunks[0]}"


sys.path.insert(0, HERE)  # so 'import make_sample_pdf' works when run from root
ensure_pdf()

print("Reading the PDF and splitting it into chunks...")
text = read_pdf_text(PDF_PATH)          # from examples/shared.py
chunks = chunk_text(text)               # from examples/shared.py
print(f"Got {len(chunks)} chunks from the PDF.\n")

model = load_model()
chunk_vectors = model.encode(chunks, normalize_embeddings=True)

question = "How many days do I have to refund an annual plan?"
print(f'\nQuestion: "{question}"\n')

q_vec = model.encode(question, normalize_embeddings=True)
scores = cos_sim(q_vec, chunk_vectors)[0].tolist()
ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
top_chunks = [c for c, _ in ranked[:2]]

print("Retrieved chunks (most relevant pieces of the PDF):")
for chunk, score in ranked[:2]:
    print(f"  ({score:0.3f}) {chunk[:90]}...")

answer = fake_llm_answer(question, top_chunks)
print(f"\nAnswer: {answer}")

print(
    "\nThis is classic RAG over a PDF: chunk -> embed -> retrieve -> answer."
    "\nCompare with examples/pdf-vectorless/example.py, which answers the SAME"
    "\nPDF WITHOUT embeddings, by reasoning over its headings."
)
