# Vector Database — Learning Project

A hands-on project for learning how vector databases work, built up in phases
from a tiny in-memory demo to a real indexed search with FAISS.


## Why this matters in AI engineering

Normal databases match **exact words**; AI apps need to search by **meaning**.
A vector database stores the *meaning* of data as numbers (embeddings) and finds
the closest matches fast. This is what lets you build **RAG** (Retrieval-Augmented
Generation) — giving an LLM your own up-to-date data so it answers from facts
instead of hallucinating. It powers semantic search, recommendations, dedup, image
search, and agent memory.

> Mental model: **embed your data → store vectors → embed the question →
> retrieve nearest vectors → feed them to the LLM.**

Full breakdown of *why, where, and when* in [learning_notes.md](learning_notes.md#why-vector-databases-matter-in-ai-engineering).

---

## Setup

This project uses a virtual environment (`.venv`) and `requirements.txt`.

```bash
# 1. Go into the project folder
cd /Users/pragneshpatel/ai/embedding

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install dependencies (one time)
pip install -r requirements.txt
```

When the env is active your prompt shows `(.venv)`. To leave it later: `deactivate`.

> First run downloads the embedding model (~90 MB); after that it is cached and fast.

---

## Run the examples

There is one runnable example per phase in [examples/](examples/). Each is heavily
commented for beginners and prints what it's doing. With the env activated:

**Core path (one per phase):**
```bash
python examples/phase1/example.py              # the core idea, in memory
python examples/phase2/example.py              # real FAISS index + metadata + filtering
python examples/phase3/example.py              # ANN at scale (brute force vs HNSW)
python examples/phase4/example_4a_chunking.py  # splitting documents into chunks
python examples/phase4/example_4b_rag.py       # the full RAG loop
python examples/phase5/example.py              # hybrid search + re-ranking
python examples/phase6/example.py              # agentic (search, judge, retry)
python examples/phase6b-crag/example.py        # corrective RAG (grade, correct, refuse)
python examples/phase7/example.py              # vectorless RAG (PageIndex style)
```

**Bonus demos (apply the ideas to images and real PDFs):**
```bash
python examples/image-embeddings/example.py    # pictures as vectors + text-to-image (CLIP)
python examples/pdf-rag/example.py             # classic RAG over a real PDF
python examples/pdf-vectorless/example.py      # SAME PDF answered with NO embeddings
```

> Phases that would normally call an LLM (4b, 6, both PDF demos) use a small,
> clearly-labeled **fake LLM** so everything runs with no API key. The retrieval
> parts are real. Each example folder also has its own short `README.md`.
> Tip: run the two PDF demos back-to-back to compare embedding RAG vs. vectorless.

---

## File index

| Path | What it is |
|---|---|
| [README.md](README.md) | This index — setup, file map, and the learning roadmap |
| [learning_notes.md](learning_notes.md) | **The full guide** — start here; plain-language theory for every phase |
| [rag_architecture_types.md](rag_architecture_types.md) | One-page map of the RAG architecture types (naive, hybrid, agentic, CRAG, graph, vectorless, multi-modal) |
| [context_engineering.md](context_engineering.md) | How retrieval fits into the bigger job of assembling the LLM's prompt (context engineering) |
| [examples/](examples/) | Runnable, commented examples: one per phase (`phase1` … `phase7`) plus bonus `image-embeddings`, `pdf-rag`, `pdf-vectorless` |
| [requirements.txt](requirements.txt) | Python dependencies |

**New to this?** Read the "Start here" section of
**[learning_notes.md](learning_notes.md#start-here-read-this-first-if-youre-brand-new)**,
then run `examples/phase1/example.py`.

---

## Learning roadmap

A path from first principles to the techniques used in production AI systems today.
Each line has a one-line "what it is" so you know why it's on the list. Ask me to
expand any single item and I'll write the deep dive.

### Phase 1 — The core idea (in memory)
- **Embeddings** — turning text into a list of numbers (a vector) that captures meaning.
- **Similarity metrics** — cosine, dot product, euclidean; how "closeness" is measured.
- **Semantic search** — finding the nearest meaning instead of matching exact words.
- → details in [learning_notes.md](learning_notes.md#phase-1--the-core-idea-in-memory-demo)

### Phase 2 — Real vector databases
- **Metadata storage** — keeping the original text + tags (category, date) beside each vector.
- **A real index (FAISS)** — replacing the hand-written loop with a proper search index.
- **Metric comparison** — seeing cosine vs. euclidean behave on the same query.
- **Metadata filtering** — narrowing results by tags alongside similarity.
- → details in [learning_notes.md](learning_notes.md#phase-2--real-vector-databases)

### Phase 3 — Scaling the index (ANN)
- **Nearest-neighbour problem** — why scanning every vector breaks past millions of items.
- **Approximate Nearest Neighbour (ANN)** — trading a sliver of accuracy for huge speed.
  - **HNSW** — a layered graph: big jumps across far data, then fine-tuning.
  - **IVF** — cluster the vectors, then search only the relevant clusters.
  - **PQ (Product Quantization)** — compress vectors (e.g. 512 → 8 bytes) to fit memory.
- **Persistence** — saving and loading the index to disk so it survives restarts.

### Phase 4 — Building real RAG
- **Document loading & chunking** — splitting PDFs/docs into pieces worth embedding.
- **Chunking strategies** — fixed-size, overlap, sentence/semantic chunking and their trade-offs.
- **Retrieval-Augmented Generation (RAG)** — retrieving relevant chunks and feeding them to an LLM.
- **Production vector DBs** — Chroma (local), then Pinecone / Weaviate / Qdrant / pgvector (hosted).
- **Embedding model choice** — open models vs. API embeddings; dimensions, cost, quality.

### Phase 5 — Better retrieval quality
- **Hybrid search** — combining keyword (BM25) with vector search for the best of both.
- **Re-ranking** — a cross-encoder re-orders the top results so the best one lands first.
- **Query transformation** — rewriting / expanding the question (multi-query, HyDE) before search.
- **Evaluation** — measuring retrieval and answer quality (RAGAS-style metrics) instead of guessing.

### Phase 6 — Advanced & agentic retrieval
- **Multi-modal embeddings** — searching images, audio, and code with the same idea
  (see the [image-embeddings](examples/image-embeddings/) demo and the
  [deep dive](learning_notes.md#deep-dive-image-embeddings-searching-pictures)).
- **Knowledge graphs / GraphRAG** — using relationships between facts, not just similarity.
- **Agentic RAG** — an agent decides what to retrieve, refines, and retrieves again in a loop.
- **Long-context vs. retrieval** — when a huge context window replaces retrieval, and when it doesn't.

### Phase 7 — Latest market direction
- **Vectorless RAG (PageIndex)** — drop embeddings/vector DB; let the LLM *reason* over a
  document's structure (table-of-contents tree) to pick the right section. Aims to fix
  chunking and "similarity ≠ relevance" problems. Reference: *What Is PageIndex? How to
  Build a Vectorless RAG System (No Embeddings, No Vector DB).*
- **Reasoning-based retrieval** — the broader shift from "find similar" to "decide what's relevant".
- → details in [learning_notes.md](learning_notes.md#phase-7--latest-market-direction-vectorless-rag-with-pageindex)
