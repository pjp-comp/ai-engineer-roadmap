# RAG Architecture Types (the big map)

The phases in [learning_notes.md](learning_notes.md) build *one* RAG loop and then
improve it. But "RAG" isn't a single design — it's a family of architectures.
This file is the map: each type, what changes, and which phase covers it.

> **One-line way to hold it all:**
> **Naive RAG** is the loop · **Advanced/Hybrid** makes each step better ·
> **Agentic** turns the loop into a decision process · **Graph** and
> **Vectorless** change *what* you retrieve over · **Multi-modal** changes the
> *type* of data · **Corrective (CRAG)** adds a safety net when retrieval is bad.

---

## 1. Naive / Standard RAG
The baseline loop: chunk docs → embed → store → at query time embed the question,
retrieve top-k nearest chunks, stuff them into the prompt, generate.
- **Here:** [Phase 4](learning_notes.md#deep-dive-phase-4--building-real-rag)
  (`examples/phase4/example_4b_rag.py`, `examples/pdf-rag/`).
- **Good for:** getting started, FAQ bots, moderate doc sets.
- **Weak spots:** arbitrary chunking, "similarity ≠ relevance," no quality control.

## 2. Advanced / Hybrid RAG
The same loop with retrieval-quality upgrades layered around it:
- *Pre-retrieval:* query transformation (multi-query, HyDE).
- *Retrieval:* hybrid search (keyword/BM25 + vector) for exact terms **and** meaning.
- *Post-retrieval:* re-ranking with a cross-encoder; metadata filtering.
- **Here:** [Phase 5](learning_notes.md#deep-dive-phase-5--better-retrieval-quality).
- **Good for:** production systems where naive retrieval misses exact terms or ordering.

## 3. Agentic RAG
Retrieval becomes a **decision loop**: search → judge "is this enough?" → rewrite
the query / pick another source → retry → answer. Can route across multiple
indexes and tools.
- **Here:** [Phase 6](learning_notes.md#deep-dive-phase-6--advanced--agentic-retrieval).
- **Good for:** ambiguous questions, multi-step research, multiple data sources.
- **Cost:** more LLM calls, higher latency.

## 4. Corrective / Self-RAG (CRAG)
A structured cousin of agentic RAG: the model **grades** the retrieved chunks
before trusting them. If they're weak, it **falls back** (e.g. rewrite + retry, or
a different source) instead of answering from bad context.
- **Here:** [Phase 6b demo](examples/phase6b-crag/example.py).
- **Good for:** reducing hallucinations when the knowledge base has gaps.

## 5. Graph RAG
Builds a **knowledge graph** (entities + relationships) and retrieves by
traversing connections, not just vector similarity. Often combined with vectors.
- **Here:** named in [Phase 6](learning_notes.md#deep-dive-phase-6--advanced--agentic-retrieval).
- **Good for:** multi-hop questions where *relationships* carry the answer.

## 6. Vectorless / Reasoning-based RAG (PageIndex)
No embeddings, no vector DB. Build a table-of-contents tree; the LLM **reasons**
about which section to open, then reads it with context intact.
- **Here:** [Phase 7](learning_notes.md#phase-7--latest-market-direction-vectorless-rag-with-pageindex).
- **Good for:** long, well-structured docs (manuals, contracts, filings).
- **Trade-off:** weaker on huge, flat, unstructured collections.

## 7. Multi-modal RAG
The same retrieve-then-generate loop over images/audio/code, via models like CLIP
that embed several modalities into one shared space (text query → image results).
- **Here:** [image-embeddings deep dive](learning_notes.md#deep-dive-image-embeddings-searching-pictures).

## Worth knowing: long-context "RAG-less"
When the whole corpus fits, skip retrieval and paste everything into a large
context window. Simpler, but more expensive per call and weaker on large or
changing data — see the long-context note in
[Phase 6](learning_notes.md#deep-dive-phase-6--advanced--agentic-retrieval).

---

## Quick comparison

| Type | What changes | Needs vectors? | Best for |
|---|---|---|---|
| Naive | the basic loop | yes | getting started, FAQs |
| Advanced/Hybrid | better each step | yes | production retrieval quality |
| Agentic | loop → decisions | yes | ambiguous, multi-step questions |
| Corrective (CRAG) | grade + fallback | yes | knowledge bases with gaps |
| Graph | retrieve over relationships | optional | multi-hop, relational questions |
| Vectorless | reason over structure | no | long, structured documents |
| Multi-modal | non-text data | yes | image/audio/code search |

---

## Catalog: individual RAG techniques

The 7 types above are the *shapes* of a RAG system. Within them sit dozens of
specific **techniques** you can mix and match. The catalog below is organized by
the stage of the pipeline each technique improves.

> **Source & attribution:** this catalog follows the technique list and category
> structure of Nir Diamant's open-source
> [**RAG_Techniques**](https://github.com/NirDiamant/RAG_Techniques) repository
> (each technique there has a runnable notebook), whose visual companion is the
> book *RAG Made Simple: The Complete Visual Guide to Retrieval-Augmented
> Generation*. Names below are taken from that repo; the one-line descriptions are
> summaries. Go there (or to the book) for the authoritative, in-depth version.
> Where a technique already has a demo in **this** repo, it's cross-linked.

### 🌱 Foundational — getting a basic pipeline working
- **Basic RAG** — the core chunk → embed → retrieve → generate loop.
  *(this repo: [Phase 4](learning_notes.md#deep-dive-phase-4--building-real-rag))*
- **RAG over CSV / JSON** — semantic search across tabular or structured records.
- **Reliable RAG** — add validation/grounding checks so answers stick to sources.
- **Optimizing chunk sizes** — tune fixed-size segmentation for the content.
  *(this repo: [Phase 4 chunking](learning_notes.md#deep-dive-phase-4--building-real-rag))*
- **Proposition chunking** — split into standalone factual statements, not blind slices.

### 🔍 Query enhancement — fix the question before searching
- **Query transformations** — rewrite, step-back, or decompose the query.
- **HyDE (Hypothetical Document Embedding)** — draft a fake answer, search with *that*.
- **HyPE (Hypothetical Prompt Embeddings)** — precompute likely questions to match against.
  *(this repo touches query transforms in [Phase 5](learning_notes.md#deep-dive-phase-5--better-retrieval-quality))*

### 📚 Context & content enrichment — give each chunk more meaning
- **Contextual chunk headers** — prepend the document/section title to each chunk.
- **Relevant segment extraction** — stitch several chunks into one coherent passage.
- **Context window enhancement** — also pull the neighbouring sentences around a hit.
- **Semantic chunking** — split on topic boundaries, not character counts.
- **Contextual compression** — condense retrieved text down to just the relevant bits.
- **Document augmentation** — generate questions per chunk to make it more findable.

### 🚀 Advanced retrieval — get better candidates back
- **Fusion retrieval** — hybrid keyword + vector search.
  *(this repo: hybrid search in [Phase 5](learning_notes.md#deep-dive-phase-5--better-retrieval-quality))*
- **Intelligent reranking** — an LLM/cross-encoder re-scores the shortlist.
  *(this repo: re-ranking in [Phase 5](learning_notes.md#deep-dive-phase-5--better-retrieval-quality))*
- **Multi-faceted filtering** — filter by metadata, score threshold, content, diversity.
- **Hierarchical indices** — a summary tier over a detail tier for two-stage search.
- **Dartboard retrieval** — balance relevance *and* diversity in the results.
- **Multi-modal retrieval** — bring images in via captioning or vision LLMs.
  *(this repo: [image embeddings](learning_notes.md#deep-dive-image-embeddings-searching-pictures))*

### 🔁 Iterative & adaptive — change strategy on the fly
- **Retrieval with feedback loops** — learn from user interaction over time.
- **Adaptive retrieval** — pick a retrieval strategy based on the query type.

### 📊 Evaluation — measure quality instead of guessing
- **DeepEval** — score correctness, faithfulness, contextual relevancy.
- **GroUSE** — grounded LLM evaluation.
- **End-to-end RAG evaluation** — assess the whole pipeline.
- **Open-RAG-Eval** — UMBRELA scoring and hallucination detection.

### 🧠 Memory-augmented
- **MemoRAG** — key/value memory extraction with surrogate queries.

### 🔬 Explainability
- **Explainable retrieval** — surface *why* each passage was retrieved.

### 🏗️ Advanced architectures — the "shapes" from above, in depth
- **Agentic RAG** — the retrieve → judge → retry decision loop.
  *(this repo: [Phase 6](learning_notes.md#deep-dive-phase-6--advanced--agentic-retrieval))*
- **Self-RAG** — the model decides *when* to retrieve and whether to trust results.
- **Corrective RAG (CRAG)** — grade retrieval, correct or fall back.
  *(this repo: [Phase 6b demo](examples/phase6b-crag/example.py))*
- **Graph RAG** — retrieve over an entity/relationship knowledge graph (variants:
  Milvus-backed, Microsoft GraphRAG's community extraction, local + attribution).
- **RAPTOR** — recursive, tree-organized summarization for multi-level retrieval.
- **Sophisticated controllable agent** — a deterministic graph-based agent for hard,
  multi-step questions.

> **How this maps back:** most "foundational / query / enrichment / retrieval /
> evaluation" items are *techniques inside* the **Advanced/Hybrid** type; the
> "advanced architectures" are the **Agentic**, **CRAG**, **Graph**, and
> **Multi-modal** types seen in more depth.

## References
- Nir Diamant — [RAG_Techniques (GitHub)](https://github.com/NirDiamant/RAG_Techniques) — runnable notebook per technique.
- Nir Diamant — *RAG Made Simple: The Complete Visual Guide to Retrieval-Augmented Generation* (Super AI Engineering Series) — the book's visual companion to the repo.
