# How embeddings are generated (the theory behind Step 1)

An **embedding** is a list of numbers (a *vector*) that captures the **meaning** of
a piece of text. Similar meaning → similar numbers. This file answers three
questions a beginner always asks:

1. What are the **different ways** to generate embeddings?
2. Which is a **reliable process** in the market?
3. What do **popular AI-powered applications** actually use?

---

## 1. The ways to generate embeddings (oldest → newest)

There are three broad families. Only the third is what modern RAG uses, but it
helps to know why the older ones fell short.

### A. Count-based / keyword (e.g. TF-IDF, Bag-of-Words)
- **Idea:** score each word by how often it appears (and how rare it is overall).
- **Strength:** simple, fast, great at *exact keyword* matching.
- **Weakness:** no understanding of meaning. "car" and "automobile" look totally
  unrelated. It can't tell *young cat* ≈ *kitten*.
- **Still used for:** keyword search and as the *keyword half* of **hybrid search**.

### B. Static word embeddings (word2vec, GloVe, FastText — ~2013–2017)
- **Idea:** learn one vector per word from lots of text, so related words are near.
- **Breakthrough:** captured meaning ("king − man + woman ≈ queen").
- **Weakness:** **one fixed vector per word, ignoring context.** The word "table"
  gets the same vector whether it means furniture or a spreadsheet. And there's no
  good vector for a whole *sentence*.

### C. Transformer / contextual sentence embeddings (2019 → today) ✅
- **Idea:** a transformer model (BERT-family) reads the **whole sentence at once**
  using *self-attention*, so each word's vector depends on its neighbours. Then the
  sentence is pooled into one vector.
- **This is what `sentence-transformers` (our Step 1) does**, and what every serious
  RAG system uses today. It's the only family that reliably captures *meaning of a
  whole passage*.

> **In one line:** TF-IDF matches *words*, word2vec matches *isolated words*,
> transformers match *meaning in context*. RAG needs the last one.

---

## 2. Is there a reliable process in the market?

Yes — the reliable pattern is: **use a strong pretrained transformer embedding
model, then measure it on your own data.** Two anchors:

- **The MTEB leaderboard** (Massive Text Embedding Benchmark) is the de-facto public
  ranking — it scores models across ~8 task types (retrieval, clustering, STS, …)
  on 50+ datasets. Great *starting point* for picking a model.
- **But benchmarks aren't the final word.** Some models are partly trained on data
  overlapping the benchmark, which inflates scores. The reliable process is to run
  a **small evaluation on YOUR corpus** with metrics like **MRR** and **NDCG** (or,
  for beginners, the "known-answer probe" in our Step 2: ask questions you know the
  answer to and check the right chunk ranks #1).

**Reliable process, in order:**
1. Start with a small, fast model (e.g. `all-MiniLM-L6-v2` — what this repo uses).
2. Build the pipeline and get it working end-to-end.
3. Evaluate on your own questions; only then swap in a bigger/better model if needed.
4. Consider **hybrid search** (keyword + vector) and **re-ranking** to boost quality.

---

## 3. What do popular AI applications use?

Two camps, both common in production:

### API embeddings (managed, easiest to ship)
You send text to a provider and get vectors back — no model to host.
- **OpenAI** `text-embedding-3-small` / `text-embedding-3-large` — extremely common
  default; strong quality, simple developer experience.
- **Cohere Embed**, **Voyage AI**, **Google Gemini Embedding** — other popular APIs.
- **Trade-off:** per-call cost, data leaves your machine, and if you change models
  you must **re-embed everything**.

### Open-source models (self-hosted, full control)
You run the model yourself (free, offline, private) — like this repo does.
- **BGE** family (e.g. **BGE-M3**) and **E5** (e.g. **E5-large-v2**, **E5-Mistral**)
  are the go-to open models for RAG; **Qwem/Qwen3-Embedding** and **Nomic Embed**
  are also widely used.
- **`all-MiniLM-L6-v2`** (ours) is the classic *lightweight* choice — small, fast,
  and good enough to learn on and to prototype real systems.
- **Trade-off:** you own deployment, batching, monitoring, and upgrades.

> **What most real RAG apps do:** start with an API model (OpenAI) *or* a strong
> open model (BGE-M3 / E5), keep the embedding model **the same** for documents and
> queries, and add hybrid search + re-ranking when quality matters. There's **no
> single "best"** — it depends on your language, latency, cost, privacy, and
> document length. Measure on your own data.

---

## How this maps to Step 1

Our `step1_generate.py` uses family **C** (a transformer sentence model,
`all-MiniLM-L6-v2`) via `sentence-transformers`. It's the reliable, beginner-
friendly choice: open-source, offline, 384-dimensional, and good enough that the
Step 2 tests clearly pass. To use an API model instead, you'd swap the
`model.encode(...)` call for the provider's embeddings endpoint — the rest of the
pipeline (store, search) stays identical.

---

### Sources
- [MTEB — Massive Text Embedding Benchmark (overview)](https://zeroentropy.dev/concepts/mteb/) ·
  [MTEB leaderboard notes](https://modal.com/blog/mteb-leaderboard-article)
- [Best embedding models for RAG in 2026 (StackAI)](https://www.stackai.com/insights/best-embedding-models-for-rag-in-2026-a-comparison-guide) ·
  [Embedding models comparison (openxcell)](https://www.openxcell.com/blog/best-embedding-models)
- [OpenAI text-embedding-3 vs open-source (Buzhou)](https://www.buzhou.io/en/articles/embedding-model-selection-guide-openai-text-embedding-3-vs-open-source-alternatives)
- [How to choose an embedding model (Qdrant)](https://qdrant.tech/articles/how-to-choose-an-embedding-model/) ·
  [TF-IDF vs embeddings (PyImageSearch)](https://pyimagesearch.com/2026/02/09/tf-idf-vs-embeddings-from-keywords-to-semantic-search/)

*Model names and rankings change fast — treat the above as a snapshot and check the
current MTEB leaderboard when choosing.*
