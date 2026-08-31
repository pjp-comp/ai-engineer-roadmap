# Embeddings 101 — from your own text file to semantic search

A complete, beginner-first lesson in **three runnable steps**. You bring a plain
text file; you finish understanding how AI turns it into searchable "meaning."

Everything runs **locally, no API key**. Each step **explains, demonstrates, and
draws a picture** so you can *see* what's happening.

## The 3 steps

| Step | File | What you learn |
|---|---|---|
| **1. Generate** | [`step1_generate.py`](step1_generate.py) | turn your text into embedding vectors |
| **2. Test** | [`step2_test.py`](step2_test.py) | prove the embeddings are actually good |
| **3. Search** | [`step3_search.py`](step3_search.py) | ask anything, get the meaningful content back |

Plus **[HOW_EMBEDDINGS_WORK.md](HOW_EMBEDDINGS_WORK.md)** — the theory you asked
for: the different ways to generate embeddings, which are reliable, and what
popular AI apps actually use (with sources).

## Run it (in order)
```bash
# from the project root, with the venv active (see the top-level README)
python examples/embeddings-101/step1_generate.py   # make the vectors
python examples/embeddings-101/step2_test.py       # check they're good (+ a picture)
python examples/embeddings-101/step3_search.py     # search by meaning (+ a picture)

# ask your own question:
python examples/embeddings-101/step3_search.py -q "your question here"
```

## Use YOUR text
Open [`my_text.txt`](my_text.txt) and replace it with your own content
(paragraphs separated by blank lines — each paragraph becomes one searchable
chunk). Then re-run step 1 → 2 → 3.

---

## Step 1 — Generate embeddings
Reads `my_text.txt`, splits it into paragraph chunks, and turns each chunk into a
**vector of 384 numbers** using a local AI model (`all-MiniLM-L6-v2`). Saves the
vectors so the next steps can use them.

> An embedding = a list of numbers that captures **meaning**. Similar meaning →
> similar numbers. That's the whole foundation.

**Curious how many ways there are to make embeddings, and what real apps use?**
Read [HOW_EMBEDDINGS_WORK.md](HOW_EMBEDDINGS_WORK.md).

## Step 2 — Test that the embeddings are good
You can't eyeball 384 numbers, so we verify the one promise embeddings make —
*similar meaning → similar numbers* — four ways:

- **A. Similarity matrix** — every chunk scored against every other. Each chunk must
  be most similar to **itself** (≈1.00), and related topics should score higher
  than unrelated ones.
- **B. 2-D map** (`similarity_map.svg`) — the 384-D vectors squeezed to 2-D and
  drawn as dots. Related chunks land **near each other**. *Open it in a browser.*
- **C. Known-answer probe** — ask a question you already know the answer to and
  confirm the right chunk ranks #1.
- **D. MRR & NDCG** — the two metrics real teams use to *measure* retrieval quality
  (not just eyeball it). See below.

### D. Measuring quality: MRR & NDCG
Instead of "looks about right", these put a **number** on how good retrieval is —
which is how you compare models or settings on *your* data. Both are 0–1, higher is
better:

- **MRR (Mean Reciprocal Rank)** — for each test question, score = `1 / (rank of the
  correct chunk)`. Right chunk at #1 → 1.0, at #2 → 0.5, at #3 → 0.33 …, then
  averaged over all questions.
- **NDCG@k** — rewards putting the correct chunk near the **top** of the first `k`
  results (with one correct answer, `1/log2(rank+1)`, so #1 = 1.0).

To compute either you need **labels**: questions paired with the chunk that should
answer them. The demo **auto-builds** a few labels from your document (a slice of a
chunk becomes a query whose answer is that chunk) so it runs on *any* text — a
near-1.0 score just confirms the pipeline works. For a **real** evaluation, replace
`build_labels()` in [`step2_test.py`](step2_test.py) with your own hand-written
questions; that harder test is exactly how teams pick an embedding model.

If all four look right, your embeddings work.

## Step 3 — Search by meaning
Type any question. It's turned into a vector and compared to every chunk; the
closest chunks come back, **ranked by meaning — not keywords**. You can ask using
completely different words than the text and still get the right answer.

It also draws `query_map.svg`: your **question** as a red dot among the chunks, with
the **retrieved** ones in green — so you can *see* what it matched.

### How `search()` actually works (the whole trick in 4 lines)
The search is simpler than it looks — the real logic in
[`step3_search.py`](step3_search.py) is just this:

```python
def search(question, vectors, chunks, model, k=3):
    q = model.encode(question, normalize_embeddings=True)  # 1. question → vector
    scores = vectors @ q                                   # 2. compare to every chunk
    order = np.argsort(-scores)[:k]                        # 3. rank, keep top k
    ...                                                    # 4. show chunks[i]
```

**1. Turn the question into a vector.** The chunks are numbers but your question is
text, so we embed the question with the **same model** used in Step 1 — that puts it
in the *same 384-dimensional space* as the chunks. `normalize_embeddings=True`
scales it to length 1 (needed for step 2 to be clean cosine similarity).

**2. Compare it to every chunk at once.** `vectors @ q` is one matrix-multiply:
`vectors` is `(15, 384)` and `q` is `(384,)`, so the result is `(15,)` — one score
per chunk. Each score is the **dot product** of that chunk's vector with the
question's, and because every vector has length 1, *dot product = cosine
similarity* (0 = unrelated, 1 = identical meaning).

```
        q (your question, 384 numbers)
vectors ┤ chunk 1  • ─ dot ─▶ 0.48
(15×384)┤ chunk 2  • ─ dot ─▶ 0.13
        ┤ chunk 3  • ─ dot ─▶ 0.51   ← highest = most relevant
        ┤   ...                ...
        └─▶ scores = [0.48, 0.13, 0.51, … ]
```

**3. Rank and keep the best.** `np.argsort(-scores)[:k]` sorts the scores
highest-first and takes the top `k` chunk indices.

**4. Show them.** For each winning index `i`, print `scores[i]` and the original
`chunks[i]` text.

> **One sentence:** embed the question into the same space as the chunks, take the
> dot product with every chunk (= cosine similarity, since all vectors are
> unit-length), sort highest-first, return the top `k`.

There's **no keyword matching anywhere** — a question about "growing" matches a
chunk about "77 percent growth rate" purely because their *meanings* land near each
other. Everything fancy in production RAG (FAISS, ANN indexes, re-ranking) just
makes step 2 *faster at scale* or the ranking *more accurate*; with 15 chunks the
plain `vectors @ q` is instant and exact.

---

## The sample document
`my_text.txt` is a realistic **company quarterly general meeting minutes** —
attendees, financial review, sales, hiring, decisions, action items, and more.
That variety makes the semantic search convincing. Swap in your own text anytime.

## What you'll see (real output)
```
🔎 "who runs the technology team?"
  1. (score 0.4xx) ████████
     Attendees: … Aisha Khan (VP of Engineering) …
```
The words "runs", "technology", and "team" appear **nowhere** in the text — it
matched "VP of Engineering" purely by **meaning**. That's semantic search, the
heart of RAG.

## Words to know
- **Embedding / vector** — the list of numbers representing a text's meaning.
- **Chunk** — one small piece of your document (here, one paragraph).
- **Cosine similarity** — a 0–1 score for how close two meanings are.
- **Semantic search** — finding by meaning instead of exact words.

→ Next in the repo: [phase2](../phase2/) (a real vector index) ·
[pdf-rag](../pdf-rag/) (RAG over a PDF) ·
[full theory](../../learning_notes.md)
