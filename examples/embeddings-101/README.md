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
*similar meaning → similar numbers* — three ways:

- **A. Similarity matrix** — every chunk scored against every other. Each chunk must
  be most similar to **itself** (≈1.00), and related topics should score higher
  than unrelated ones.
- **B. 2-D map** (`similarity_map.svg`) — the 384-D vectors squeezed to 2-D and
  drawn as dots. Related chunks land **near each other**. *Open it in a browser.*
- **C. Known-answer probe** — ask a question you already know the answer to and
  confirm the right chunk ranks #1.

If all three look right, your embeddings work.

## Step 3 — Search by meaning
Type any question. It's turned into a vector and compared to every chunk; the
closest chunks come back, **ranked by meaning — not keywords**. You can ask using
completely different words than the text and still get the right answer.

It also draws `query_map.svg`: your **question** as a red dot among the chunks, with
the **retrieved** ones in green — so you can *see* what it matched.

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
