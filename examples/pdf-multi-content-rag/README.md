# Multimodal RAG over any PDF — text + tables + images

**Big idea:** real documents aren't just text — they have **paragraphs, tables,
and images/charts**. This example builds one vector database over all three, from
**any PDF**, and retrieves across them with a single query.

## Run
```bash
# from the project root, with the venv active (see top-level README)

# 1) the bundled sample (has text, a table, and a real chart image)
python examples/pdf-multi-content-rag/example.py

# 2) your own PDF
python examples/pdf-multi-content-rag/example.py --pdf path/to/your.pdf

# 3) your own PDF + your own questions
python examples/pdf-multi-content-rag/example.py --pdf your.pdf \
    -q "what were total revenues?" -q "revenue by region"
```
First run downloads the CLIP model (~600 MB) into the project's `.models/` cache.

## What it does
1. **Extracts** three content types from any PDF:
   - **text** — each page's prose, split into overlapping chunks.
   - **tables** — pulled with **pdfplumber** (real table structure), cleaned, and
     linearized to readable rows.
   - **images** — extracted from the PDF; banner/rule decoration is filtered out
     so only real figures remain. Saved to `extracted_images/`.
2. **Embeds everything with CLIP** — one model that maps *both* text and images
   into the **same** vector space, so a text query can match any content type.
3. **Builds a FAISS vector DB** storing the vector + the original content + a
   `type` (text / table / image) and `page` tag.
4. **Retrieves** by query and prints nicely-formatted results: the matched
   **type**, the similarity **score**, and the **page**.

## What you'll see
On the bundled sample, three queries each surface a different content type; on a
real report the answers naturally live in text and tables. Example on a real
12-page financial report:

```
Extracted: 8 table, 60 text
Stored 68 items in a FAISS vector DB ...

🔎 Query: "revenue by geographic area"
1. 📝 TEXT  (score 0.881, page 6)
       Revenue by Geographic Area Quarterly ... United States $34,036 ...
```

## The "latest way" — and honest caveats
- **Shared embedding space (CLIP) + type metadata** is the modern multimodal-RAG
  approach: one query retrieves across modalities.
- **Images use "describe-then-embed"**: store the picture *and* a short caption,
  embed both, average them. CLIP alone scores charts too low against text
  queries; the caption makes them findable. In production the caption comes from
  a vision/captioning model — here it's derived locally to stay API-key-free.
- **Not every PDF has figures.** A text-heavy financial report may have only
  decorative banners (which we filter out), so "find me a chart" correctly
  returns text/tables — the real answers. The tool adapts to the document.
- **This is retrieval, not generation.** It returns the relevant pieces; wiring
  the top results into a real LLM call (with "answer only from this context") is
  the next step — see [classic PDF RAG](../pdf-rag/) for where that goes.

## Words to know
- **Multimodal embedding** — one model, one vector space, several data types.
- **CLIP** — the model that shares a space for text and images.
- **Describe-then-embed** — caption an image, embed the caption too, so text
  queries can find it.
- **Type / page metadata** — stored beside each vector so you know what matched
  and where.

→ Related: [image-embeddings demo](../image-embeddings/) ·
[classic PDF RAG](../pdf-rag/) ·
[full theory](../../learning_notes.md#deep-dive-image-embeddings-searching-pictures)
