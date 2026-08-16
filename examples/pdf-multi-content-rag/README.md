# Multimodal RAG over your files — text + tables + images

**Big idea:** real documents aren't just text — they have **paragraphs, tables,
and images/charts**, and they come as **PDFs, pictures, and notes**. This example
builds one vector database over all of them and retrieves across them with a
single query, telling you **which file** each answer came from.

## What it does
1. **Reads every supported file** you drop in `assets/` (see below).
2. **Extracts content by type:**
   - **text** — prose (from PDFs and `.txt`/`.md` files), split into overlapping chunks.
   - **tables** — pulled from PDFs with **pdfplumber** (real table structure),
     cleaned, and linearized to readable rows.
   - **images** — from PDFs (banner/rule decoration is filtered out so only real
     figures remain) and standalone image files. Cropped PDF images are saved to
     `extracted_images/`.
3. **Embeds everything with CLIP** — one model that maps *both* text and images
   into the **same** vector space, so a text query can match any content type.
4. **Builds a FAISS vector DB**, storing each vector with metadata: its `type`
   (text / table / image), its `source` file, and its `page`.
5. **Retrieves** by query and prints nicely-formatted results: the matched
   **type**, similarity **score**, **source file**, and **page**.

## Your files go in `assets/`
Drop any mix of these into **`examples/pdf-multi-content-rag/assets/`**:
- **PDFs** (`.pdf`) — text, tables, and images are all extracted
- **Images** (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`)
- **Text** (`.txt`, `.md`)

By default the demo indexes **every supported file in `assets/`** into one vector
DB. The folder ships empty (your files are git-ignored), so add at least one file
before running — or pass `--pdf path/to/file.pdf` to index a single file.

## Run
```bash
# from the project root, with the venv active (see top-level README)

# 1) index everything in assets/ (the default)
python examples/pdf-multi-content-rag/example.py

# 2) ask your own questions across all assets
python examples/pdf-multi-content-rag/example.py \
    -q "who founded the company?" -q "quarterly revenue growth"

# 3) point at a different folder of files
python examples/pdf-multi-content-rag/example.py --assets path/to/folder

# 4) index a single PDF instead of a folder
python examples/pdf-multi-content-rag/example.py --pdf path/to/your.pdf
```
First run downloads the CLIP model (~600 MB) into the project's `.models/` cache.

### Run with [uv](https://docs.astral.sh/uv/) (faster, no manual venv)
[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager. From the
**project root**, one-time setup:
```bash
uv venv                                  # create .venv (uses your Python)
uv pip install -r requirements.txt       # install deps into it
```
Then either activate the venv (`source .venv/bin/activate`) and run `python …`
as above, **or** run without activating using `uv run`:
```bash
uv run python examples/pdf-multi-content-rag/example.py
uv run python examples/pdf-multi-content-rag/example.py -q "revenue by region"
```
`uv run` uses the project's `.venv` automatically, so you don't have to activate it.

## Embeddings: fresh, reused, and cleaning up
Building embeddings is the slow part, so they're **saved to `cache/`** and reused:

- **First run** (or after you add/edit/remove a file in `assets/`) — embeddings are
  **generated fresh** and saved. The demo fingerprints the files, so it notices
  changes automatically:
  ```
  Generating fresh embeddings from assets/ ...
  Saved embeddings to cache/ for next time.
  ```
- **Next run, files unchanged** — the saved embeddings are **reused** (fast, no
  re-embedding):
  ```
  Using saved embeddings (assets unchanged). Pass --rebuild to force a refresh.
  ```
- **Force fresh embeddings** without changing files:
  ```bash
  python examples/pdf-multi-content-rag/example.py --rebuild
  ```
- **Clean up** — delete the saved embeddings **and** the extracted images:
  ```bash
  python examples/pdf-multi-content-rag/example.py --clean
  ```

The `cache/` and `extracted_images/` folders are git-ignored (they're regenerated).

## What you'll see
Each result is labeled with its **type**, **score**, **source file**, and **page**,
so retrieval is fully traceable — even across many files. For example, with a PDF,
an image, and a notes file in `assets/`:

```
Reading everything in assets/ ...
  · logo.png: 1 items
  · notes.txt: 1 items
  · report.pdf: 4 items
Extracted: 2 image, 1 table, 3 text
Stored 6 items in a FAISS vector DB (text + tables + images, one shared space).

🔎 Query: "who founded the company?"
1. 📝 TEXT   (score 0.709, notes.txt, page 1)
       Company overview: Acme Corp ... founded in 2015 ...
```

## Code structure (read in this order)
The logic is split into small, single-purpose files so each step is easy to follow:

| File | Responsibility |
|---|---|
| [`chunking.py`](chunking.py) | split a page's text into overlapping chunks |
| [`extraction.py`](extraction.py) | pull text / tables / images out of a PDF, image, or text file |
| [`assets.py`](assets.py) | read every supported file from the `assets/` folder |
| [`embeddings.py`](embeddings.py) | turn items **and** queries into CLIP vectors |
| [`vector_store.py`](vector_store.py) | hold the vectors + find nearest ones (FAISS), save/load |
| [`store_cache.py`](store_cache.py) | save/reuse embeddings; fingerprint assets; `--clean` |
| [`retrieval.py`](retrieval.py) | run a query and print results nicely |
| [`pipeline.py`](pipeline.py) | ties it together: `MultimodalRAG.from_assets(...).search(...)` |
| [`example.py`](example.py) | thin entry point: CLI flags + demo queries |

The flow:
```
files ──extraction──▶ items ──embeddings──▶ vectors ──vector_store──▶ index
                                                                        │
                      query ──embeddings──▶ q_vector ──────search───────┘──▶ retrieval
```

## The "latest way" — and honest caveats
- **Shared embedding space (CLIP) + metadata** is the modern multimodal-RAG
  approach: one query retrieves across modalities and files.
- **Images use "describe-then-embed"**: store the picture *and* a short caption,
  embed both, average them. CLIP alone scores charts too low against text
  queries; the caption makes them findable. In production the caption comes from
  a vision/captioning model — here it's derived locally to stay API-key-free.
- **CLIP truncates long text** (~77 tokens), so heavy `.txt` matching is
  approximate. The right *source* is still retrieved and labeled; for text-only
  corpora the `all-MiniLM` text model retrieves more precisely.
- **Not every file has figures.** A text-heavy report may have only decorative
  banners (which we filter out), so "find me a chart" correctly returns
  text/tables — the real answers.
- **This is retrieval, not generation.** It returns the relevant pieces; wiring
  the top results into a real LLM call (with "answer only from this context") is
  the next step — see [classic PDF RAG](../pdf-rag/) for where that goes.

## Words to know
- **Multimodal embedding** — one model, one vector space, several data types.
- **CLIP** — the model that shares a space for text and images.
- **Describe-then-embed** — caption an image, embed the caption too, so text
  queries can find it.
- **Type / source / page metadata** — stored beside each vector so you know what
  matched and where.

→ Related: [image-embeddings demo](../image-embeddings/) ·
[classic PDF RAG](../pdf-rag/) ·
[full theory](../../learning_notes.md#deep-dive-image-embeddings-searching-pictures)
