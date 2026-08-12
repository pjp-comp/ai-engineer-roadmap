"""
MULTIMODAL RAG over ANY PDF — text + tables + images in one vector DB.

Real documents mix content types: paragraphs, tables of numbers, and
charts/images. This builds ONE vector database over all three, from ANY PDF, and
retrieves across them with a single query.

How it works (the modern approach):
    - TEXT   : each page's prose is split into overlapping chunks.
    - TABLES : pulled out with pdfplumber (real table structure), linearized to
               readable rows, and embedded as text.
    - IMAGES : extracted from the PDF, embedded with CLIP, and paired with a
               short caption so they're findable by words ("describe-then-embed").
    Everything is embedded into ONE CLIP space (text and images share a space),
    stored in a FAISS vector DB with TYPE + PAGE metadata, and retrieved by
    cosine similarity — regardless of which content type the answer lives in.

Point it at any PDF with --pdf; defaults to the bundled sample.
No API key needed. CLIP runs locally.

Run (from project root, venv active):
    python examples/pdf-multi-content-rag/example.py
    python examples/pdf-multi-content-rag/example.py --pdf path/to/your.pdf
    python examples/pdf-multi-content-rag/example.py --pdf your.pdf -q "your question"
"""

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import pdfplumber
from PIL import Image

from shared import load_clip

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(HERE, "sample_document_with_chart.pdf")
IMG_DIR = os.path.join(HERE, "extracted_images")

# CLIP's text encoder truncates at 77 tokens, so keep text chunks short.
CHUNK_CHARS = 300
CHUNK_OVERLAP = 60


# --------------------------------------------------------------------------
# 1. EXTRACT text, tables, and images from ANY PDF
# --------------------------------------------------------------------------
def _chunk(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def _clean_table(rows):
    """Drop empty cells/rows and linearize a pdfplumber table to readable text.

    Returns (pretty, embed_text) or (None, None) if the table is effectively empty.
    """
    cleaned = []
    for row in rows:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        cells = [c for c in cells if c]  # drop empty cells
        if cells:
            cleaned.append(cells)
    if len(cleaned) < 2:  # need at least a header + one row to be a real table
        return None, None
    pretty = "\n".join("  |  ".join(r) for r in cleaned)
    # a flat, sentence-like version helps CLIP's text encoder match queries
    header = cleaned[0]
    body_desc = "; ".join(", ".join(r) for r in cleaned[1:])
    embed_text = f"Table with columns {', '.join(header)}. Data: {body_desc}"
    return pretty, embed_text[: 1200]  # keep it bounded


def extract_items(pdf_path):
    """Return a list of dicts: {type, page, content, ...} for text/table/image."""
    items = []
    os.makedirs(IMG_DIR, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # -- tables (do this first so we can subtract them from the text) --
            page_tables = page.extract_tables() or []
            for t in page_tables:
                pretty, embed_text = _clean_table(t)
                if pretty:
                    items.append({"type": "table", "page": page_num,
                                  "content": pretty, "embed_text": embed_text})

            # -- text: the page prose, split into chunks --
            text = page.extract_text() or ""
            for chunk in _chunk(text):
                if len(chunk.strip()) >= 25:  # skip tiny fragments
                    items.append({"type": "text", "page": page_num,
                                  "content": chunk.strip()})

            # -- images --
            for j, img in enumerate(page.images):
                pil = _crop_image(page, img)
                if pil is None:
                    continue
                out = os.path.join(IMG_DIR, f"page{page_num}_img{j}.png")
                pil.save(out)
                items.append({
                    "type": "image", "page": page_num,
                    "content": f"[image] {os.path.basename(out)}  ({pil.size[0]}x{pil.size[1]})",
                    "image": pil,
                    "caption": f"figure, chart, or graphic on page {page_num} of the document",
                    "path": out,
                })
    return items


def _crop_image(page, img):
    """Render just the image's region of the page to a PIL image (robust across
    PDF encodings). Returns None if it can't be rendered or is too tiny."""
    try:
        bbox = (max(img["x0"], 0), max(page.height - img["y1"], 0),
                min(img["x1"], page.width), min(page.height - img["y0"], page.height))
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < 40 or h < 40:
            return None  # skip icons/rules
        # skip banner/rule strips: very wide-and-short (or tall-and-thin) bands
        # that are page decoration, not real figures.
        if w / h > 6 or h / w > 6:
            return None
        cropped = page.crop(bbox).to_image(resolution=120)
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    except Exception:
        return None


# --------------------------------------------------------------------------
# 2 + 3. EMBED everything with CLIP and BUILD the FAISS vector DB
# --------------------------------------------------------------------------
def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def build_vector_db(items, model):
    """Embed each item into ONE CLIP space and store in a FAISS index."""
    vectors = []
    for it in items:
        if it["type"] == "image":
            img_vec = model.encode(it["image"], normalize_embeddings=True)
            cap_vec = model.encode(it["caption"], normalize_embeddings=True)
            vectors.append(_unit(img_vec + cap_vec))  # picture + description
        else:
            vectors.append(model.encode(it.get("embed_text", it["content"]),
                                        normalize_embeddings=True))
    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])  # inner product == cosine (normalized)
    index.add(matrix)
    return index


# --------------------------------------------------------------------------
# 4. RETRIEVE by query and show results nicely
# --------------------------------------------------------------------------
ICON = {"text": "📝 TEXT ", "table": "📊 TABLE", "image": "🖼️  IMAGE"}


def search(query, index, items, model, k=4):
    q_vec = model.encode(query, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(np.array([q_vec]), k)

    print(f'\n🔎 Query: "{query}"')
    print("─" * 72)
    for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), 1):
        if i < 0:
            continue
        it = items[i]
        print(f"{rank}. {ICON[it['type']]}  (score {score:0.3f}, page {it['page']})")
        body = it["content"]
        preview = body if it["type"] == "table" else (body[:220] + ("…" if len(body) > 220 else ""))
        for line in preview.splitlines():
            print(f"        {line}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Multimodal RAG over any PDF.")
    ap.add_argument("--pdf", default=DEFAULT_PDF, help="path to a PDF")
    ap.add_argument("-q", "--query", action="append",
                    help="a query (repeatable); omit to use built-in demo queries")
    ap.add_argument("-k", type=int, default=4, help="results per query")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"PDF not found: {args.pdf}")

    print(f"Reading {os.path.basename(args.pdf)} and extracting its content...")
    items = extract_items(args.pdf)
    counts = {}
    for it in items:
        counts[it["type"]] = counts.get(it["type"], 0) + 1
    print("Extracted:", ", ".join(f"{n} {t}" for t, n in sorted(counts.items())) or "nothing")
    if not items:
        sys.exit("No content extracted — is this a scanned/empty PDF?")

    model = load_clip()
    index = build_vector_db(items, model)
    print(f"Stored {index.ntotal} items in a FAISS vector DB "
          f"(text + tables + images, one shared space).")

    queries = args.query or [
        "What were the total revenues this quarter?",
        "revenue by geographic area",
        "a chart or figure in the report",
    ]
    for q in queries:
        search(q, index, items, model, k=args.k)

    print("Takeaway: one CLIP-powered vector DB retrieves across text, tables, and")
    print("images from ANY PDF. The TYPE + PAGE metadata tells you what matched and")
    print(f"where. Extracted images were saved to {os.path.basename(IMG_DIR)}/.")


if __name__ == "__main__":
    main()
