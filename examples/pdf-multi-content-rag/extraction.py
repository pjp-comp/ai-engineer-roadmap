"""
EXTRACTION — pull the three content types out of a PDF.

A real document mixes:
    - TEXT   : paragraphs of prose
    - TABLES : grids of numbers (extracted with pdfplumber's table finder)
    - IMAGES : charts / figures / graphics

This module turns a PDF into a flat list of "items", each a dict:
    {"type": "text"|"table"|"image", "page": N, "content": ..., ...}

Everything downstream (embedding, storing, retrieving) works off that list.
"""

import io
import os

import pdfplumber
from PIL import Image

from chunking import chunk


def clean_table(rows):
    """Drop empty cells/rows and linearize a pdfplumber table to readable text.

    Returns (pretty, embed_text) or (None, None) if the table is effectively
    empty. `pretty` is for display; `embed_text` is a flat, sentence-like form
    that CLIP's text encoder matches against queries better.
    """
    cleaned = []
    for row in rows:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        cells = [c for c in cells if c]  # drop empty cells
        if cells:
            cleaned.append(cells)
    if len(cleaned) < 2:  # need a header + at least one row to be a real table
        return None, None
    pretty = "\n".join("  |  ".join(r) for r in cleaned)
    header = cleaned[0]
    body_desc = "; ".join(", ".join(r) for r in cleaned[1:])
    embed_text = f"Table with columns {', '.join(header)}. Data: {body_desc}"
    return pretty, embed_text[:1200]  # keep it bounded


def crop_image(page, img):
    """Render just an image's region of the page to a PIL image.

    Returns None when it can't render, or when the region is too small (icons)
    or is a thin banner/rule strip rather than a real figure.
    """
    try:
        bbox = (max(img["x0"], 0), max(page.height - img["y1"], 0),
                min(img["x1"], page.width), min(page.height - img["y0"], page.height))
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < 40 or h < 40:
            return None  # skip icons / rules
        if w / h > 6 or h / w > 6:
            return None  # skip banner/rule strips (page decoration, not figures)
        cropped = page.crop(bbox).to_image(resolution=120)
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    except Exception:
        return None


# File extensions we know how to read.
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEXT_EXTS = {".txt", ".md"}
SUPPORTED_EXTS = PDF_EXTS | IMAGE_EXTS | TEXT_EXTS


def extract_pdf(pdf_path, image_dir, source):
    """Return items for text/table/image found in one PDF."""
    items = []
    os.makedirs(image_dir, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # -- tables first (a table's text is also caught by extract_text,
            #    but keeping both is fine for a teaching demo) --
            for t in (page.extract_tables() or []):
                pretty, embed_text = clean_table(t)
                if pretty:
                    items.append({"type": "table", "source": source,
                                  "page": page_num, "content": pretty,
                                  "embed_text": embed_text})

            # -- text: the page prose, split into chunks --
            for piece in chunk(page.extract_text() or ""):
                if len(piece.strip()) >= 25:  # skip tiny fragments
                    items.append({"type": "text", "source": source,
                                  "page": page_num, "content": piece.strip()})

            # -- images: crop, save, and remember a caption --
            for j, img in enumerate(page.images):
                pil = crop_image(page, img)
                if pil is None:
                    continue
                out = os.path.join(image_dir, f"{source}_page{page_num}_img{j}.png")
                pil.save(out)
                items.append({
                    "type": "image", "source": source, "page": page_num,
                    "content": f"[image] {os.path.basename(out)}  "
                               f"({pil.size[0]}x{pil.size[1]})",
                    "image": pil,
                    "caption": f"figure, chart, or graphic on page {page_num} "
                               f"of {source}",
                    "path": out,
                })
    return items


def extract_image_file(path, source):
    """Return one image item for a standalone picture file (.png/.jpg/...)."""
    try:
        pil = Image.open(path).convert("RGB")
    except Exception:
        return []
    return [{
        "type": "image", "source": source, "page": 1,
        "content": f"[image] {source}  ({pil.size[0]}x{pil.size[1]})",
        "image": pil,
        "caption": f"image file named {source}",
        "path": os.path.abspath(path),
    }]


def extract_text_file(path, source):
    """Return text items for a plain-text/markdown file, split into chunks."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return []
    items = []
    for piece in chunk(text):
        if len(piece.strip()) >= 25:
            items.append({"type": "text", "source": source, "page": 1,
                          "content": piece.strip()})
    return items


def extract_any(path, image_dir):
    """Extract items from any supported file, routing by extension.

    Every item is tagged with `source` = the file's name, so results can say
    which file they came from.
    """
    ext = os.path.splitext(path)[1].lower()
    source = os.path.basename(path)
    if ext in PDF_EXTS:
        return extract_pdf(path, image_dir, source)
    if ext in IMAGE_EXTS:
        return extract_image_file(path, source)
    if ext in TEXT_EXTS:
        return extract_text_file(path, source)
    return []  # unsupported type — skip


# Backwards-compatible alias (older code / --pdf path calls this).
def extract_items(pdf_path, image_dir):
    return extract_pdf(pdf_path, image_dir, os.path.basename(pdf_path))
