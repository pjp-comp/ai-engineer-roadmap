"""
SHARED HELPERS — the few things every example needs, in one place.

Why this file exists:
    The model name, the chunking logic, and the PDF reader were copy-pasted into
    many examples. Keeping them here means you change them ONCE and every demo
    picks up the change.

What's here:
    MODEL_NAME      the embedding model every example uses
    load_model()    load (and cache) that model
    chunk_text()    split a long text into overlapping chunks
    read_pdf_text() pull the plain text out of a PDF

Where the model is stored:
    Model files are cached in the project's own `.models/` folder (set below via
    HF_HOME) so everything this project downloads stays inside the project.
    It is downloaded ONCE (~90 MB) and reused by every example after that.
    `.models/` is git-ignored.
"""

import os

# --- Keep downloaded model files inside the project ------------------------
# This must be set BEFORE importing sentence_transformers / huggingface libs,
# because they read this environment variable at import time.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE = os.path.join(PROJECT_ROOT, ".models")
os.environ.setdefault("HF_HOME", MODEL_CACHE)

from sentence_transformers import SentenceTransformer  # noqa: E402  (after HF_HOME)

# --- The one place the model names are written -----------------------------
# Text-only model: small, fast, 384-dimensional. Good default for semantic search.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Multimodal model: CLIP embeds BOTH text and images into ONE shared space, so a
# text query and a picture can be compared directly. Used by the multimodal demo.
CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

_model = None
_clip = None


def load_model(name=MODEL_NAME):
    """Load the text embedding model (downloads on first use, then cached).

    The loaded model is kept in memory, so calling this twice in one script
    does not reload it.
    """
    global _model
    if _model is None or name != MODEL_NAME:
        print(f"Loading the embedding model ({name})...")
        _model = SentenceTransformer(name)
    return _model


def load_clip(name=CLIP_MODEL_NAME):
    """Load the CLIP multimodal model (text + images in one vector space).

    Cached in memory like load_model(). First call downloads ~600 MB.
    """
    global _clip
    if _clip is None:
        print(f"Loading the multimodal (CLIP) model ({name})...")
        _clip = SentenceTransformer(name)
    return _clip


def chunk_text(text, size=200, overlap=40):
    """Split text into overlapping chunks so ideas aren't cut at a boundary.

    size    how many characters per chunk
    overlap how many characters neighbouring chunks share, so a sentence split
            across a boundary still appears whole in at least one chunk
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap  # step back by 'overlap' so chunks share edges
    return chunks


def read_pdf_text(path):
    """Return all the text from a PDF as one string."""
    from pypdf import PdfReader  # imported here so non-PDF demos don't need it

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
