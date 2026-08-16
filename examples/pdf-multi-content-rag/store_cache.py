"""
STORE CACHE — persist embeddings so we don't rebuild them every run.

What gets saved (into cache/):
    index.faiss    the FAISS vector index
    items.pkl      the item metadata (type, source, page, content, caption, path)
    fingerprint    a hash of the assets/ files, so we know if they changed

Reuse logic ("auto-detect changes"):
    On run, we fingerprint assets/. If a cache exists with the SAME fingerprint,
    we load it (fast — no re-embedding). If the files changed (or --rebuild is
    passed), we re-embed and overwrite the cache.

Note: the PIL image objects are dropped before saving (they're only needed at
embedding time). The saved `path` still points at the cropped image on disk.
"""

import hashlib
import os
import pickle
import shutil

from vector_store import VectorStore

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
ITEMS_PATH = os.path.join(CACHE_DIR, "items.pkl")
FINGERPRINT_PATH = os.path.join(CACHE_DIR, "fingerprint")


def fingerprint(files):
    """A hash of the given files' (name, size, mtime). Changes when they change."""
    h = hashlib.sha256()
    for path in sorted(files):
        st = os.stat(path)
        h.update(os.path.basename(path).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def is_fresh(files):
    """True if a cache exists AND matches the current assets fingerprint."""
    if not (os.path.exists(INDEX_PATH) and os.path.exists(FINGERPRINT_PATH)):
        return False
    with open(FINGERPRINT_PATH) as f:
        saved = f.read().strip()
    return saved == fingerprint(files)


def save(store, items, files):
    """Persist the index, items (minus PIL images), and the fingerprint."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    store.save(INDEX_PATH)
    slim = [{k: v for k, v in it.items() if k != "image"} for it in items]
    with open(ITEMS_PATH, "wb") as f:
        pickle.dump(slim, f)
    with open(FINGERPRINT_PATH, "w") as f:
        f.write(fingerprint(files))


def load():
    """Load (store, items) from the cache. Assumes is_fresh() was true."""
    store = VectorStore.load(INDEX_PATH)
    with open(ITEMS_PATH, "rb") as f:
        items = pickle.load(f)
    return store, items


def clean():
    """Delete the cache AND the extracted-images folder. Returns what was removed."""
    removed = []
    for path in (CACHE_DIR, os.path.join(HERE, "extracted_images")):
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(os.path.basename(path) + "/")
    return removed
