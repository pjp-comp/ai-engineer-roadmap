"""
EMBEDDINGS — turn each item into a vector using CLIP.

CLIP is special: it embeds BOTH text and images into ONE shared space, so a
text query and a picture can be compared directly. That's what lets a single
search cover text, tables, and images at once.

Per item type:
    - text  -> embed the chunk text
    - table -> embed the flat "embed_text" (sentence-like form of the table)
    - image -> embed the PICTURE and its CAPTION, then average the two, so the
               item is findable both by how it looks and by words describing it
"""

import numpy as np

from shared import load_clip


def _unit(v):
    """Scale a vector to length 1 (so inner product == cosine similarity)."""
    return v / (np.linalg.norm(v) + 1e-9)


def embed_items(items, model=None):
    """Embed every item into one CLIP space. Returns a float32 matrix.

    Row i is the vector for items[i], so position ties a vector back to its
    item (and its type/page metadata).
    """
    model = model or load_clip()
    vectors = []
    for it in items:
        if it["type"] == "image":
            img_vec = model.encode(it["image"], normalize_embeddings=True)
            cap_vec = model.encode(it["caption"], normalize_embeddings=True)
            vectors.append(_unit(img_vec + cap_vec))  # picture + description
        else:
            text = it.get("embed_text", it["content"])
            vectors.append(model.encode(text, normalize_embeddings=True))
    return np.asarray(vectors, dtype="float32")


def embed_query(query, model=None):
    """Embed a text query into the same CLIP space as the items."""
    model = model or load_clip()
    return model.encode(query, normalize_embeddings=True).astype("float32")
