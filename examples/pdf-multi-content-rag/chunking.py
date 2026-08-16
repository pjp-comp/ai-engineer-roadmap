"""
CHUNKING — split a long text into small overlapping pieces.

Why chunk at all?
    You can't usefully embed a whole page as one vector — it blurs every idea
    into one average. So we cut the text into small pieces and embed each one.

Why overlap?
    A sentence sitting on a chunk boundary would get cut in half. Letting
    neighbouring chunks share a few characters means that sentence still appears
    whole in at least one chunk.

Note on size:
    CLIP's text encoder truncates at ~77 tokens, so we keep chunks short.
"""

# CLIP truncates long text, so keep chunks small.
CHUNK_CHARS = 300
CHUNK_OVERLAP = 60


def chunk(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Split `text` into overlapping chunks of ~`size` characters."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap  # step back by `overlap` so chunks share edges
    return out
