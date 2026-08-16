"""
PIPELINE — wire the pieces together into one object.

This is the whole multimodal RAG flow, using the focused modules:

    PDF ──extraction──▶ items ──embeddings──▶ vectors ──vector_store──▶ index
                                                                          │
                        query ──embeddings──▶ q_vector ──────search──────┘
                                                                          │
                                                            retrieval ◀───┘

Read the modules in this order to understand it:
    chunking.py → extraction.py → embeddings.py → vector_store.py → retrieval.py
"""

import os

import store_cache
from assets import list_asset_files, load_assets
from embeddings import embed_items, embed_query
from extraction import extract_items
from retrieval import print_results
from shared import load_clip
from vector_store import VectorStore

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "extracted_images")
ASSETS_DIR = os.path.join(HERE, "assets")


class MultimodalRAG:
    """Build a searchable multimodal index from files, then query it."""

    def __init__(self, items, store, model):
        self.items = items
        self.store = store
        self.model = model

    @classmethod
    def _from_items(cls, items, model):
        if not items:
            raise ValueError("No content extracted — are the files scanned/empty?")
        matrix = embed_items(items, model)
        return cls(items, VectorStore.from_matrix(matrix), model)

    @classmethod
    def from_pdf(cls, pdf_path, image_dir=IMG_DIR):
        """Extract → embed → store from a single PDF (not cached)."""
        model = load_clip()
        return cls._from_items(extract_items(pdf_path, image_dir), model)

    @classmethod
    def from_assets(cls, assets_dir=ASSETS_DIR, image_dir=IMG_DIR, rebuild=False):
        """Index every supported file in assets/, reusing a saved cache if the
        files haven't changed.

        rebuild=True forces fresh embeddings even if the cache is up to date.
        """
        files = list_asset_files(assets_dir)

        # Reuse the saved embeddings if they still match the current files.
        if not rebuild and store_cache.is_fresh(files):
            print("Using saved embeddings (assets unchanged). "
                  "Pass --rebuild to force a refresh.")
            store, items = store_cache.load()
            return cls(items, store, load_clip())

        # Otherwise (first run, files changed, or --rebuild): embed fresh + save.
        model = load_clip()
        print(f"Generating fresh embeddings from {os.path.basename(assets_dir)}/ ...")
        items = load_assets(assets_dir, image_dir)
        rag = cls._from_items(items, model)
        store_cache.save(rag.store, rag.items, files)
        print(f"Saved embeddings to {os.path.basename(store_cache.CACHE_DIR)}/ "
              "for next time.")
        return rag

    def counts(self):
        """How many of each content type were indexed."""
        out = {}
        for it in self.items:
            out[it["type"]] = out.get(it["type"], 0) + 1
        return out

    def search(self, query, k=4):
        """Embed the query, find nearest items, and print them."""
        q_vec = embed_query(query, self.model)
        hits = self.store.search(q_vec, k=k)
        print_results(query, hits, self.items)
        return hits
