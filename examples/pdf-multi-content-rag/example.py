"""
MULTIMODAL RAG over ANY PDF — text + tables + images in one vector DB.

Real documents mix content types: paragraphs, tables of numbers, and
charts/images. This builds ONE vector database over all three, from ANY PDF, and
retrieves across them with a single query.

This file is just the ENTRY POINT (CLI + demo queries). The actual work lives in
small, single-purpose modules — read them in this order to learn the flow:

    chunking.py     split page text into overlapping pieces
    extraction.py   pull text / tables / images out of the PDF
    embeddings.py   turn items (and queries) into CLIP vectors
    vector_store.py hold the vectors + find the nearest ones (FAISS)
    retrieval.py    run a query and print results nicely
    pipeline.py     ties it all together (MultimodalRAG)

Point it at any PDF with --pdf; defaults to the bundled sample.
No API key needed. CLIP runs locally.

Run (from project root, venv active):
    python examples/pdf-multi-content-rag/example.py
    python examples/pdf-multi-content-rag/example.py --pdf path/to/your.pdf
    python examples/pdf-multi-content-rag/example.py --pdf your.pdf -q "your question"
"""

import argparse
import os
import sys

# Allow importing this folder's own modules (chunking, pipeline, ...) AND the
# shared helpers, which live in the parent examples/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                    # this example's modules
sys.path.insert(0, os.path.dirname(HERE))   # examples/  (has shared.py)

import store_cache
from assets import list_asset_files
from pipeline import ASSETS_DIR, IMG_DIR, MultimodalRAG


def main():
    ap = argparse.ArgumentParser(
        description="Multimodal RAG over the assets/ folder (or a single PDF).")
    ap.add_argument("--pdf", help="index a single PDF instead of the assets/ folder")
    ap.add_argument("--assets", default=ASSETS_DIR,
                    help="folder of PDFs/images/text files to index (default: assets/)")
    ap.add_argument("--rebuild", action="store_true",
                    help="force fresh embeddings even if the saved cache is current")
    ap.add_argument("--clean", action="store_true",
                    help="delete saved embeddings + extracted images, then exit")
    ap.add_argument("-q", "--query", action="append",
                    help="a query (repeatable); omit to use built-in demo queries")
    ap.add_argument("-k", type=int, default=4, help="results per query")
    args = ap.parse_args()

    # --clean: wipe generated artifacts and stop.
    if args.clean:
        removed = store_cache.clean()
        print("Removed: " + (", ".join(removed) if removed else "nothing to clean"))
        return

    try:
        if args.pdf:
            if not os.path.exists(args.pdf):
                sys.exit(f"PDF not found: {args.pdf}")
            print(f"Reading {os.path.basename(args.pdf)} ...")
            rag = MultimodalRAG.from_pdf(args.pdf)
        else:
            # Default: read every supported file from assets/ (cached).
            if not list_asset_files(args.assets):
                sys.exit(
                    f"No files to index in {os.path.relpath(args.assets)}/.\n"
                    "Drop some PDFs / images / text files in there, or pass "
                    "--pdf path/to/file.pdf.")
            rag = MultimodalRAG.from_assets(args.assets, rebuild=args.rebuild)
    except ValueError as e:
        sys.exit(str(e))

    counts = rag.counts()
    print("Extracted:", ", ".join(f"{n} {t}" for t, n in sorted(counts.items())))
    print(f"Stored {rag.store.size} items in a FAISS vector DB "
          f"(text + tables + images, one shared space).")

    queries = args.query or [
        "What were the total revenues this quarter?",
        "revenue by geographic area",
        "a chart or figure in the report",
    ]
    for q in queries:
        rag.search(q, k=args.k)

    print("Takeaway: one CLIP-powered vector DB retrieves across text, tables, and")
    print("images from every file in assets/. The TYPE + SOURCE + PAGE metadata")
    print(f"tells you what matched and where. Cropped images saved to "
          f"{os.path.basename(IMG_DIR)}/.")


if __name__ == "__main__":
    main()
