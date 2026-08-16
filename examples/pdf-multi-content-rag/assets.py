"""
ASSETS — read every supported file from the assets/ folder.

Drop any mix of PDFs, images, and text files into assets/ and this indexes
them all into ONE vector DB. Each item remembers which file it came from
(its `source`), so results can say "from report.pdf, page 3".
"""

import os

from extraction import SUPPORTED_EXTS, extract_any


def list_asset_files(assets_dir):
    """Return the supported files in assets/ (sorted, non-recursive)."""
    if not os.path.isdir(assets_dir):
        return []
    files = []
    for name in sorted(os.listdir(assets_dir)):
        path = os.path.join(assets_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
            files.append(path)
    return files


def load_assets(assets_dir, image_dir):
    """Extract items from every supported file in assets/. Returns a list."""
    items = []
    for path in list_asset_files(assets_dir):
        found = extract_any(path, image_dir)
        print(f"  · {os.path.basename(path)}: {len(found)} items")
        items.extend(found)
    return items
