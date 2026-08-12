# Multimodal RAG — text + tables + images in one vector DB

**Big idea:** real documents aren't just text — they have **paragraphs, tables,
and images/charts**. This example builds one vector database over all three and
retrieves across them with a single query.

## Run
```bash
# from the project root, with the venv active (see top-level README)
python examples/pdf-multi-content-rag/example.py
```
First run downloads the CLIP model (~600 MB) into the project's `.models/` cache.

## What it does
1. **Reads** `sample_document_with_chart.pdf` and pulls out its three content
   types: the **text** paragraph, the **table** (Quarter / Revenue / Growth),
   and the embedded **chart image**.
2. **Embeds everything with CLIP** — one model that maps *both* text and images
   into the **same** vector space, so a text query can match any content type.
3. **Builds a FAISS vector DB** storing the vector + the original content + a
   `type` tag (text / table / image) — the "store 3 things" pattern of a real
   vector database.
4. **Retrieves** by query and prints nicely-formatted results: the matched
   **type**, the similarity **score**, and the **page**.

## What you'll see
Three queries, each designed to surface a different content type as the top hit:

| Query | Top result |
|---|---|
| "What does the company report say?" | 📝 **text** |
| "quarterly revenue growth numbers" | 📊 **table** |
| "picture of the revenue bar chart" | 🖼️ **image** |

Extracted images are saved to `extracted_images/` so you can open them.

## The "latest way" — and an honest caveat
The modern approach to multimodal RAG is a **shared embedding space** (CLIP) plus
**type metadata**, so one query retrieves across modalities. But CLIP is trained
on natural photos and is **weak at charts/diagrams**, and its text↔image scores
sit on a lower scale than text↔text scores.

The production fix (used here) is **"describe-then-embed"**: store the image
*and* a short **caption**, then embed both and average them. That's why the chart
becomes findable by words. In a real system the caption comes from an
image-captioning / vision model (e.g. a vision LLM); here we use the document's
own words about the graphic to keep it API-key-free.

## Words to know
- **Multimodal embedding** — one model, one vector space, for several data types.
- **CLIP** — the model that shares a space for text and images.
- **Describe-then-embed** — caption an image, then embed the caption too, so text
  queries can find it.
- **Type metadata** — the `text` / `table` / `image` tag stored beside each vector.

→ Related: [image-embeddings demo](../image-embeddings/) ·
[classic PDF RAG](../pdf-rag/) ·
[full theory](../../learning_notes.md#deep-dive-image-embeddings-searching-pictures)
