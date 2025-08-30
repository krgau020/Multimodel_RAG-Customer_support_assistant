
# Build prompt with text and image context for LLM

import os
from pathlib import Path
from textwrap import shorten



from src.ingestion.process_image import preprocess_image  # Returns caption






# -------------------------- Prompt Builder --------------------------
def build_prompt(query: str, docs, query_image_path: str = None):
    contexts = []

    # Caption query image if provided
    query_img_desc = ""
    if query_image_path and Path(query_image_path).exists():
        try:
            q_caption = preprocess_image(query_image_path)
            query_img_desc = f"\n\nQuestion Image: {q_caption}"
        except Exception as e:
            query_img_desc = f"\n\n[Query image processing failed: {e}]"

    # Process retrieved docs
    for d in docs:
        snippet = shorten(d.page_content.replace("\n", " "), width=800, placeholder=" …")
        meta = d.metadata
        title = f"{meta.get('product_name','Unknown')} (ASIN: {meta.get('asin','?')})"

        image_desc = ""
        image_path = meta.get("image_path")
        if image_path and Path(image_path).exists():
            try:
                caption = preprocess_image(image_path)
                image_desc = f"\n  Image Description: {caption}"
            except Exception as e:
                image_desc = f"\n  [Image processing failed: {e}]"

        contexts.append(f"- {title}\n  {snippet}{image_desc}")

    context_block = "\n\n".join(contexts) if contexts else "No context retrieved."

    return f"""You are a customer support assistant. Answer based ONLY on the context.

Question:
{query}{query_img_desc}

Context:
{context_block}

Rules:
- Use only the given context, don't invent facts.
- If image description is provided, include relevant visual details.
- Present troubleshooting steps as bullet points if available.
"""
