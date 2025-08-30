#load and chunk JSON product data with image paths for multimodal retrieval

import json
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _stringify_specifications(spec):
    """Turn specs (possibly a dict) into a readable string."""
    if spec is None:
        return ""
    if isinstance(spec, dict):
        return ", ".join(f"{k}: {v}" for k, v in spec.items())
    return str(spec)

def _product_iter(obj):
    """Yield product dicts whether obj is a dict (single) or list (multiple)."""
    if isinstance(obj, dict):
        # single product
        yield obj
    elif isinstance(obj, list):
        # list of products
        for p in obj:
            if isinstance(p, dict):
                yield p
            else:
                # skip non-dicts
                continue
    else:
        # unsupported root type
        return

def load_json_data(json_folder, chunk_size=300, chunk_overlap=50, debug=True):
    """
    Load JSON files, create semantic chunks, and associate product image path.
    Supports:
      - One product per file (dict)
      - Multiple products per file (list[dict])
    """
    all_chunks = []

    for json_file in Path(json_folder).glob("*.json"):
        if debug:
            print(f"\n[DEBUG] Reading JSON file: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            root = json.load(f)

        # Iterate products regardless of whether file has a dict or list
        for product in _product_iter(root):
            asin = product.get("asin", "")
            name = product.get("name", "Unknown")
            category = product.get("category", "Unknown")
            description = product.get("description", "")

            support_data = product.get("support_data", {}) or {}
            common_issues = support_data.get("common_issues", []) or []
            steps = support_data.get("troubleshooting_steps", []) or []
            warranty = support_data.get("warranty", "")
            specifications = _stringify_specifications(support_data.get("specifications", {}))

            image_path = product.get("image_url", "")  # expected to be a local path

            product_text = (
                f"Product: {name} ({asin})\n"
                f"Category: {category}\n"
                f"Description: {description}\n"
                f"Common Issues: {', '.join(common_issues)}\n"
                f"Troubleshooting: {', '.join(steps)}\n"
                f"Warranty: {warranty}\n"
                f"Specifications: {specifications}\n"
            )

            metadata = {
                "asin": asin,
                "product_name": name,
                "image_path": image_path,
                "source_file": str(json_file)
            }

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            temp_doc = Document(page_content=product_text, metadata=metadata)
            chunks = splitter.split_documents([temp_doc])

            for chunk in chunks:
                # Ensure image path is attached to each chunk of this product
                chunk.metadata["image_path"] = image_path
                all_chunks.append(chunk)

    if debug:
        print("\n[DEBUG] Loaded Chunks:")
        for i, c in enumerate(all_chunks, start=1):
            print(f"\n--- Chunk {i} ---")
            print(c.page_content)
            print(f"Metadata: {c.metadata}")

    return all_chunks
