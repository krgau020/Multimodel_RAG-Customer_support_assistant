import os
import faiss
import pickle
import numpy as np
from src.embedding.image_embadding import embed_image
from src.embedding.text_embedding import embed_text

def build_faiss_index(chunks, index_path="Dataset/processed_data/faiss.index", debug=True):
    docs = []
    embeddings = []
    metadata_list = []

    if not chunks:
        raise ValueError("No chunks provided to build_faiss_index — check your data loader.")

    for i, chunk in enumerate(chunks):
        if debug:
            print(f"\n[DEBUG] Processing chunk {i+1}/{len(chunks)}")
            print(f"Metadata: {chunk.metadata}")

        # Text embedding
        text_emb = embed_text(chunk.page_content, debug=debug)

        # Image embedding
        image_path = chunk.metadata.get("image_path", None)
        if image_path and os.path.exists(image_path):
            image_emb = embed_image(image_path, debug=debug)
        else:
            print(f"⚠️ Warning: Image path missing or not found for chunk {i}: {image_path}")
            image_emb = np.zeros(text_emb.shape)  # same dimension as text embedding

        # Sanity checks
        if text_emb is None or text_emb.size == 0:
            raise ValueError(f"❌ No text embedding for chunk {i}")
        if image_emb is None or image_emb.size == 0:
            raise ValueError(f"❌ No image embedding for chunk {i}")

        # Combine embeddings
        combined_emb = np.concatenate([text_emb, image_emb])
        embeddings.append(combined_emb)
        metadata_list.append(chunk.metadata)
        docs.append(chunk)

        if debug:
            print(f"Combined embedding shape: {combined_emb.shape}")
            print(f"First 5 values: {combined_emb[:5]}")

    if not embeddings:
        raise ValueError("❌ No embeddings created — check your embedding functions.")

    embeddings = np.array(embeddings).astype("float32")
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and data
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(index_path.replace(".index", "_store.pkl"), "wb") as f:
        pickle.dump((docs, embeddings, metadata_list), f)

    print(f"\n✅ FAISS index built with {len(docs)} docs, dimension {dim}")

