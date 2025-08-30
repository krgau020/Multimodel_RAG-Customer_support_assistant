
# ==============================
# Load FAISS index
# ==============================
from langchain_community.vectorstores import FAISS
import pickle

def load_faiss_index(index_path):
    """
    Load a multimodal FAISS index using precomputed embeddings.
    """
    with open(index_path.replace(".index", "_store.pkl"), "rb") as f:
        docs, embeddings, metadata_list = pickle.load(f)

    if not docs or not embeddings.any():
        raise ValueError("❌ FAISS store is empty — check build_faiss_index output.")

    vectorstore = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(docs, embeddings)],
        embedding=None,  # Already have embeddings
        metadatas=metadata_list
    )
    return vectorstore















import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.embedding.text_embedding import embed_text
from src.embedding.image_embadding import embed_image

# --------------------------
# Helper: concat embedding spaces
# --------------------------
def embed_query_for_concat_space(query_text: str) -> np.ndarray:
    q_text = embed_text(query_text, debug=False)
    q_img_pad = np.zeros_like(q_text)
    return np.concatenate([q_text, q_img_pad]).astype("float32")


def embed_image_for_concat_space(image_path: str) -> np.ndarray:
    img_emb = embed_image(image_path, debug=False)
    img_text_pad = np.zeros_like(img_emb)
    return np.concatenate([img_text_pad, img_emb]).astype("float32")


# --------------------------
# Retrieval functions
# --------------------------
def retrieve_by_text(vectorstore, query_text: str, k: int = 4):
    q_vec = embed_query_for_concat_space(query_text)
    return vectorstore.similarity_search_by_vector(embedding=q_vec, k=k)


def retrieve_by_image(vectorstore, image_path: str, k: int = 4):
    q_vec = embed_image_for_concat_space(image_path)
    return vectorstore.similarity_search_by_vector(embedding=q_vec, k=k)


def retrieve_by_text_and_image(vectorstore, query_text: str, image_path: str, k: int = 4):
    text_emb = embed_query_for_concat_space(query_text)
    image_emb = embed_image_for_concat_space(image_path)
    combined_emb = (text_emb + image_emb) / 2.0
    return vectorstore.similarity_search_by_vector(embedding=combined_emb, k=k)
