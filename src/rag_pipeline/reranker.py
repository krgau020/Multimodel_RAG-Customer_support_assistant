# # src/rag_pipeline/reranker.py
# ## better for text to text re-ranking








# from typing import List
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from langchain.docstore.document import Document

# # Load cross-encoder model once
# _rerank_model_name = "BAAI/bge-reranker-large"
# _tokenizer = AutoTokenizer.from_pretrained(_rerank_model_name)
# _model = AutoModelForSequenceClassification.from_pretrained(_rerank_model_name)

# def rerank(query: str, docs: List[Document], top_k: int = 4, debug: bool = True) -> List[Document]:
#     """
#     Re-rank retrieved docs using a cross-encoder reranker.
#     Args:
#         query: User query text
#         docs: List of LangChain Documents retrieved from FAISS
#         top_k: Return top_k docs after reranking
#     """
#     if not docs:
#         return []

#     pairs = [(query, d.page_content) for d in docs]
#     inputs = _tokenizer(
#         pairs,
#         padding=True,
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         scores = _model(**inputs).logits.squeeze(-1).cpu().numpy()

#     # Attach scores to docs
#     scored_docs = list(zip(docs, scores))
#     scored_docs.sort(key=lambda x: x[1], reverse=True)

#     if debug:
#         print("\n📊 Re-ranking scores:")
#         for i, (d, s) in enumerate(scored_docs, 1):
#             print(f"{i}. {d.metadata.get('product_name','Unknown')} | Score: {s:.4f}")

#     reranked_docs = [d for d, _ in scored_docs[:top_k]]
#     return reranked_docs
