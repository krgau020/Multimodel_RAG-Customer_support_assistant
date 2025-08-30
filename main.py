


# # main.py

# from pathlib import Path
# from textwrap import shorten


# from src.ingestion.load_json_and_chunk import load_json_data
# from src.vector_space.vectordb import build_faiss_index

# from src.utils.prompt_builder import build_prompt
# from src.utils.run_llm import run_llm
# from src.rag_pipeline.retriever import (
#     load_faiss_index,
#     retrieve_by_text,
#     retrieve_by_image,
#     retrieve_by_text_and_image
# )


# INDEX_PATH = "Dataset/processed_data/faiss.index"
# JSON_PATH = "Dataset/text-data_json"


# # -------------------------- Indexing --------------------------
# def ensure_index():
#     if not Path(INDEX_PATH).exists():
#         print("No index found. Building new FAISS index...")
#         chunks = load_json_data(JSON_PATH, chunk_size=300, chunk_overlap=50, debug=True)
#         build_faiss_index(chunks, INDEX_PATH, debug=True)
#     else:
#         print("Index already exists. Skipping build.")


# def get_vectorstore():
#     vectorstore = load_faiss_index(INDEX_PATH)
#     print("✅ Loaded FAISS index with", len(vectorstore.docstore._dict), "documents.")
#     return vectorstore






# # -------------------------- QA Functions --------------------------
# def answer_question(vectorstore, query: str):
#     docs = retrieve_by_text(vectorstore, query, k=4)

#     print("\n🔍 Retrieved context (text query):")
#     for i, d in enumerate(docs, 1):
#         meta = d.metadata
#         print(f"{i}. {meta.get('product_name','Unknown')} | ASIN: {meta.get('asin','?')}")
#         print(f"   Image: {meta.get('image_path','N/A')}")
#         print("   Snip:", shorten(d.page_content.replace('\n', ' '), width=140, placeholder=" …"))

#     prompt = build_prompt(query, docs)
#     return run_llm(prompt)


# def answer_image_question(vectorstore, image_path: str, query: str = None):
#     if query:
#         docs = retrieve_by_text_and_image(vectorstore, query, image_path, k=4)
#     else:
#         docs = retrieve_by_image(vectorstore, image_path, k=4)

#     print("\n🔍 Retrieved context (image query):")
#     for i, d in enumerate(docs, 1):
#         meta = d.metadata
#         print(f"{i}. {meta.get('product_name','Unknown')} | ASIN: {meta.get('asin','?')}")
#         print(f"   Image: {meta.get('image_path','N/A')}")
#         print("   Snip:", shorten(d.page_content.replace('\n', ' '), width=140, placeholder=" …"))

#     prompt = build_prompt(query or "", docs, query_image_path=image_path)
#     return run_llm(prompt)




# # -------------------------- Main --------------------------
# if __name__ == "__main__":
#     ensure_index()
#     vs = get_vectorstore()

#     # Example 1: text-only query
#     #answer_question(vs, "What is the warranty of the Garmin smartwatch?")




#     # Example 2: image + text query

#     # answer_image_question(
#     #     vs,
#     #     image_path=r"Dataset/images/1.Citizen-CZ-smart-watch-3.jpg",
#     #     query="What is the specification of this image?"
#     # )


#     # Example 3: image 

#     answer_image_question(
#         vs,
#         image_path=r"Dataset/images/1.Citizen-CZ-smart-watch-3.jpg",
#     )



























### main.py with evaluation


from pathlib import Path
from textwrap import shorten

from src.ingestion.load_json_and_chunk import load_json_data
from src.vector_space.vectordb import build_faiss_index
from src.utils.prompt_builder import build_prompt
from src.utils.run_llm import run_llm
from src.rag_pipeline.retriever import (
    load_faiss_index,
    retrieve_by_text,
    retrieve_by_image,
    retrieve_by_text_and_image
)
from src.evaluation.ragas_eval import run_ragas_eval   # 👈 NEW IMPORT

INDEX_PATH = "Dataset/processed_data/faiss.index"
JSON_PATH = "Dataset/text-data_json"


# -------------------------- Indexing --------------------------
def ensure_index():
    if not Path(INDEX_PATH).exists():
        print("No index found. Building new FAISS index...")
        chunks = load_json_data(JSON_PATH, chunk_size=300, chunk_overlap=50, debug=True)
        build_faiss_index(chunks, INDEX_PATH, debug=True)
    else:
        print("Index already exists. Skipping build.")


def get_vectorstore():
    vectorstore = load_faiss_index(INDEX_PATH)
    print("✅ Loaded FAISS index with", len(vectorstore.docstore._dict), "documents.")
    return vectorstore


# -------------------------- QA Functions --------------------------
def answer_question(vectorstore, query: str):
    docs = retrieve_by_text(vectorstore, query, k=4)

    print("\n🔍 Retrieved context (text query):")
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        print(f"{i}. {meta.get('product_name','Unknown')} | ASIN: {meta.get('asin','?')}")
        print(f"   Image: {meta.get('image_path','N/A')}")
        print("   Snip:", shorten(d.page_content.replace('\n', ' '), width=140, placeholder=" …"))

    prompt = build_prompt(query, docs)
    answer = run_llm(prompt)

    # Run evaluation 👇
    contexts = [[doc.page_content for doc in docs]]
    results = run_ragas_eval([query], [answer], contexts)
    print("\n📊 RAGAS Evaluation Results (Text Query):")
    print(results)

    return answer


def answer_image_question(vectorstore, image_path: str, query: str = None):
    if query:
        docs = retrieve_by_text_and_image(vectorstore, query, image_path, k=4)
    else:
        docs = retrieve_by_image(vectorstore, image_path, k=4)

    print("\n🔍 Retrieved context (image query):")
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        print(f"{i}. {meta.get('product_name','Unknown')} | ASIN: {meta.get('asin','?')}")
        print(f"   Image: {meta.get('image_path','N/A')}")
        print("   Snip:", shorten(d.page_content.replace('\n', ' '), width=140, placeholder=" …"))

    prompt = build_prompt(query or "", docs, query_image_path=image_path)
    answer = run_llm(prompt)




    # Run evaluation 👇
    contexts = [[doc.page_content for doc in docs]]
    results = run_ragas_eval([query or ""], [answer], contexts)
    print("\n📊 RAGAS Evaluation Results (Image Query):")
    print(results)

    return answer


# -------------------------- Main --------------------------
if __name__ == "__main__":
    ensure_index()
    vs = get_vectorstore()

    # Example 1: text-only query
    # answer_question(vs, "What is the warranty of the Garmin smartwatch?")

    # Example 2: image + text query
    # answer_image_question(
    #     vs,
    #     image_path=r"Dataset/images/1.Citizen-CZ-smart-watch-3.jpg",
    #     query="What is the specification of this image?"
    # )

    # Example 3: image only
    answer_image_question(
        vs,
        image_path=r"C:\Users\admin\Desktop\multimodel-rag-Customer_support\uploads\7.smart-dimmer.jpg",
    )
# -------------------------- End --------------------------











# # main.py with reranker -- Ignore --

# from pathlib import Path
# from textwrap import shorten

# from src.ingestion.load_json_and_chunk import load_json_data
# from src.vector_space.vectordb import build_faiss_index

# from src.utils.prompt_builder import build_prompt
# from src.utils.run_llm import run_llm
# from src.rag_pipeline.retriever import (
#     load_faiss_index,
#     retrieve_by_text,
#     retrieve_by_image,
#     retrieve_by_text_and_image
# )
# from src.rag_pipeline.reranker import rerank   # ✅ new import


# INDEX_PATH = "Dataset/processed_data/faiss.index"
# JSON_PATH = "Dataset/text-data_json"


# # -------------------------- Indexing --------------------------
# def ensure_index():
#     if not Path(INDEX_PATH).exists():
#         print("No index found. Building new FAISS index...")
#         chunks = load_json_data(JSON_PATH, chunk_size=300, chunk_overlap=50, debug=True)
#         build_faiss_index(chunks, INDEX_PATH, debug=True)
#     else:
#         print("Index already exists. Skipping build.")


# def get_vectorstore():
#     vectorstore = load_faiss_index(INDEX_PATH)
#     print("✅ Loaded FAISS index with", len(vectorstore.docstore._dict), "documents.")
#     return vectorstore


# # -------------------------- QA Functions --------------------------
# def answer_question(vectorstore, query: str):
#     docs = retrieve_by_text(vectorstore, query, k=10)   # fetch more
#     docs = rerank(query, docs, top_k=4)                 # ✅ re-rank

#     print("\n🔍 Retrieved + Re-ranked context (text query):")
#     for i, d in enumerate(docs, 1):
#         meta = d.metadata
#         print(f"{i}. {meta.get('product_name','Unknown')} | ASIN: {meta.get('asin','?')}")
#         print(f"   Image: {meta.get('image_path','N/A')}")
#         print("   Snip:", shorten(d.page_content.replace('\n', ' '), width=140, placeholder=" …"))

#     prompt = build_prompt(query, docs)
#     return run_llm(prompt)


# def answer_image_question(vectorstore, image_path: str, query: str = None):
#     if query:
#         docs = retrieve_by_text_and_image(vectorstore, query, image_path, k=10)
#         docs = rerank(query, docs, top_k=4)             # ✅ re-rank
#     else:
#         docs = retrieve_by_image(vectorstore, image_path, k=4)
#         # reranker needs query text, so only apply when query is given

#     print("\n🔍 Retrieved context (image query):")
#     for i, d in enumerate(docs, 1):
#         meta = d.metadata
#         print(f"{i}. {meta.get('product_name','Unknown')} | ASIN: {meta.get('asin','?')}")
#         print(f"   Image: {meta.get('image_path','N/A')}")
#         print("   Snip:", shorten(d.page_content.replace('\n', ' '), width=140, placeholder=" …"))

#     prompt = build_prompt(query or "", docs, query_image_path=image_path)
#     return run_llm(prompt)


# # -------------------------- Main --------------------------
# if __name__ == "__main__":
#     ensure_index()
#     vs = get_vectorstore()

#     # # Example 1: text-only query
#     # answer_question(vs, "What is the warranty of the Garmin smartwatch?")

#     # Example 2: image + text query
#     answer_image_question(
#         vs,
#         image_path=r"Dataset/images/1.Citizen-CZ-smart-watch-3.jpg",
#         query="What is the specification of this image?"
#     )

#     # Example 3: image only (no rerank)
#     # answer_image_question(
#     #     vs,
#     #     image_path=r"Dataset/images/1.Citizen-CZ-smart-watch-3.jpg",
#     # )
