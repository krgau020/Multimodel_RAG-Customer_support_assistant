# fastapi_main.py

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import shutil
import os
from pathlib import Path
import uvicorn

from src.ingestion.load_json_and_chunk import load_json_data
from src.vector_space.vectordb import build_faiss_index
from src.ingestion.process_image import preprocess_image  # Returns caption
from src.rag_pipeline.retriever import (
    load_faiss_index,
    retrieve_by_text,
    retrieve_by_image,
    retrieve_by_text_and_image
)
from src.utils.prompt_builder import build_prompt
from src.utils.run_llm import run_llm

INDEX_PATH = "Dataset/processed_data/faiss.index"
JSON_PATH = "Dataset/text-data_json"

app = FastAPI(title="Multimodal RAG API - Customer Support")

# -------------------------- Startup --------------------------
@app.on_event("startup")
def startup_event():
    global vectorstore
    if not Path(INDEX_PATH).exists():
        print("No index found. Building new FAISS index...")
        chunks = load_json_data(JSON_PATH, chunk_size=300, chunk_overlap=50, debug=True)
        build_faiss_index(chunks, INDEX_PATH, debug=True)
    vectorstore = load_faiss_index(INDEX_PATH)
    print("✅ FAISS index loaded with", len(vectorstore.docstore._dict), "documents.")


# -------------------------- Models --------------------------
class QueryRequest(BaseModel):
    query: Optional[str] = None
    image_path: Optional[str] = None
    top_k: int = 4


# -------------------------- Endpoints --------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query_text")
async def query_text(request: QueryRequest):
    if not request.query:
        return {"error": "Provide a text query."}
    docs = retrieve_by_text(vectorstore, request.query, k=request.top_k)
    prompt = build_prompt(request.query, docs)
    answer = run_llm(prompt)
    return {"answer": answer}


@app.post("/query_image")
async def query_image(file: UploadFile = File(...), top_k: int = Form(4)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = retrieve_by_image(vectorstore, file_path, k=top_k)
    prompt = build_prompt("", docs, query_image_path=file_path)
    answer = run_llm(prompt)
    return {"answer": answer}


@app.post("/query_image_text")
async def query_image_text(file: UploadFile = File(...), query: str = Form(...), top_k: int = Form(4)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = retrieve_by_text_and_image(vectorstore, query, file_path, k=top_k)
    prompt = build_prompt(query, docs, query_image_path=file_path)
    answer = run_llm(prompt)
    return {"answer": answer}


# -------------------------- Run --------------------------
if __name__ == "__main__":
    
    uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8080, reload=False)












""""To run this fast api
    1. python fastapi_main.py
    
    go to http://localhost:8080/docs
    
    and try out all the options
    
    """



















# # fastapi_main.py with reranker -- Ignore --

# from fastapi import FastAPI, UploadFile, File, Form
# from pydantic import BaseModel
# from typing import Optional
# import shutil
# import os
# from pathlib import Path
# import uvicorn

# from src.ingestion.load_json_and_chunk import load_json_data
# from src.vector_space.vectordb import build_faiss_index
# from src.ingestion.process_image import preprocess_image  # Returns caption
# from src.rag_pipeline.retriever import (
#     load_faiss_index,
#     retrieve_by_text,
#     retrieve_by_image,
#     retrieve_by_text_and_image
# )
# from src.rag_pipeline.reranker import rerank   # ✅ NEW
# from src.utils.prompt_builder import build_prompt
# from src.utils.run_llm import run_llm

# INDEX_PATH = "Dataset/processed_data/faiss.index"
# JSON_PATH = "Dataset/text-data_json"

# app = FastAPI(title="Multimodal RAG API - Customer Support")

# # -------------------------- Startup --------------------------
# @app.on_event("startup")
# def startup_event():
#     global vectorstore
#     if not Path(INDEX_PATH).exists():
#         print("No index found. Building new FAISS index...")
#         chunks = load_json_data(JSON_PATH, chunk_size=300, chunk_overlap=50, debug=True)
#         build_faiss_index(chunks, INDEX_PATH, debug=True)
#     vectorstore = load_faiss_index(INDEX_PATH)
#     print("✅ FAISS index loaded with", len(vectorstore.docstore._dict), "documents.")


# # -------------------------- Models --------------------------
# class QueryRequest(BaseModel):
#     query: Optional[str] = None
#     image_path: Optional[str] = None
#     top_k: int = 4


# # -------------------------- Endpoints --------------------------
# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# @app.post("/query_text")
# async def query_text(request: QueryRequest):
#     if not request.query:
#         return {"error": "Provide a text query."}

#     # retrieve more, rerank, keep top_k
#     docs = retrieve_by_text(vectorstore, request.query, k=max(request.top_k * 2, 10))
#     docs = rerank(request.query, docs, top_k=request.top_k)

#     prompt = build_prompt(request.query, docs)
#     answer = run_llm(prompt)
#     return {"answer": answer}


# @app.post("/query_image")
# async def query_image(file: UploadFile = File(...), top_k: int = Form(4)):
#     upload_dir = "uploads"
#     os.makedirs(upload_dir, exist_ok=True)
#     file_path = os.path.join(upload_dir, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Image-only queries → no text for reranker, keep as-is
#     docs = retrieve_by_image(vectorstore, file_path, k=top_k)

#     prompt = build_prompt("", docs, query_image_path=file_path)
#     answer = run_llm(prompt)
#     return {"answer": answer}


# @app.post("/query_image_text")
# async def query_image_text(file: UploadFile = File(...), query: str = Form(...), top_k: int = Form(4)):
#     upload_dir = "uploads"
#     os.makedirs(upload_dir, exist_ok=True)
#     file_path = os.path.join(upload_dir, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # retrieve more, rerank, keep top_k
#     docs = retrieve_by_text_and_image(vectorstore, query, file_path, k=max(top_k * 2, 10))
#     docs = rerank(query, docs, top_k=top_k)

#     prompt = build_prompt(query, docs, query_image_path=file_path)
#     answer = run_llm(prompt)
#     return {"answer": answer}


# # -------------------------- Run --------------------------
# if __name__ == "__main__":
#     uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8080, reload=False)










