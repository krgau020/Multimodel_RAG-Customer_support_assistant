import streamlit as st
from pathlib import Path
from src.rag_pipeline.retriever import (
    load_faiss_index,
    retrieve_by_text,
    retrieve_by_image,
    retrieve_by_text_and_image
)
from src.ingestion.load_json_and_chunk import load_json_data
from src.vector_space.vectordb import build_faiss_index
from src.utils.prompt_builder import build_prompt
from src.utils.run_llm import run_llm
import tempfile, shutil, os

# -------------------- Config --------------------
INDEX_PATH = "Dataset/processed_data/faiss.index"
JSON_PATH = "Dataset/text-data_json"

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Multimodal Customer Support RAG", layout="wide")
st.title("📦 Customer Support Assistant (Text + Image)")

st.markdown("#### Input your query below. You can use text, image, or both.")

query_text = st.text_area("Text Query", placeholder="Type your question here...")
query_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if query_image:
    st.image(query_image, caption="Uploaded Image", width=256)

submit_btn = st.button("Generate Answer")

# -------------------- Load or Build FAISS --------------------
@st.cache_resource(show_spinner=True)
def load_or_build_index():
    if not Path(INDEX_PATH).exists():
        st.info("No FAISS index found. Building new index...")
        chunks = load_json_data(JSON_PATH, chunk_size=300, chunk_overlap=50, debug=False)
        build_faiss_index(chunks, INDEX_PATH, debug=True)
        st.success("✅ FAISS index built successfully.")
    vectorstore = load_faiss_index(INDEX_PATH)
    return vectorstore

vectorstore = load_or_build_index()

# -------------------- Helper: Temp save uploaded image --------------------
def save_uploaded_image(uploaded_file):
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / uploaded_file.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path, tmp_dir

# -------------------- Answer Generation --------------------
if submit_btn:
    if not query_text and not query_image:
        st.warning("⚠️ Please provide text or upload an image (or both).")
    else:
        image_path, tmp_dir = (None, None)
        if query_image:
            image_path, tmp_dir = save_uploaded_image(query_image)

        try:
            # Retrieve docs
            if query_text and image_path:
                docs = retrieve_by_text_and_image(vectorstore, query_text, str(image_path), k=4)
            elif query_text:
                docs = retrieve_by_text(vectorstore, query_text, k=4)
            elif image_path:
                docs = retrieve_by_image(vectorstore, str(image_path), k=4)
            else:
                docs = []

            # # Show retrieved docs
            # st.markdown("#### 🔍 Retrieved Documents/Chunks:")
            # if docs:
            #     for i, doc in enumerate(docs):
            #         doc_text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            #         st.markdown(f"**Chunk {i+1}:** {doc_text}")
            # else:
            #     st.info("No relevant documents found.")

            # Build prompt and run LLM
            prompt = build_prompt(query_text or "", docs, query_image_path=str(image_path) if image_path else None)
            with st.spinner("Generating answer..."):
                answer = run_llm(prompt)
            st.success("✅ Answer generated!")
            st.markdown("### 🟢 LLM Answer:")
            st.text_area("LLM Answer", value=answer, height=300, label_visibility="collapsed")
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            # Cleanup temp image
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)


















# ## with reranker  -- Ignore --


# import streamlit as st
# from pathlib import Path
# from src.rag_pipeline.retriever import (
#     load_faiss_index,
#     retrieve_by_text,
#     retrieve_by_image,
#     retrieve_by_text_and_image
# )
# from src.rag_pipeline.reranker import rerank   # ✅ NEW
# from src.ingestion.load_json_and_chunk import load_json_data
# from src.vector_space.vectordb import build_faiss_index
# from src.utils.prompt_builder import build_prompt
# from src.utils.run_llm import run_llm
# import tempfile, shutil, os

# # -------------------- Config --------------------
# INDEX_PATH = "Dataset/processed_data/faiss.index"
# JSON_PATH = "Dataset/text-data_json"

# # -------------------- Streamlit UI --------------------
# st.set_page_config(page_title="Multimodal Customer Support RAG", layout="wide")
# st.title("📦 Customer Support Assistant (Text + Image)")

# st.markdown("#### Input your query below. You can use text, image, or both.")

# query_text = st.text_area("Text Query", placeholder="Type your question here...")
# query_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
# top_k = st.slider("Number of results (Top-K)", min_value=1, max_value=10, value=4)  # ✅ adjustable top_k

# if query_image:
#     st.image(query_image, caption="Uploaded Image", width=256)

# submit_btn = st.button("Generate Answer")

# # -------------------- Load or Build FAISS --------------------
# @st.cache_resource(show_spinner=True)
# def load_or_build_index():
#     if not Path(INDEX_PATH).exists():
#         st.info("No FAISS index found. Building new index...")
#         chunks = load_json_data(JSON_PATH, chunk_size=300, chunk_overlap=50, debug=False)
#         build_faiss_index(chunks, INDEX_PATH, debug=True)
#         st.success("✅ FAISS index built successfully.")
#     vectorstore = load_faiss_index(INDEX_PATH)
#     return vectorstore

# vectorstore = load_or_build_index()

# # -------------------- Helper: Temp save uploaded image --------------------
# def save_uploaded_image(uploaded_file):
#     tmp_dir = tempfile.mkdtemp()
#     tmp_path = Path(tmp_dir) / uploaded_file.name
#     with open(tmp_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return tmp_path, tmp_dir

# # -------------------- Answer Generation --------------------
# if submit_btn:
#     if not query_text and not query_image:
#         st.warning("⚠️ Please provide text or upload an image (or both).")
#     else:
#         image_path, tmp_dir = (None, None)
#         if query_image:
#             image_path, tmp_dir = save_uploaded_image(query_image)

#         try:
#             # Retrieve docs (get more for reranking)
#             if query_text and image_path:
#                 docs = retrieve_by_text_and_image(vectorstore, query_text, str(image_path), k=max(top_k * 2, 10))
#                 docs = rerank(query_text, docs, top_k=top_k)
#             elif query_text:
#                 docs = retrieve_by_text(vectorstore, query_text, k=max(top_k * 2, 10))
#                 docs = rerank(query_text, docs, top_k=top_k)
#             elif image_path:
#                 # Image-only query → no reranking (needs text)
#                 docs = retrieve_by_image(vectorstore, str(image_path), k=top_k)
#             else:
#                 docs = []

#             # Build prompt and run LLM
#             prompt = build_prompt(query_text or "", docs, query_image_path=str(image_path) if image_path else None)
#             with st.spinner("Generating answer..."):
#                 answer = run_llm(prompt)
#             st.success("✅ Answer generated!")
#             st.markdown("### 🟢 LLM Answer:")
#             st.text_area("LLM Answer", value=answer, height=300, label_visibility="collapsed")

#         except Exception as e:
#             st.error(f"❌ Error: {e}")
#         finally:
#             # Cleanup temp image
#             if tmp_dir and os.path.exists(tmp_dir):
#                 shutil.rmtree(tmp_dir)
