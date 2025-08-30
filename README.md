
````markdown
# 🛍️ Multimodal RAG Customer Support System

A **Retrieval-Augmented Generation (RAG)** system designed for **customer support** on product catalogs.  
This project combines **textual product information** (descriptions, specifications, warranty details, FAQs) with **visual features** from product images to enable **multimodal question answering**.

Users can ask natural language queries (e.g., *“What is the warranty on the Garmin smartwatch?”* or *“Show me the red sneakers”*), and the system retrieves the most relevant product chunks (text + image embeddings) and generates accurate answers using an LLM.

---

## ✨ Features

- 🔎 **Multimodal Retrieval**: Supports both text and image queries.  
- 🖼 **Image-Aware Context**: Embeddings are generated from both **product descriptions** and **product images** (1024-dim joint vector).  
- 📦 **Product Metadata**: Each chunk retains metadata (ASIN, product name, image path, etc.).  
- 🧩 **Chunking by Product**: Long product descriptions are split into smaller, searchable chunks. Each new product starts a fresh chunking pipeline.  
- 🤖 **LLM-powered Answers**: Retrieved product chunks are passed into an LLM prompt for natural, fluent responses.  
- ⚡ **Vector Database Backed**: Efficient semantic search using embeddings.  
- 🛠 **Extensible**: Easy to plug in different LLMs or embedding models.  

---

## 🏗️ Implementation Overview

### 1. Data Ingestion
- Input: Product catalog with **text descriptions** and **images**.  
- For each product:
  - Text is split into **chunks**.  
  - Image embeddings are extracted.  
  - Combined into a **joint 1024-dim representation** (512 text + 512 image).  

### 2. Storage in Vector Database
- All embeddings + metadata are stored in a **vector DB**.  
- Each entry includes:
  - `page_content` → chunked text  
  - `metadata` → product_name, ASIN, image_path, etc.  
  - `embedding` → multimodal (text+image)  

### 3. Query Flow
- **Text Query**:
  - Encoded into a text embedding (512-dim).  
  - Compared against the 1024-dim multimodal vectors (text similarity dominates where image is absent).  
- **Image Query**:
  - Encoded into an image embedding (512-dim).  
  - Compared in the same multimodal space.  
- **Retrieval**:
  - Top-k similar chunks are retrieved with metadata.  
- **LLM Answer Generation**:
  - Retrieved chunks are passed into a **custom prompt template**.  
  - LLM generates a natural-language response.  

---

## 🚀 Usage

### Installation
```bash
git clone https://github.com/your-username/multimodal-rag-support.git
cd multimodal-rag-support
pip install -r requirements.txt
````

### Running the System

```bash
python main.py
```

### Example

```python
# Query about warranty (text-only)
answer_question(vectorstore, "What is the warranty of the Garmin smartwatch?")

# Query with product image
answer_question_with_image(vectorstore, "Show me this model in blue", image_path="sneaker.jpg")
```

**Sample Output**

```
🔍 Retrieved context:
1. Garmin Venu | ASIN: B08KJ
   Image: images/garmin.jpg
   Snip: This smartwatch includes a 1-year warranty covering manufacturing defects …

🤖 Answer:
The Garmin smartwatch comes with a 1-year warranty that covers manufacturing issues, but excludes accidental damage.
```

---

## 📌 Use Cases

* 🛒 **E-commerce Support Bots**: Let customers ask about warranty, specs, sizes, or availability.
* 📖 **Knowledge Base Search**: Index product manuals and provide support answers.
* 🎯 **Visual Product Search**: Customers can upload product photos to find similar items.
* 🤝 **Hybrid Support Agents**: Combine human + AI support with contextual product grounding.

---



## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/) for RAG pipelines
* [OpenAI / Hugging Face](https://huggingface.co/) for LLMs and embeddings
* Vector DB (FAISS)

---

