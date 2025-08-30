
# run LLM with Gemini 1.5

# main.py
import os

from dotenv import load_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv()

def run_llm(prompt: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not set in .env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )
    resp = llm.invoke(prompt)
    print("\n🟩 Answer:\n")
    print(resp.content)
    return resp.content

