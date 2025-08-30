# RAG pipeline evaluation with Gemini model

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.evaluation import load_evaluator

# Load API keys from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env file!")

# Initialize Gemini model for evaluation
eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def run_ragas_eval(queries, answers, contexts):
    """
    Run evaluation on RAG responses using LangChain evaluators (with Gemini).
    Args:
        queries (list[str]): List of input queries
        answers (list[str]): Generated answers from RAG
        contexts (list[list[str]]): Retrieved context chunks
    Returns:
        list[dict]: Evaluation results for each query
    """
    evaluator = load_evaluator("labeled_criteria", criteria="correctness", llm=eval_llm)

    results = []
    for query, answer, ctx in zip(queries, answers, contexts):
        grade = evaluator.evaluate_strings(
            input=query,
            prediction=answer,
            reference="\n".join(ctx) if ctx else "No context provided"
        )
        results.append({
            "query": query,
            "answer": answer,
            "context": ctx,
            "grade": grade.get("score", grade.get("reasoning", "N/A"))
        })

    return results
