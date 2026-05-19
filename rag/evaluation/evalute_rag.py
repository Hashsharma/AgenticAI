"""
RAG Evaluation Script (v1.2.x Compatible) - With DeepEval Wrapper
"""

import os
import sys
import json
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- FIXED IMPORTS ---
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# PipelinePromptTemplate import fix
try:
    from langchain.prompts.pipeline import PipelinePromptTemplate
except ImportError:
    try:
        from langchain.prompts import PipelinePromptTemplate
    except ImportError:
        class PipelinePromptTemplate:
            def __init__(self, *args, **kwargs):
                raise ImportError("PipelinePromptTemplate not available")

# DeepEval metrics - Import but we'll use them differently
try:
    from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print("Warning: deepeval not installed. Install with: pip install deepeval")

# ---------------- LM Studio Configuration ----------------
MISTRAL_API = os.getenv("MISTRAL_API", "http://localhost:1234/v1")

# ---------------- Custom Wrapper for DeepEval with Local Models ----------------
class LocalLLMWrapper:
    """Wrapper to make local LLM compatible with DeepEval"""
    def __init__(self, base_url: str, model_name: str = "local-model"):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            base_url=base_url,
        )
    
    def generate(self, prompt: str) -> str:
        """Generate response using local LLM"""
        response = self.llm.invoke(prompt)
        return response.content

# Only initialize DeepEval metrics if available and if you want to use them
if DEEPEVAL_AVAILABLE:
    # Note: This may still not work as DeepEval expects OpenAI API
    # You might need to monkey-patch or use a different approach
    print("Note: DeepEval metrics may not work with local models")
    
    # These will likely fail with local models
    # correctness_metric = GEval(...)
    # faithfulness_metric = FaithfulnessMetric(...)
    # relevance_metric = ContextualRelevancyMetric(...)

# ---------------- RAG Evaluation ----------------
def evaluate_rag(retriever, num_questions: int = 5, use_deepeval: bool = False) -> Dict[str, Any]:
    llm = ChatOpenAI(
        temperature=0,
        model_name="local-model",
        base_url=MISTRAL_API,
        api_key=""
    )

    # Evaluation chain using LangChain (works with local models)
    eval_prompt = PromptTemplate.from_template("""
    Question: {question}
    Context: {context}
    Rate 1-5 for Relevance, Completeness, and Conciseness. Return ONLY JSON.
    """)
    eval_chain = eval_prompt | llm | StrOutputParser()

    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} questions about climate change. One per line."
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    raw_qs = question_chain.invoke({"num_questions": num_questions})
    questions = [q.strip() for q in raw_qs.strip().split("\n") if q.strip()]

    results = []
    for question in questions:
        docs = retriever.invoke(question)
        context_text = "\n".join([d.page_content for d in docs])

        eval_output = eval_chain.invoke({"question": question, "context": context_text})
        
        result = {
            "question": question,
            "context": context_text,
            "eval_result": eval_output
        }
        
        # Optionally add DeepEval metrics if requested (may fail)
        if use_deepeval and DEEPEVAL_AVAILABLE:
            try:
                # This will likely fail with local models
                result["deepeval_warning"] = "DeepEval requires OpenAI API, not compatible with local models"
            except Exception as e:
                result["deepeval_error"] = str(e)
        
        results.append(result)

    return {
        "results": results,
        "average_scores": calculate_average_scores(results)
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    score_sums = defaultdict(float)
    count = 0
    for r in results:
        try:
            clean_json = r["eval_result"].replace("```json", "").replace("```", "").strip()
            scores = json.loads(clean_json)
            for k, v in scores.items():
                score_sums[k] += float(v)
            count += 1
        except: continue
    return {k: v / count for k, v in score_sums.items()} if count > 0 else {}

if __name__ == "__main__":
    # Test LM Studio connection
    try:
        test_llm = ChatOpenAI(
            temperature=0,
            model_name="test",
            base_url=LM_STUDIO_URL,
            api_key="lm-studio"
        )
        response = test_llm.invoke("Say 'LM Studio is working'")
        print(f"LM Studio connection successful: {response.content}")
        
        # To run evaluation:
        # output = evaluate_rag(your_retriever_instance)
        
    except Exception as e:
        print(f"LM Studio connection failed: {e}")
        print("Make sure LM Studio is running and a model is loaded")