import requests
from typing import List, Dict, Any
from src.core.config import GROQ_API_KEY
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# Groq API - OpenAI-compatible, free tier, extremely fast
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class LLMGenerator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize the LLM using the Groq API.
        Free tier: 14,400 requests/day, no credit card needed.
        Model options:
          - llama-3.1-8b-instant  (fastest)
          - llama3-70b-8192       (most capable)
          - mixtral-8x7b-32768    (great for long contexts)
        """
        logger.info(f"Initializing LLM via Groq: {model_name}")
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not set in environment or .env!")
        self.model_name = model_name
        self.api_key = GROQ_API_KEY

    def generate_from_prompt(self, prompt: str) -> str:
        """
        Reusable function to generate a response from any raw prompt using Groq.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }
        
        try:
            logger.info("Generating response via Groq API...")
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return text
            
        except requests.exceptions.HTTPError as e:
            error_msg = response.json().get("error", {}).get("message", str(e))
            logger.error(f"Groq API HTTP Error: {error_msg}")
            return f"Error generating response: {error_msg}"
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        RAG-specific answer generator with strict anti-hallucination prompting.
        """
        if not retrieved_chunks:
            logger.info("No contexts provided to LLM.")
            return "I don't have enough context in the provided documents to answer this question."

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.get('metadata', {}).get('source', 'Unknown Document')
            page_num = chunk.get('metadata', {}).get('page_number', 'N/A')
            context_parts.append(
                f"--- Document [{i+1}] (Source: {source}, Page: {page_num}) ---\n{chunk['text']}"
            )
            
        context_text = "\n\n".join(context_parts)

        prompt = f"""ROLE: You are an advanced AI Document Intelligence Agent. Your sole purpose is to analyze the provided Context Documents and provide a precise, accurate answer to the user's question.

STRICT RULES:
1. NO HALLUCINATION: Base your answer *strictly* on the Context Documents only.
2. NO OUTSIDE KNOWLEDGE: If the answer is not in the context, respond exactly with: "Insufficient data to answer this question accurately."
3. IN-TEXT CITATIONS: Cite facts using document IDs like [Document 1] or [Document 2].

OUTPUT FORMAT:
**Answer:**
[Your detailed answer with in-text citations]

**Sources:**
- **[Document X]**: [Source Name] (Page [Y])
  > "[Relevant quote from document]"

---
CONTEXT DOCUMENTS:
{context_text}

---
USER QUESTION: {query}"""

        return self.generate_from_prompt(prompt)
