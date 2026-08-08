from typing import List, Dict, Any
from src.llm.base import BaseLLM
from src.llm.groq import GroqLLM
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class LLMGenerator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", llm_engine: BaseLLM = None):
        """
        Initialize the LLM Generator wrapper.
        
        Args:
            model_name: Default model to use if instantiating GroqLLM.
            llm_engine: Optional BaseLLM engine (DIP dependency injection).
        """
        if llm_engine:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = GroqLLM(model_name=model_name)

    def generate_from_prompt(self, prompt: str) -> str:
        """
        Generates response using the injected LLM engine.
        """
        return self.llm_engine.generate(prompt)

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
