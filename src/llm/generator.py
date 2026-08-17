from typing import List, Dict, Any, Optional
try:
    from src.llm.base import BaseLLM
    from src.llm.groq import GroqLLM
    from src.llm.gemini import GeminiLLM
    from src.core.config import GROQ_API_KEY, GEMINI_API_KEY
    from src.core.logger import setup_logger
except ImportError:
    from llm.base import BaseLLM
    from llm.groq import GroqLLM
    from llm.gemini import GeminiLLM
    from core.config import GROQ_API_KEY, GEMINI_API_KEY
    from core.logger import setup_logger

logger = setup_logger(__name__)

class LLMGenerator:
    def __init__(self, model_name: Optional[str] = None, provider: Optional[str] = None, llm_engine: Optional[BaseLLM] = None):
        """
        Initialize the LLM Generator wrapper.
        
        Args:
            model_name: Optional model override.
            provider:   'groq' or 'gemini'. Defaults to 'groq' if GROQ_API_KEY is set, else 'gemini'.
            llm_engine: Optional BaseLLM engine (DIP dependency injection).
        """
        if llm_engine:
            self.llm_engine = llm_engine
        elif provider == "gemini" or (not GROQ_API_KEY and GEMINI_API_KEY):
            self.llm_engine = GeminiLLM(model_name=model_name or "gemini-1.5-flash")
        else:
            self.llm_engine = GroqLLM(model_name=model_name or "llama-3.3-70b-versatile")

    def generate_from_prompt(self, prompt: str) -> str:
        """
        Generates response using the injected LLM engine.
        """
        return self.llm_engine.generate(prompt)

    def rewrite_query(self, query: str, history: List[Dict[str, Any]]) -> str:
        """
        Reformulates follow-up queries based on the conversation history to make them standalone queries.
        """
        if not history:
            return query

        history_lines = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_lines.append(f"{role.capitalize()}: {content}")
        history_text = "\n".join(history_lines)

        prompt = f"""You are an expert search assistant. Your job is to rewrite the follow-up query into a standalone, fully-contextualized search query.
Use the conversation history to replace any pronouns or implicit references (like 'they', 'it', 'them', 'that', 'there', etc.) with the correct noun/subject.

Rules:
1. Return ONLY the final rewritten search query.
2. Do not include any conversational filler, explanations, markdown formatting, or notes.
3. If the query is already standalone and does not need any context from the history, return the query unchanged.

CONVERSATION HISTORY:
{history_text}

FOLLOW-UP QUERY: {query}
STANDALONE REWRITTEN QUERY:"""

        try:
            rewritten = self.generate_from_prompt(prompt).strip()
            if (rewritten.startswith('"') and rewritten.endswith('"')) or (rewritten.startswith("'") and rewritten.endswith("'")):
                rewritten = rewritten[1:-1].strip()
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}")
            return query

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        RAG-specific answer generator with strict anti-hallucination prompting and context retention.
        """
        if not retrieved_chunks:
            logger.info("No contexts provided to LLM.")
            return "I don't have enough context in the provided documents to answer this question."

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk.get('metadata', {})
            source = meta.get('source', 'Unknown Document')
            if meta.get('source_type') == 'youtube':
                video_title = meta.get('video_title', 'YouTube Video')
                time_range = meta.get('formatted_time_range', '')
                context_parts.append(
                    f"--- Document [{i+1}] (YouTube Video: '{video_title}', Timestamp: {time_range}, URL: {source}) ---\n{chunk['text']}"
                )
            else:
                page_num = meta.get('page_number', 'N/A')
                context_parts.append(
                    f"--- Document [{i+1}] (Source: {source}, Page: {page_num}) ---\n{chunk['text']}"
                )
            
        context_text = "\n\n".join(context_parts)

        # Reconstruct conversation history text for context retention
        history_text = ""
        if history:
            history_lines = []
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_lines.append(f"{role.capitalize()}: {content}")
            history_text = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n---\n"

        prompt = f"""ROLE: You are an advanced AI Document Intelligence Agent. Your sole purpose is to analyze the provided Context Documents and provide a precise, accurate answer to the user's question.

STRICT RULES:
1. NO HALLUCINATION: Base your answer *strictly* on the Context Documents only.
2. NO OUTSIDE KNOWLEDGE: If the answer is not in the context, respond exactly with: "Insufficient data to answer this question accurately."
3. IN-TEXT CITATIONS: Cite facts using document IDs like [Document 1] or [Document 2].

OUTPUT FORMAT:
**Answer:**
[Your detailed answer with in-text citations]

**Sources:**
- **[Document X]**: [Source Name / Video Title] (Page [Y] / Timestamp [MM:SS])
  > "[Relevant quote from document]"

---
{history_text}CONTEXT DOCUMENTS:
{context_text}

---
USER QUESTION: {query}"""

        return self.generate_from_prompt(prompt)
