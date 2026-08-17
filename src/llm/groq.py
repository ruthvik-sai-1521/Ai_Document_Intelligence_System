import time
import requests
from src.llm.base import BaseLLM
from src.core.config import GROQ_API_KEY
from src.core.logger import setup_logger

logger = setup_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class GroqLLM(BaseLLM):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        logger.info(f"Initializing GroqLLM client: {model_name}")
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not set in environment or .env!")
        self.model_name = model_name
        self.api_key = GROQ_API_KEY

    def generate(self, prompt: str) -> str:
        """Generate response from Groq completions REST endpoint with rate-limit retries."""
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
        
        max_retries = 3
        backoff = 5.0
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response via Groq API (Attempt {attempt+1}/{max_retries})...")
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries - 1:
                    logger.warning(f"Groq API Rate limit hit (429). Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                    
                response.raise_for_status()
                
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                return text
                
            except requests.exceptions.HTTPError as e:
                res = e.response
                status_code = res.status_code if res is not None else None
                if status_code == 429 and attempt < max_retries - 1:
                    logger.warning(f"Groq API HTTP 429 Rate limit hit. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                error_msg = str(e)
                if res is not None:
                    try:
                        error_msg = res.json().get("error", {}).get("message", str(e))
                    except Exception:
                        pass
                logger.error(f"Groq API HTTP Error: {error_msg}")
                return f"Error generating response: {error_msg}"
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return f"Error generating response: {str(e)}"

        return "Error generating response: Exceeded maximum retries."

