import requests
from src.llm.base import BaseLLM
from src.core.config import GEMINI_API_KEY
from src.core.logger import setup_logger

logger = setup_logger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        logger.info(f"Initializing GeminiLLM client: {model_name}")
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set in environment or .env!")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Generate response from Google Gemini REST endpoint."""
        if not self.api_key:
            return "Error generating response: GEMINI_API_KEY is missing."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048
            }
        }
        
        try:
            logger.info("Generating response via Google Gemini API...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "Error generating response: Empty candidate returned by Gemini API."
            
        except requests.exceptions.HTTPError as e:
            try:
                error_msg = response.json().get("error", {}).get("message", str(e))
            except Exception:
                error_msg = str(e)
            logger.error(f"Gemini API HTTP Error: {error_msg}")
            return f"Error generating response: {error_msg}"
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
