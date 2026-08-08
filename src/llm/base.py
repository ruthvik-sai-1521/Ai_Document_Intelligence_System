from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text completion response for the given prompt.
        
        Args:
            prompt: The formatted prompt string sent to the model.
            
        Returns:
            The generated text response string.
        """
        pass
