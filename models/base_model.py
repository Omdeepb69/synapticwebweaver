from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all models."""
    
    @abstractmethod
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """
        Generate text based on a prompt.
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated text
        """
        pass
    
    @abstractmethod
    def embed(self, text):
        """
        Get text embeddings.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Text embedding
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save model.
        
        Args:
            path (str): Save path
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load model.
        
        Args:
            path (str): Load path
        """
        pass