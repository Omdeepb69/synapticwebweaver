import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

from config.settings import MODEL_NAME, DEVICE
from utils.logger import logger
from models.base_model import BaseModel

class LLMAdapter(BaseModel):
    """Adapter for pre-trained language models."""
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the LLM adapter.
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to use (cuda/cpu)
        """
        self.model_name = model_name or MODEL_NAME
        self.device = device or DEVICE
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing LLM adapter with model: {self.model_name}")
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model: {self.model_name} on {self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cuda":
                self.model.to(self.device)
            
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # Clean up GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        gc.collect()
    
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
        # Ensure model is loaded
        self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to(self.device)
            
        # Set generation parameters
        gen_kwargs = {
            "max_length": inputs["input_ids"].shape[1] + max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.92,
            "top_k": 50,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
            
        # Decode and return only the new tokens
        generated_text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_text
    
    def embed(self, text):
        """
        Get text embeddings from the model's hidden states.
        Note: This is a simple approximation. For proper embeddings,
        use a dedicated embedding model.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Text embedding (average of last hidden layer)
        """
        # Ensure model is loaded
        self.load_model()
        
        inputs = self.tokenizer(text, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to(self.device)
            
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Use the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Calculate mean of token embeddings (excluding padding)
        attention_mask = inputs["attention_mask"]
        mean_embedding = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        
        # Convert to list and return
        return mean_embedding[0].cpu().numpy().tolist()
    
    def save(self, path):
        """
        Save model and tokenizer.
        
        Args:
            path (str): Save path
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Cannot save: model or tokenizer not loaded")
            return
            
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load(self, path):
        """
        Load model and tokenizer from local path.
        
        Args:
            path (str): Load path
        """
        try:
            # Unload existing model if any
            self.unload_model()
            
            logger.info(f"Loading tokenizer from {path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            logger.info(f"Loading model from {path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cuda":
                self.model.to(self.device)
                
            logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise