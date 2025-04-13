import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm

from config.settings import (
    PROCESSED_DATA_DIR, KNOWLEDGE_BASE_DIR,
    MODEL_NAME, DEVICE, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS
)
from utils.logger import logger

class TextDataset(Dataset):
    """Custom dataset for text fine-tuning."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of texts
            tokenizer: Tokenizer for encoding
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
        logger.info(f"Created dataset with {len(texts)} texts")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert from batch format (1, seq_len) to (seq_len)
        item = {
            key: val.squeeze(0) for key, val in encodings.items()
        }
        
        return item

class ModelFineTuner:
    """Class for fine-tuning language models on the collected data."""
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to use (cuda/cpu)
        """
        self.model_name = model_name or MODEL_NAME
        self.device = device or DEVICE
        self.model = None
        self.tokenizer = None
        
        # Create directory for fine-tuned models
        self.output_dir = os.path.join(KNOWLEDGE_BASE_DIR, "fine_tuned_models")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_training_data(self):
        """
        Load training data from processed documents.
        
        Returns:
            list: List of texts for training
        """
        texts = []
        
        # Load knowledge base metadata
        kb_meta_path = os.path.join(KNOWLEDGE_BASE_DIR, "vector_db_metadata.json")
        if not os.path.exists(kb_meta_path):
            logger.error("Knowledge base metadata not found")
            return texts
            
        with open(kb_meta_path, 'r') as f:
            kb_meta = json.load(f)
            
        # Use chunks as training data
        texts = kb_meta.get('chunks', [])
        
        logger.info(f"Loaded {len(texts)} text chunks for training")
        return texts
    
    def fine_tune(self, output_name=None, epochs=None, learning_rate=None, batch_size=None):
        """
        Fine-tune the language model on collected data.
        
        Args:
            output_name (str): Name for the fine-tuned model
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            
        Returns:
            str: Path to the fine-tuned model
        """
        # Set parameters
        epochs = epochs or NUM_EPOCHS
        learning_rate = learning_rate or LEARNING_RATE
        batch_size = batch_size or BATCH_SIZE
        output_name = output_name or f"fine_tuned_{self.model_name.split('/')[-1]}_{epochs}epochs"
        
        # Output directory for this run
        run_output_dir = os.path.join(self.output_dir, output_name)
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Load model and tokenizer
        logger.info(f"Loading base model and tokenizer: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Ensure the tokenizer has a padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Error loading model for fine-tuning: {e}")
            return None
        
        # Load training data
        texts = self._load_training_data()
        if not texts:
            logger.error("No training data available")
            return None
            
        # Create dataset
        dataset = TextDataset(texts, self.tokenizer)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We want causal language modeling, not masked
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=2,
            logging_dir=os.path.join(run_output_dir, "logs"),
            logging_steps=100,
            fp16=self.device == "cuda",
            report_to="none"  # Disable wandb, etc.
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )
        
        # Train the model
        logger.info(f"Starting fine-tuning process with {len(texts)} examples")
        try:
            trainer.train()
            
            # Save the final model
            final_model_path = os.path.join(run_output_dir, "final_model")
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            logger.info(f"Fine-tuning completed. Model saved to {final_model_path}")
            return final_model_path
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return None
    
    def generate_from_fine_tuned(self, model_path, prompt, max_tokens=100, temperature=0.7):
        """
        Generate text from a fine-tuned model.
        
        Args:
            model_path (str): Path to the fine-tuned model
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated text
        """
        try:
            # Load the fine-tuned model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                model = model.to(self.device)
                
            # Generate text
            inputs = tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.device)
                
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating from fine-tuned model: {e}")
            return f"Error generating text: {str(e)}"