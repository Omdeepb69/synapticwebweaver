import cmd
import json
import os
from datetime import datetime

from pipeline.scraper import WebScraper
from pipeline.processor import DataProcessor
from pipeline.knowledge_base import KnowledgeBase
from models.llm_adapter import LLMAdapter
from models.fine_tuning import ModelFineTuner
from utils.logger import logger

class LLMCmd(cmd.Cmd):
    """Command-line interface for the Internet Learning LLM."""
    
    intro = '''
    =====================================================
    Internet Learning LLM Framework - Interactive Console
    Type "help" to see available commands
    =====================================================
    '''
    prompt = 'llm> '
    
    def __init__(self):
        """Initialize the CLI."""
        super().__init__()
        self.kb = KnowledgeBase()
        self.llm = LLMAdapter()
        self.fine_tuner = ModelFineTuner()
        self.current_fine_tuned_model = None
    
    def do_help(self, arg):
        """Show help message."""
        commands = {
            "scrape <url> [max_pages] [domain_only]": "Scrape web pages starting from URL",
            "process": "Process scraped data",
            "build_kb": "Build knowledge base from processed data",
            "query <text> [top_k]": "Query the knowledge base",
            "generate <prompt> [max_tokens] [temperature]": "Generate text using the LLM",
            "add_document <file_path> [url]": "Add a document to the knowledge base",
            "fine_tune [output_name] [epochs] [learning_rate]": "Fine-tune the LLM on collected data",
            "use_model <model_path>": "Use a fine-tuned model for generation",
            "stats": "Show knowledge base statistics",
            "exit": "Exit the program"
        }
        
        print("\nAvailable commands:")
        for cmd, desc in commands.items():
            print(f"  {cmd.ljust(40)} {desc}")
        print()
    
    def do_scrape(self, arg):
        """Scrape web pages starting from URL."""
        args = arg.split()
        if not args:
            print("Error: URL is required")
            return
            
        url = args[0]
        max_pages = int(args[1]) if len(args) > 1 else 10
        domain_only = args[2].lower() == "true" if len(args) > 2 else True
        
        print(f"Scraping {url} (max pages: {max_pages}, domain only: {domain_only})...")
        
        try:
            scraper = WebScraper(use_selenium=False)
            documents = scraper.crawl(url, max_pages=max_pages, domain_restrict=domain_only)
            scraper.close()
            
            print(f"Scraping completed. Collected {len(documents)} documents.")
        except Exception as e:
            print(f"Error during scraping: {e}")
    
    def do_process(self, arg):
        """Process scraped data."""
        try:
            processor = DataProcessor()
            processed_docs = processor.process_all()
            
            print(f"Processing completed. Processed {len(processed_docs)} documents.")
        except Exception as e:
            print(f"Error during processing: {e}")
    
    def do_build_kb(self, arg):
        """Build knowledge base from processed data."""
        try:
            kb_info = self.kb.build_from_processed_data()
            
            print(f"Knowledge base built with {kb_info['vector_count']} vectors.")
        except Exception as e:
            print(f"Error building knowledge base: {e}")
    
    def do_query(self, arg):
        """Query the knowledge base."""
        args = arg.split(maxsplit=1)
        if not args:
            print("Error: Query text is required")
            return
            
        query = args[0]
        top_k = int(args[1]) if len(args) > 1 else 5
        
        try:
            results = self.kb.query(query, top_k=top_k)
            
            print(f"\nQuery: {query}")
            print(f"Found {len(results)} results:")
            
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
                print(f"Document: {result.get('document', {}).get('url', 'Unknown')}")
                print(f"Chunk: {result['chunk'][:200]}...")
            
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
    
    def do_generate(self, arg):
        """Generate text using the LLM."""
        args = arg.split(maxsplit=2)
        if not args:
            print("Error: Prompt is required")
            return
            
        prompt = args[0]
        max_tokens = int(args[1]) if len(args) > 1 else 100
        temperature = float(args[2]) if len(args) > 2 else 0.7
        
        try:
            print(f"Generating with {'fine-tuned' if self.current_fine_tuned_model else 'base'} model...")
            
            if self.current_fine_tuned_model:
                output = self.fine_tuner.generate_from_fine_tuned(
                    self.current_fine_tuned_model,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                output = self.llm.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            print("\n--- Generated Text ---")
            print(output)
            
        except Exception as e:
            print(f"Error generating text: {e}")
    
    def do_add_document(self, arg):
        """Add a document to the knowledge base."""
        args = arg.split()
        if not args:
            print("Error: File path is required")
            return
            
        file_path = args[0]
        url = args[1] if len(args) > 1 else None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            doc_id = self.kb.add_new_document(text, url=url)
            
            print(f"Document added successfully. ID: {doc_id}")
            
        except Exception as e:
            print(f"Error adding document: {e}")
    
    def do_fine_tune(self, arg):
        """Fine-tune the LLM on collected data."""
        args = arg.split()
        output_name = args[0] if len(args) > 0 else None
        epochs = int(args[1]) if len(args) > 1 else 3
        learning_rate = float(args[2]) if len(args) > 2 else 5e-5
        
        try:
            print("Starting fine-tuning process...")
            model_path = self.fine_tuner.fine_tune(
                output_name=output_name,
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            if model_path:
                print(f"Fine-tuning completed successfully. Model saved to: {model_path}")
                self.current_fine_tuned_model = model_path
            else:
                print("Fine-tuning failed.")
                
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
    
    def do_use_model(self, arg):
        """Use a fine-tuned model for generation."""
        if not arg:
            print("Error: Model path is required")
            return
            
        model_path = arg.strip()
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
            
        self.current_fine_tuned_model = model_path
        print(f"Now using model: {model_path}")
    
    def do_stats(self, arg):
        """Show knowledge base statistics."""
        try:
            stats = self.kb.get_stats()
            
            print("\n--- Knowledge Base Statistics ---")
            print(f"Documents: {stats['document_count']}")
            print(f"Vectors: {stats['vector_count']}")
            print(f"Text chunks: {stats['chunk_count']}")
            print(f"Domains: {', '.join(stats['domains'][:5])}{'...' if len(stats['domains']) > 5 else ''}")
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
    
    def do_exit(self, arg):
        """Exit the program."""
        print("Exiting Internet Learning LLM Framework. Goodbye!")
        return True

def start_cli():
    """Start the command-line interface."""
    LLMCmd().cmdloop()

if __name__ == "__main__":
    start_cli()