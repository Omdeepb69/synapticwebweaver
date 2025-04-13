import argparse
import os
import sys

from app.api import start_api
from app.interface import start_cli
from pipeline.scraper import WebScraper
from pipeline.processor import DataProcessor
from pipeline.knowledge_base import KnowledgeBase
from models.llm_adapter import LLMAdapter
from models.fine_tuning import ModelFineTuner
from utils.logger import logger
from config.settings import KNOWLEDGE_BASE_DIR

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Internet Learning LLM Framework")
    
    # Add arguments
    parser.add_argument('--mode', type=str, choices=['cli', 'api', 'scrape', 'process', 'build_kb', 'fine_tune'],
                        default='cli', help='Operation mode')
    parser.add_argument('--url', type=str, help='URL to start scraping from')
    parser.add_argument('--max_pages', type=int, default=10, help='Maximum number of pages to scrape')
    parser.add_argument('--domain_only', action='store_true', help='Restrict scraping to starting domain')
    parser.add_argument('--epochs', type=int, default=3, help='Number of fine-tuning epochs')
    parser.add_argument('--model_output', type=str, help='Output name for fine-tuned model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'cli':
        # Start CLI
        start_cli()
        
    elif args.mode == 'api':
        # Start API server
        start_api()
        
    elif args.mode == 'scrape':
        # Scrape web pages
        if not args.url:
            logger.error("URL is required for scrape mode")
            parser.print_help()
            sys.exit(1)
            
        logger.info(f"Starting scraping from {args.url}")
        scraper = WebScraper(use_selenium=False)
        documents = scraper.crawl(
            args.url,
            max_pages=args.max_pages,
            domain_restrict=args.domain_only
        )
        scraper.close()
        
        logger.info(f"Scraping completed. Collected {len(documents)} documents")
        
    elif args.mode == 'process':
        # Process scraped data
        logger.info("Starting data processing")
        processor = DataProcessor()
        processed_docs = processor.process_all()
        
        logger.info(f"Processing completed. Processed {len(processed_docs)} documents")
        
    elif args.mode == 'build_kb':
        # Build knowledge base
        logger.info("Building knowledge base")
        kb = KnowledgeBase()
        kb_info = kb.build_from_processed_data()
        
        logger.info(f"Knowledge base built with {kb_info['vector_count']} vectors")
        
    elif args.mode == 'fine_tune':
        # Fine-tune model
        logger.info("Starting fine-tuning process")
        fine_tuner = ModelFineTuner()
        model_path = fine_tuner.fine_tune(
            output_name=args.model_output,
            epochs=args.epochs
        )
        
        if model_path:
            logger.info(f"Fine-tuning completed. Model saved to: {model_path}")
        else:
            logger.error("Fine-tuning failed")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    
    # Run main
    main()