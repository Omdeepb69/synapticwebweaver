from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import uvicorn

from pipeline.scraper import WebScraper
from pipeline.processor import DataProcessor
from pipeline.knowledge_base import KnowledgeBase
from models.llm_adapter import LLMAdapter
from models.fine_tuning import ModelFineTuner
from config.settings import API_HOST, API_PORT
from utils.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="Internet Learning LLM",
    description="An API for an LLM that learns from internet data",
    version="0.1.0"
)

# Initialize components
knowledge_base = KnowledgeBase()
llm = LLMAdapter()
fine_tuner = ModelFineTuner()

# Define request/response models
class ScrapeRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to start crawling from")
    max_pages: Optional[int] = Field(10, description="Maximum number of pages to crawl")
    domain_restrict: Optional[bool] = Field(True, description="Whether to restrict crawling to the starting domain")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    top_k: Optional[int] = Field(5, description="Number of results to return")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: Optional[int] = Field(100, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    model_path: Optional[str] = Field(None, description="Path to a fine-tuned model (if any)")

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="Document text")
    url: Optional[str] = Field(None, description="Document URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FineTuneRequest(BaseModel):
    output_name: Optional[str] = Field(None, description="Name for the fine-tuned model")
    epochs: Optional[int] = Field(3, description="Number of training epochs")
    learning_rate: Optional[float] = Field(5e-5, description="Learning rate")
    batch_size: Optional[int] = Field(8, description="Batch size")

# API routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {"message": "Internet Learning LLM API", "docs": "/docs"}

@app.post("/scrape")
async def scrape(request: ScrapeRequest):
    """Scrape web pages starting from a URL."""
    try:
        # Initialize web scraper
        scraper = WebScraper(use_selenium=False)
        
        # Crawl pages
        documents = scraper.crawl(
            str(request.url),
            max_pages=request.max_pages,
            domain_restrict=request.domain_restrict
        )
        
        # Process data
        processor = DataProcessor()
        processor.process_all()
        
        # Update knowledge base
        kb_info = knowledge_base.build_from_processed_data()
        
        # Clean up
        scraper.close()
        
        return {
            "message": "Scraping completed successfully",
            "documents_scraped": len(documents),
            "knowledge_base_stats": knowledge_base.get_stats()
        }
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.post("/query")
async def query(request: QueryRequest):
    """Query the knowledge base."""
    try:
        results = knowledge_base.query(request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text using the LLM."""
    try:
        if request.model_path:
            # Use fine-tuned model
            output = fine_tuner.generate_from_fine_tuned(
                request.model_path,
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        else:
            # Use base model
            output = llm.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        
        return {
            "prompt": request.prompt,
            "generated_text": output
        }
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/add_document")
async def add_document(request: AddDocumentRequest):
    """Add a document to the knowledge base."""
    try:
        doc_id = knowledge_base.add_new_document(
            request.text,
            url=request.url,
            metadata=request.metadata
        )
        
        return {
            "message": "Document added successfully",
            "document_id": doc_id
        }
    except Exception as e:
        logger.error(f"Error in add_document endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Adding document failed: {str(e)}")

@app.post("/fine_tune")
async def fine_tune(request: FineTuneRequest):
    """Fine-tune the LLM on the collected data."""
    try:
        # Check if knowledge base has data
        kb_stats = knowledge_base.get_stats()
        if kb_stats["vector_count"] == 0:
            raise HTTPException(status_code=400, detail="No data in knowledge base for fine-tuning")
        
        # Fine-tune model
        model_path = fine_tuner.fine_tune(
            output_name=request.output_name,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size
        )
        
        if not model_path:
            raise HTTPException(status_code=500, detail="Fine-tuning failed")
        
        return {
            "message": "Fine-tuning completed successfully",
            "model_path": model_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fine_tune endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")

@app.get("/kb_stats")
async def kb_stats():
    """Get knowledge base statistics."""
    try:
        stats = knowledge_base.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error in kb_stats endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Start server if executed directly
def start_api():
    """Start the FastAPI server."""
    uvicorn.run(app, host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    start_api()