import os
import json
import faiss
import numpy as np
from datetime import datetime

from config.settings import KNOWLEDGE_BASE_DIR, PROCESSED_DATA_DIR
from utils.logger import logger
from pipeline.vectorizer import TextVectorizer

class KnowledgeBase:
    def __init__(self, vectorizer=None):
        """
        Initialize the knowledge base.
        
        Args:
            vectorizer (TextVectorizer): Text vectorizer instance
        """
        self.vectorizer = vectorizer or TextVectorizer()
        self.index_path = os.path.join(KNOWLEDGE_BASE_DIR, "vector_index.faiss")
        self.metadata_path = os.path.join(KNOWLEDGE_BASE_DIR, "vector_db_metadata.json")
        self.doc_info_path = os.path.join(KNOWLEDGE_BASE_DIR, "document_info.json")
        
        # Load existing knowledge base if it exists
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load the knowledge base from disk."""
        # Reset state
        self.index = None
        self.metadata = {'documents': [], 'chunks': [], 'chunk_to_doc': [], 'vector_count': 0}
        self.document_info = {}
        
        # Check if knowledge base exists
        if (os.path.exists(self.index_path) and 
            os.path.exists(self.metadata_path)):
            try:
                # Load index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Load document info if exists
                if os.path.exists(self.doc_info_path):
                    with open(self.doc_info_path, 'r') as f:
                        self.document_info = json.load(f)
                
                logger.info(f"Loaded knowledge base with {self.metadata['vector_count']} vectors " +
                           f"from {len(self.metadata['documents'])} documents")
                
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
                # Reset state on error
                self.index = None
                self.metadata = {'documents': [], 'chunks': [], 'chunk_to_doc': [], 'vector_count': 0}
                self.document_info = {}
    
    def build_from_processed_data(self):
        """Build the knowledge base from processed data."""
        # Vectorize all documents
        vector_db = self.vectorizer.vectorize_all()
        
        # Load document info
        self._load_document_info()
        
        # Reload knowledge base
        self.load_knowledge_base()
        
        return vector_db
    
    def _load_document_info(self):
        """Load information about all processed documents."""
        document_info = {}
        
        # Find all processed document files
        for root, _, files in os.walk(PROCESSED_DATA_DIR):
            for file in files:
                if file.endswith('_processed.json'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            
                        doc_id = doc['id']
                        document_info[doc_id] = {
                            'id': doc_id,
                            'url': doc.get('url', ''),
                            'domain': doc.get('domain', ''),
                            'crawl_date': doc.get('crawl_date', ''),
                            'word_count': doc.get('word_count', 0),
                            'keywords': doc.get('keywords', []),
                            'entities': doc.get('entities', {})
                        }
                    except Exception as e:
                        logger.error(f"Error loading document info from {file_path}: {e}")
        
        # Save document info
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
        with open(self.doc_info_path, 'w') as f:
            json.dump(document_info, f, indent=2)
        
        self.document_info = document_info
        logger.info(f"Loaded info for {len(document_info)} documents")
    
    def query(self, text, top_k=5):
        """
        Query the knowledge base.
        
        Args:
            text (str): Query text
            top_k (int): Number of results to return
            
        Returns:
            list: Top similar chunks with metadata
        """
        if not self.index:
            logger.error("Knowledge base not loaded. Build or load it first.")
            return []
        
        # Get results from vectorizer
        results = self.vectorizer.query(text, top_k)
        
        # Enhance results with document info
        for result in results:
            doc_id = result['document_id']
            if doc_id in self.document_info:
                result['document'] = self.document_info[doc_id]
        
        return results
    
    def add_new_document(self, text, url=None, metadata=None):
        """
        Add a new document to the knowledge base.
        
        Args:
            text (str): Document text
            url (str): Document URL
            metadata (dict): Additional metadata
            
        Returns:
            str: Document ID
        """
        from utils.helpers import generate_document_id, chunk_text, get_domain
        from pipeline.processor import DataProcessor
        
        if not text:
            return None
            
        # Generate document ID
        doc_id = generate_document_id(url or "local_document", text)
        
        # Process the document
        processor = DataProcessor()
        cleaned_text = processor._clean_text(text)
        chunks = chunk_text(cleaned_text)
        entities = processor._extract_named_entities(cleaned_text[:100000])
        keywords = processor._extract_keywords(cleaned_text)
        
        # Create document info
        domain = get_domain(url) if url else "local"
        doc_info = {
            'id': doc_id,
            'url': url or f"local:{doc_id}",
            'domain': domain,
            'crawl_date': datetime.now().isoformat(),
            'word_count': len(cleaned_text.split()),
            'keywords': keywords,
            'entities': entities,
            **(metadata or {})
        }
        
        # Update document info
        self.document_info[doc_id] = doc_info
        with open(self.doc_info_path, 'w') as f:
            json.dump(self.document_info, f, indent=2)
        
        # Vectorize chunks
        vectors = self.vectorizer._vectorize_chunks(chunks)
        
        # Add to index
        if vectors.size > 0 and self.index:
            self.index.add(vectors.astype('float32'))
            
            # Update metadata
            start_idx = self.metadata['vector_count']
            self.metadata['documents'].append(doc_id)
            self.metadata['chunks'].extend(chunks)
            self.metadata['chunk_to_doc'].extend([doc_id] * len(chunks))
            self.metadata['vector_count'] += len(vectors)
            
            # Save updated index and metadata
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        
        return doc_id
    
    def get_document_info(self, doc_id):
        """Get information about a document."""
        return self.document_info.get(doc_id)
    
    def get_stats(self):
        """Get statistics about the knowledge base."""
        return {
            'document_count': len(self.document_info),
            'vector_count': self.metadata.get('vector_count', 0),
            'chunk_count': len(self.metadata.get('chunks', [])),
            'domains': list(set(doc.get('domain', '') for doc in self.document_info.values()))
        }