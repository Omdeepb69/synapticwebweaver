import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import faiss

from config.settings import PROCESSED_DATA_DIR, KNOWLEDGE_BASE_DIR, VECTOR_DIMENSION, DEVICE
from utils.logger import logger

class TextVectorizer:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", max_workers=4):
        """
        Initialize the text vectorizer.
        
        Args:
            model_name (str): Name of the SentenceTransformer model
            max_workers (int): Maximum number of worker threads
        """
        self.max_workers = max_workers
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        logger.info(f"Model loaded, using device: {self.model.device}")
    
    def _vectorize_chunks(self, chunks):
        """
        Vectorize a list of text chunks.
        
        Args:
            chunks (list): List of text chunks
            
        Returns:
            numpy.ndarray: Array of vectors
        """
        if not chunks:
            return np.array([])
            
        # Encode text chunks to vectors
        with torch.no_grad():
            embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        return embeddings
    
    def _process_document(self, doc_path):
        """
        Process a single document to generate vectors.
        
        Args:
            doc_path (str): Path to processed document
            
        Returns:
            tuple: (document ID, vectors, chunks)
        """
        try:
            # Load processed document
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            
            doc_id = doc['id']
            chunks = doc['chunks']
            
            if not chunks:
                return doc_id, np.array([]), []
            
            # Vectorize chunks
            vectors = self._vectorize_chunks(chunks)
            
            return doc_id, vectors, chunks
            
        except Exception as e:
            logger.error(f"Error vectorizing document {doc_path}: {e}")
            return None, np.array([]), []
    
    def vectorize_all(self):
        """
        Vectorize all processed documents.
        
        Returns:
            dict: Document vectors and metadata
        """
        # Find all processed document files
        processed_files = []
        for root, _, files in os.walk(PROCESSED_DATA_DIR):
            for file in files:
                if file.endswith('_processed.json'):
                    processed_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(processed_files)} documents to vectorize")
        
        # Process documents in parallel
        all_vectors = []
        all_chunks = []
        doc_ids = []
        chunk_to_doc = []  # Maps each chunk index to its document ID
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Vectorize documents with progress bar
            for doc_id, vectors, chunks in tqdm(
                executor.map(self._process_document, processed_files),
                total=len(processed_files),
                desc="Vectorizing documents"
            ):
                if doc_id and vectors.size > 0:
                    doc_ids.append(doc_id)
                    all_vectors.append(vectors)
                    
                    # Track which document each chunk belongs to
                    chunk_start_idx = len(all_chunks)
                    all_chunks.extend(chunks)
                    chunk_to_doc.extend([doc_id] * len(chunks))
        
        # Combine all vectors into a single array
        combined_vectors = np.vstack(all_vectors) if all_vectors else np.array([])
        
        logger.info(f"Generated {len(combined_vectors)} vectors from {len(doc_ids)} documents")
        
        # Create a FAISS index for fast similarity search
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        if combined_vectors.size > 0:
            index.add(combined_vectors.astype('float32'))
        
        # Create vector database info
        vector_db = {
            'documents': doc_ids,
            'chunks': all_chunks,
            'chunk_to_doc': chunk_to_doc,
            'vector_count': len(combined_vectors)
        }
        
        # Save index and metadata
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
        faiss.write_index(index, os.path.join(KNOWLEDGE_BASE_DIR, "vector_index.faiss"))
        
        with open(os.path.join(KNOWLEDGE_BASE_DIR, "vector_db_metadata.json"), 'w') as f:
            json.dump(vector_db, f, indent=2)
        
        return vector_db
    
    def query(self, text, top_k=5):
        """
        Query the vector database for similar chunks.
        
        Args:
            text (str): Query text
            top_k (int): Number of results to return
            
        Returns:
            list: Top similar chunks with metadata
        """
        # Check if index exists
        index_path = os.path.join(KNOWLEDGE_BASE_DIR, "vector_index.faiss")
        metadata_path = os.path.join(KNOWLEDGE_BASE_DIR, "vector_db_metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.error("Vector database not found. Run vectorize_all first.")
            return []
        
        # Load index and metadata
        index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Vectorize query
        query_vector = self.model.encode([text])[0].reshape(1, -1).astype('float32')
        
        # Search for similar vectors
        distances, indices = index.search(query_vector, min(top_k, index.ntotal))
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(metadata['chunks']):
                continue
                
            doc_id = metadata['chunk_to_doc'][idx]
            results.append({
                'chunk': metadata['chunks'][idx],
                'document_id': doc_id,
                'score': float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
            })
        
        return results