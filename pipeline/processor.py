import os
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, KB_CHUNK_SIZE
from utils.logger import logger
from utils.helpers import chunk_text, normalize_text

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DataProcessor:
    def __init__(self, max_workers=4):
        """
        Initialize the data processor.
        
        Args:
            max_workers (int): Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.stop_words = set(stopwords.words('english'))
    
    def _clean_text(self, text):
        """
        Clean text by removing noise and normalizing.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Normalize whitespace
        text = normalize_text(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_named_entities(self, text):
        """
        Extract named entities from text using spaCy.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Named entities by category
        """
        if not text:
            return {}
            
        doc = nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
            
        return entities
    
    def _extract_keywords(self, text, top_n=10):
        """
        Extract top keywords from text based on frequency.
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to extract
            
        Returns:
            list: Top keywords
        """
        if not text:
            return []
            
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() 
                  and token not in self.stop_words and len(token) > 2]
        
        # Count frequency
        freq_dist = nltk.FreqDist(tokens)
        
        # Get top keywords
        return [word for word, _ in freq_dist.most_common(top_n)]
    
    def _process_document(self, doc_path):
        """
        Process a single document.
        
        Args:
            doc_path (str): Path to document metadata file
            
        Returns:
            dict: Processed document
        """
        try:
            # Load metadata
            with open(doc_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            doc_id = metadata['id']
            doc_dir = os.path.dirname(doc_path)
            
            # Load text
            text_path = os.path.join(doc_dir, f"{doc_id}_text.txt")
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Process text
            entities = self._extract_named_entities(cleaned_text[:100000])  # Limit size for NER
            keywords = self._extract_keywords(cleaned_text)
            
            # Split into sentences
            sentences = sent_tokenize(cleaned_text)
            
            # Create chunks for the knowledge base
            chunks = chunk_text(cleaned_text, KB_CHUNK_SIZE)
            
            # Processed document
            processed_doc = {
                **metadata,
                'cleaned_text': cleaned_text,
                'entities': entities,
                'keywords': keywords,
                'sentence_count': len(sentences),
                'chunks': chunks,
                'word_count': len(cleaned_text.split())
            }
            
            # Save processed document
            output_dir = os.path.join(PROCESSED_DATA_DIR, metadata['domain'].replace('.', '_'))
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"{doc_id}_processed.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_doc, f, indent=2)
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return None
    
    def process_all(self):
        """
        Process all raw documents.
        
        Returns:
            list: List of processed documents
        """
        # Find all metadata files
        meta_files = []
        for root, _, files in os.walk(RAW_DATA_DIR):
            for file in files:
                if file.endswith('_meta.json'):
                    meta_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(meta_files)} documents to process")
        
        # Process documents in parallel
        processed_docs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process documents with progress bar
            for doc in tqdm(executor.map(self._process_document, meta_files), 
                           total=len(meta_files), desc="Processing documents"):
                if doc:
                    processed_docs.append(doc)
        
        logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs