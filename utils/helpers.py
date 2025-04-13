import hashlib
import re
import unicodedata
from urllib.parse import urlparse

def get_domain(url):
    """Extract domain from URL."""
    parsed_url = urlparse(url)
    return parsed_url.netloc

def normalize_text(text):
    """Normalize text by removing extra whitespace and normalizing Unicode."""
    if not text:
        return ""
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_document_id(url, content):
    """Generate a unique hash ID for a document based on URL and content."""
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return f"{url_hash[:10]}_{content_hash[:10]}"

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping chunks of specified size."""
    if not text or chunk_size <= 0:
        return []
    
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    
    return chunks

def is_valid_url(url):
    """Check if a URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False