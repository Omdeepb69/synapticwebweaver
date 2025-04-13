import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, KNOWLEDGE_BASE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Scraping settings
MAX_PAGES_PER_DOMAIN = 100
SCRAPING_DELAY = 1.5  # seconds
USER_AGENT = "Mozilla/5.0 (Educational Research Bot)"
ROBOTS_TXT_RESPECT = True

# Knowledge base settings
VECTOR_DIMENSION = 768  # For sentence-transformers
KB_CHUNK_SIZE = 512  # Size of text chunks for knowledge base

# Model settings
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"  # Example model
DEVICE = "cuda" if os.environ.get("USE_GPU", "False").lower() == "true" else "cpu"

# Training settings
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_EPOCHS = 3

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000