import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger with file and console handlers."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Setup main logger
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = setup_logger("internet_learning_llm", log_file)