# Utility functions and logging helpers
import logging
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)

def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA-256 hash of a file for change detection.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

def get_file_metadata(filepath: str) -> dict:
    """
    Extract comprehensive metadata from a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary containing file metadata
    """
    path = Path(filepath)
    stats = path.stat()
    
    return {
        "filename": path.name,
        "filepath": str(path.absolute()),
        "file_type": path.suffix.lower(),
        "file_size": stats.st_size,
        "created_date": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified_date": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        "file_hash": compute_file_hash(filepath)
    }

def validate_environment() -> tuple[str, str, str]:
    """
    Validate required environment variables are present.
    
    Returns:
        Tuple of (openai_key, qdrant_url, qdrant_key)
        
    Raises:
        EnvironmentError: If required variables are missing
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    missing = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not qdrant_key:
        missing.append("QDRANT_API_KEY")
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please ensure your .env file contains all required keys."
        )
    
    return openai_key, qdrant_url, qdrant_key

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"