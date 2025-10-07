# Manages incremental embeddings and file hash caching
import json
from pathlib import Path
from typing import Dict, Optional, Set
from datetime import datetime

from utils import get_logger, compute_file_hash

logger = get_logger(__name__)

class EmbeddingCacheManager:
    """
    Manages local cache for incremental embeddings.
    Tracks file hashes to detect changes and avoid re-embedding.
    """
    
    def __init__(self, cache_file: str = ".unified_embedding.meta.json"):
        """
        Initialize cache manager.
        
        Args:
            cache_file: Path to cache metadata file
        """
        self.cache_file = Path(cache_file)
        self.cache_data = self._load_cache()
        logger.info(f"Initialized cache manager (file={cache_file})")
    
    def _load_cache(self) -> Dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded cache with {len(data.get('files', {}))} files")
                    return data
            except Exception as e:
                logger.warning(f"Error loading cache, starting fresh: {e}")
                return {"files": {}, "collections": {}}
        else:
            logger.info("No existing cache found, starting fresh")
            return {"files": {}, "collections": {}}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
            logger.debug("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def has_file_changed(self, filepath: str) -> bool:
        """
        Check if file has changed since last embedding.
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if file is new or modified, False if unchanged
        """
        current_hash = compute_file_hash(filepath)
        cached_hash = self.cache_data['files'].get(filepath, {}).get('hash')
        
        if cached_hash is None:
            logger.info(f"File is new: {filepath}")
            return True
        
        if current_hash != cached_hash:
            logger.info(f"File has changed: {filepath}")
            return True
        
        logger.info(f"File unchanged: {filepath}")
        return False
    
    def mark_file_embedded(
        self,
        filepath: str,
        collection_name: str,
        chunk_count: int,
        metadata: Dict
    ):
        """
        Mark a file as embedded in cache.
        
        Args:
            filepath: Path to the file
            collection_name: Name of the collection
            chunk_count: Number of chunks created
            metadata: File metadata
        """
        file_hash = compute_file_hash(filepath)
        
        self.cache_data['files'][filepath] = {
            'hash': file_hash,
            'collection': collection_name,
            'chunk_count': chunk_count,
            'embedded_date': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        # Track collection
        if collection_name not in self.cache_data['collections']:
            self.cache_data['collections'][collection_name] = {
                'created_date': datetime.now().isoformat(),
                'files': [],
                'total_chunks': 0
            }
        
        if filepath not in self.cache_data['collections'][collection_name]['files']:
            self.cache_data['collections'][collection_name]['files'].append(filepath)
        
        self.cache_data['collections'][collection_name]['total_chunks'] += chunk_count
        self.cache_data['collections'][collection_name]['last_updated'] = datetime.now().isoformat()
        
        self._save_cache()
        logger.info(f"Marked file as embedded: {filepath} ({chunk_count} chunks)")
    
    def get_collection_files(self, collection_name: str) -> Set[str]:
        """Get all files in a collection."""
        return set(self.cache_data['collections'].get(collection_name, {}).get('files', []))
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get information about a collection."""
        return self.cache_data['collections'].get(collection_name)
    
    def list_collections(self) -> Dict[str, Dict]:
        """List all collections in cache."""
        return self.cache_data['collections']
    
    def remove_collection(self, collection_name: str):
        """Remove collection from cache."""
        if collection_name in self.cache_data['collections']:
            files = self.cache_data['collections'][collection_name]['files']
            
            # Remove file entries
            for filepath in files:
                if filepath in self.cache_data['files']:
                    del self.cache_data['files'][filepath]
            
            # Remove collection
            del self.cache_data['collections'][collection_name]
            self._save_cache()
            logger.info(f"Removed collection from cache: {collection_name}")