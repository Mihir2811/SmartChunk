# Main application entry point

#!/usr/bin/env python3
"""
Unified Incremental Embedding Pipeline
Handles multiple file types with smart chunking and incremental updates.
"""

import sys
from typing import List
from pathlib import Path

import click
from dotenv import load_dotenv

from utils import (
    get_logger,
    get_file_metadata,
    validate_environment,
    format_file_size
)
from file_processors import FileProcessorFactory
from smart_chunker import SmartDocumentSplitter
from cache_manager import EmbeddingCacheManager
from qdrant_client_wrapper import QdrantManager

logger = get_logger(__name__)

class UnifiedDocumentEmbedder:
    """
    Main orchestrator for document embedding pipeline.
    """
    
    def __init__(
        self,
        openai_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        chunk_size: int = 4000,
        chunk_overlap: int = 400
    ):
        """
        Initialize the unified embedder.
        
        Args:
            openai_key: OpenAI API key
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
        """
        self.processor_factory = FileProcessorFactory()
        self.chunker = SmartDocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.cache_manager = EmbeddingCacheManager()
        self.qdrant_manager = QdrantManager(
            url=qdrant_url,
            api_key=qdrant_api_key,
            openai_key=openai_key
        )
        
        logger.info("Initialized UnifiedDocumentEmbedder")
    
    def embed_file(
        self,
        filepath: str,
        collection_name: str,
        force: bool = False,
        incremental: bool = True
    ) -> bool:
        """
        Embed a single file into a collection.
        
        Args:
            filepath: Path to the file
            collection_name: Target collection name
            force: Force re-embedding even if file hasn't changed
            incremental: Add to existing collection vs recreate
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Processing file: {filepath}")
        
        # Check if file needs processing
        if not force and not self.cache_manager.has_file_changed(filepath):
            logger.info(f"File unchanged, skipping: {filepath}")
            return True
        
        try:
            # Get file metadata
            metadata = get_file_metadata(filepath)
            logger.info(f"File metadata: {metadata['filename']} ({format_file_size(metadata['file_size'])})")
            
            # Get appropriate processor
            processor = self.processor_factory.get_processor(filepath)
            
            # Extract content and sections
            content = processor.extract_text(filepath)
            sections = processor.extract_structured_sections(filepath)
            
            # Add embedding metadata
            metadata['embedding_date'] = self.cache_manager._load_cache()  # Current timestamp
            
            # Chunk the content
            documents = self.chunker.split_content(
                content=content,
                file_type=metadata['file_type'],
                metadata=metadata,
                sections=sections
            )
            
            logger.info(f"Created {len(documents)} chunks from {filepath}")
            
            # Embed and store
            if incremental:
                self.qdrant_manager.add_documents_to_collection(
                    documents=documents,
                    collection_name=collection_name
                )
            else:
                self.qdrant_manager.create_collection_from_documents(
                    documents=documents,
                    collection_name=collection_name,
                    recreate=False
                )
            
            # Update cache
            self.cache_manager.mark_file_embedded(
                filepath=filepath,
                collection_name=collection_name,
                chunk_count=len(documents),
                metadata=metadata
            )
            
            logger.info(f"Successfully embedded {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding file {filepath}: {e}", exc_info=True)
            return False
    
    def embed_multiple_files(
        self,
        filepaths: List[str],
        collection_name: str,
        force: bool = False,
        incremental: bool = True
    ) -> dict:
        """
        Embed multiple files into a collection.
        
        Args:
            filepaths: List of file paths
            collection_name: Target collection name
            force: Force re-embedding
            incremental: Add to existing collection
            
        Returns:
            Dictionary with success/failure counts
        """
        logger.info(f"Embedding {len(filepaths)} files into collection '{collection_name}'")
        
        results = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'failed_files': []
        }
        
        for filepath in filepaths:
            try:
                success = self.embed_file(
                    filepath=filepath,
                    collection_name=collection_name,
                    force=force,
                    incremental=incremental
                )
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['failed_files'].append(filepath)
                    
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                results['failed'] += 1
                results['failed_files'].append(filepath)
        
        logger.info(
            f"Batch embedding complete: "
            f"{results['successful']} successful, "
            f"{results['failed']} failed, "
            f"{results['skipped']} skipped"
        )
        
        return results

# CLI Interface
@click.group()
def cli():
    """Unified Document Embedding Pipeline - Embed any document type into Qdrant."""
    pass

@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.argument('collection')
@click.option('--force', is_flag=True, help='Force re-embedding even if unchanged')
@click.option('--recreate', is_flag=True, help='Recreate collection instead of adding incrementally')
def embed(filepath, collection, force, recreate):
    """Embed a single file into a collection."""
    try:
        # Validate environment
        openai_key, qdrant_url, qdrant_api_key = validate_environment()
        
        # Initialize embedder
        embedder = UnifiedDocumentEmbedder(
            openai_key=openai_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        # Embed file
        success = embedder.embed_file(
            filepath=filepath,
            collection_name=collection,
            force=force,
            incremental=not recreate
        )
        
        if success:
            click.echo(click.style(f"✓ Successfully embedded {filepath}", fg='green'))
        else:
            click.echo(click.style(f"✗ Failed to embed {filepath}", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)

@cli.command()
@click.argument('collection')
@click.argument('filepaths', nargs=-1, type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='Force re-embedding')
@click.option('--recreate', is_flag=True, help='Recreate collection')
def embed_multi(collection, filepaths, force, recreate):
    """Embed multiple files into a collection."""
    if not filepaths:
        click.echo(click.style("✗ No files provided", fg='red'))
        sys.exit(1)
    
    try:
        openai_key, qdrant_url, qdrant_api_key = validate_environment()
        
        embedder = UnifiedDocumentEmbedder(
            openai_key=openai_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        results = embedder.embed_multiple_files(
            filepaths=list(filepaths),
            collection_name=collection,
            force=force,
            incremental=not recreate
        )
        
        click.echo(f"\nResults:")
        click.echo(click.style(f"  ✓ Successful: {results['successful']}", fg='green'))
        click.echo(click.style(f"  ✗ Failed: {results['failed']}", fg='red'))
        
        if results['failed_files']:
            click.echo("\nFailed files:")
            for f in results['failed_files']:
                click.echo(f"  - {f}")
                
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)

@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.argument('collection')
def add(filepath, collection):
    """Add a file incrementally to an existing collection."""
    try:
        openai_key, qdrant_url, qdrant_api_key = validate_environment()
        
        embedder = UnifiedDocumentEmbedder(
            openai_key=openai_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        success = embedder.embed_file(
            filepath=filepath,
            collection_name=collection,
            force=False,
            incremental=True
        )
        
        if success:
            click.echo(click.style(f"✓ Added {filepath} to {collection}", fg='green'))
        else:
            click.echo(click.style(f"✗ Failed to add {filepath}", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)

@cli.command()
def list():
    """List all collections."""
    try:
        openai_key, qdrant_url, qdrant_api_key = validate_environment()
        
        embedder = UnifiedDocumentEmbedder(
            openai_key=openai_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        collections = embedder.qdrant_manager.list_collections()
        
        if not collections:
            click.echo("No collections found")
            return
        
        click.echo("\nCollections:")
        for coll in collections:
            info = embedder.qdrant_manager.get_collection_info(coll)
            if info:
                click.echo(f"  • {coll} ({info['points_count']} points)")
            else:
                click.echo(f"  • {coll}")
                
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)

@cli.command()
@click.argument('collection')
def info(collection):
    """Show detailed information about a collection."""
    try:
        openai_key, qdrant_url, qdrant_api_key = validate_environment()
        
        embedder = UnifiedDocumentEmbedder(
            openai_key=openai_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        # Get Qdrant info
        qdrant_info = embedder.qdrant_manager.get_collection_info(collection)
        
        if not qdrant_info:
            click.echo(click.style(f"✗ Collection '{collection}' not found", fg='red'))
            sys.exit(1)
        
        # Get cache info
        cache_info = embedder.cache_manager.get_collection_info(collection)
        
        click.echo(f"\nCollection: {collection}")
        click.echo(f"  Points: {qdrant_info['points_count']}")
        click.echo(f"  Vectors: {qdrant_info['vectors_count']}")
        click.echo(f"  Status: {qdrant_info['status']}")
        
        if cache_info:
            click.echo(f"\nCache Info:")
            click.echo(f"  Files: {len(cache_info['files'])}")
            click.echo(f"  Total Chunks: {cache_info['total_chunks']}")
            click.echo(f"  Created: {cache_info.get('created_date', 'N/A')}")
            click.echo(f"  Last Updated: {cache_info.get('last_updated', 'N/A')}")
            
            click.echo(f"\n  Files in collection:")
            for f in cache_info['files']:
                click.echo(f"    - {f}")
                
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)

@cli.command()
@click.argument('collection')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def delete(collection, confirm):
    """Delete a collection."""
    if not confirm:
        if not click.confirm(f"Are you sure you want to delete collection '{collection}'?"):
            click.echo("Cancelled")
            return
    
    try:
        openai_key, qdrant_url, qdrant_api_key = validate_environment()
        
        embedder = UnifiedDocumentEmbedder(
            openai_key=openai_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        # Delete from Qdrant
        success = embedder.qdrant_manager.delete_collection(collection)
        
        if success:
            # Delete from cache
            embedder.cache_manager.remove_collection(collection)
            click.echo(click.style(f"✓ Deleted collection '{collection}'", fg='green'))
        else:
            click.echo(click.style(f"✗ Failed to delete collection '{collection}'", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)

if __name__ == '__main__':
    cli()