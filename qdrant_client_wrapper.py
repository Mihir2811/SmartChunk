# Wrapper around Qdrant client for CRUD operations

from typing import List, Optional, Dict, Any
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from utils import get_logger

logger = get_logger(__name__)

class QdrantManager:
    """
    Manages Qdrant operations for embedding storage and retrieval.
    """
    
    def __init__(self, url: str, api_key: str, openai_key: str):
        """
        Initialize Qdrant manager.
        
        Args:
            url: Qdrant cluster URL
            api_key: Qdrant API key
            openai_key: OpenAI API key
        """
        self.url = url
        self.api_key = api_key
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
        
        # Initialize native client for management operations
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=True
        )
        
        logger.info("Initialized Qdrant manager")
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections()
            return collection_name in [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def create_collection_from_documents(
        self,
        documents: List[Document],
        collection_name: str,
        recreate: bool = False
    ) -> Qdrant:
        """
        Create or update collection with documents.
        
        Args:
            documents: List of documents to embed
            collection_name: Name of collection
            recreate: If True, delete existing collection first
            
        Returns:
            Qdrant vectorstore instance
        """
        if recreate and self.collection_exists(collection_name):
            logger.info(f"Deleting existing collection: {collection_name}")
            self.client.delete_collection(collection_name)
        
        logger.info(f"Creating/updating collection '{collection_name}' with {len(documents)} documents")
        
        try:
            vectorstore = Qdrant.from_documents(
                documents,
                self.embedding,
                url=self.url,
                api_key=self.api_key,
                prefer_grpc=True,
                collection_name=collection_name,
            )
            
            logger.info(f"Successfully stored {len(documents)} chunks in '{collection_name}'")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_documents_to_collection(
        self,
        documents: List[Document],
        collection_name: str
    ):
        """
        Add documents to existing collection incrementally.
        
        Args:
            documents: Documents to add
            collection_name: Target collection name
        """
        logger.info(f"Adding {len(documents)} documents to collection '{collection_name}'")
        
        try:
            if not self.collection_exists(collection_name):
                logger.info(f"Collection doesn't exist, creating: {collection_name}")
                self.create_collection_from_documents(documents, collection_name)
            else:
                # Get existing vectorstore
                vectorstore = Qdrant(
                    client=self.client,
                    collection_name=collection_name,
                    embeddings=self.embedding
                )
                
                # Add documents
                vectorstore.add_documents(documents)
                
                logger.info(f"Successfully added documents to '{collection_name}'")
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection."""
        try:
            if not self.collection_exists(collection_name):
                return None
            
            info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "distance": info.config.params.vectors.distance.name if hasattr(info.config.params.vectors, 'distance') else 'unknown'
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection doesn't exist: {collection_name}")
                return False
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            collection_name: Collection to search
            query: Search query
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            vectorstore = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embedding
            )
            
            results = vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} results for query in '{collection_name}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []