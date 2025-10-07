# Implements recursive, semantic, and contextual chunking

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.docstore.document import Document

from utils import get_logger

logger = get_logger(__name__)

class SmartDocumentSplitter:
    """
    Smart document splitter that adapts chunking strategy based on content type.
    """
    
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 400,
        length_function: callable = len
    ):
        """
        Initialize the smart splitter.
        
        Args:
            chunk_size: Target size for chunks
            chunk_overlap: Overlap between consecutive chunks
            length_function: Function to measure chunk length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        
        logger.info(f"Initialized SmartDocumentSplitter (size={chunk_size}, overlap={chunk_overlap})")
    
    def split_content(
        self,
        content: str,
        file_type: str,
        metadata: Dict[str, Any],
        sections: List[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Split content into chunks with appropriate strategy.
        
        Args:
            content: Text content to split
            file_type: File extension (e.g., '.md', '.py')
            metadata: File metadata to attach to chunks
            sections: Optional pre-extracted sections with metadata
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Splitting content (type={file_type}, length={len(content)} chars)")
        
        # Determine splitting strategy based on file type
        if file_type in {'.py', '.js', '.java', '.ts', '.tsx', '.jsx', '.cpp', '.c', '.go', '.rs'}:
            chunks = self._split_code(content, file_type, metadata)
        elif file_type == '.md':
            chunks = self._split_markdown(content, metadata, sections)
        elif file_type in {'.json', '.yaml', '.yml'}:
            chunks = self._split_structured(content, metadata, sections)
        else:
            chunks = self._split_generic(content, metadata)
        
        # Add chunk numbering
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks, 1):
            chunk.metadata['chunk_number'] = i
            chunk.metadata['total_chunks'] = total_chunks
        
        logger.info(f"Created {total_chunks} chunks")
        return chunks
    
    def _split_code(self, content: str, file_type: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split code using language-aware splitter."""
        
        # Map file extensions to Language enum
        language_map = {
            '.py': Language.PYTHON,
            '.js': Language.JS,
            '.ts': Language.TS,
            '.java': Language.JAVA,
            '.cpp': Language.CPP,
            '.c': Language.CPP,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.html': Language.HTML,
        }
        
        language = language_map.get(file_type, Language.PYTHON)
        
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        except:
            # Fallback to generic if language not supported
            splitter = self._get_generic_splitter()
        
        texts = splitter.split_text(content)
        
        return [
            Document(
                page_content=text,
                metadata={**metadata, 'split_method': 'code', 'language': language.value}
            )
            for text in texts
        ]
    
    def _split_markdown(
        self,
        content: str,
        metadata: Dict[str, Any],
        sections: List[Dict[str, Any]] = None
    ) -> List[Document]:
        """Split Markdown preserving header hierarchy."""
        
        if sections:
            # Use pre-extracted sections
            documents = []
            for section in sections:
                section_content = section['content']
                
                # If section is still too large, split it further
                if len(section_content) > self.chunk_size:
                    splitter = self._get_generic_splitter()
                    sub_chunks = splitter.split_text(section_content)
                    
                    for sub_chunk in sub_chunks:
                        documents.append(Document(
                            page_content=sub_chunk,
                            metadata={
                                **metadata,
                                'section_name': section.get('section_name'),
                                'header_level': section.get('header_level'),
                                'split_method': 'markdown_section'
                            }
                        ))
                else:
                    documents.append(Document(
                        page_content=section_content,
                        metadata={
                            **metadata,
                            'section_name': section.get('section_name'),
                            'header_level': section.get('header_level'),
                            'split_method': 'markdown_section'
                        }
                    ))
            
            return documents
        else:
            # Fallback to generic splitting
            return self._split_generic(content, metadata)
    
    def _split_structured(
        self,
        content: str,
        metadata: Dict[str, Any],
        sections: List[Dict[str, Any]] = None
    ) -> List[Document]:
        """Split structured data (JSON, YAML) by logical sections."""
        
        if sections:
            documents = []
            for section in sections:
                section_content = section['content']
                
                # If section is too large, split it
                if len(section_content) > self.chunk_size:
                    splitter = self._get_generic_splitter()
                    sub_chunks = splitter.split_text(section_content)
                    
                    for sub_chunk in sub_chunks:
                        documents.append(Document(
                            page_content=sub_chunk,
                            metadata={
                                **metadata,
                                'section_name': section.get('section_name'),
                                'split_method': 'structured_section'
                            }
                        ))
                else:
                    documents.append(Document(
                        page_content=section_content,
                        metadata={
                            **metadata,
                            'section_name': section.get('section_name'),
                            'split_method': 'structured_section'
                        }
                    ))
            
            return documents
        else:
            return self._split_generic(content, metadata)
    
    def _split_generic(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Generic recursive character splitting."""
        splitter = self._get_generic_splitter()
        texts = splitter.split_text(content)
        
        return [
            Document(
                page_content=text,
                metadata={**metadata, 'split_method': 'recursive'}
            )
            for text in texts
        ]
    
    def _get_generic_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get configured generic text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )