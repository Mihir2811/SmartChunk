# Unified Incremental Embedding Pipeline

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

A production-ready Python application for intelligently embedding any document type into vector databases using OpenAI embeddings and Qdrant Cloud. Built with smart chunking strategies, incremental updates, and comprehensive metadata tracking.

---

## Overview

This pipeline solves the challenge of building high-quality Retrieval-Augmented Generation (RAG) systems by providing intelligent document processing that preserves semantic meaning while optimizing for retrieval accuracy. Whether you're processing legal contracts, technical documentation, code repositories, or Postman collections, this tool handles it all with appropriate chunking strategies.

### Key Features

* **Incremental Updates**: Hash-based change detection prevents unnecessary re-embedding, saving costs and time
* **Universal File Support**: Handles PDF, Word, Markdown, JSON, YAML, code files, and plain text
* **Smart Chunking**: Automatically selects optimal chunking strategy based on file type and content structure
* **Comprehensive Metadata**: Full provenance tracking for every chunk (source file, section, timestamps, hashes)
* **Production-Ready**: Robust error handling, logging, caching, and CLI interface
* **Extensible Architecture**: Modular design allows easy addition of new file processors
* **Cost-Optimized**: Caching and incremental processing minimize API costs
* **Qdrant Cloud Native**: Built specifically for Qdrant vector database with gRPC support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface (Click)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              UnifiedDocumentEmbedder                         │
│  (Main Orchestrator - coordinates all components)            │
└─┬──────────────┬──────────────┬────────────────┬───────────┘
  │              │              │                │
  ▼              ▼              ▼                ▼
┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐
│  File   │  │  Smart   │  │  Cache  │  │   Qdrant     │
│Processor│  │ Chunker  │  │ Manager │  │   Manager    │
│ Factory │  │          │  │         │  │              │
└─────────┘  └──────────┘  └─────────┘  └──────────────┘
     │              │            │              │
     ▼              ▼            ▼              ▼
 PDF, Word,   Recursive,    SHA-256      OpenAI API
 Markdown,    Semantic,     Hashing      Qdrant Cloud
 JSON, YAML   Language-     .meta.json   Collections
 Code, Text   Aware Split   Cache
```

### Component Responsibilities

| Component           | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| **File Processors** | Extract text and structure from different file formats      |
| **Smart Chunker**   | Apply appropriate chunking strategy based on content type   |
| **Cache Manager**   | Track file hashes and prevent redundant embeddings          |
| **Qdrant Manager**  | Handle vector storage, retrieval, and collection management |
| **CLI Interface**   | Provide user-friendly command-line operations               |

---

## Quick Start

### Prerequisites

* Python 3.10 or higher
* OpenAI API key
* Qdrant Cloud account and API key

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/unified-embedding-pipeline.git
cd unified-embedding-pipeline
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key-here
```

4. **Verify installation**

```bash
python unified_embedder.py --help
```

---

## Usage

### Basic Commands

#### Embed a Single File

```bash
python unified_embedder.py embed document.pdf my_collection
```

#### Embed Multiple Files

```bash
python unified_embedder.py embed-multi my_collection file1.pdf file2.docx code.py api.json
```

#### Add File Incrementally (Skip if Unchanged)

```bash
python unified_embedder.py add updated_document.pdf my_collection
```

#### Force Re-embedding

```bash
python unified_embedder.py embed document.pdf my_collection --force
```

#### Recreate Collection

```bash
python unified_embedder.py embed document.pdf my_collection --recreate
```

#### List All Collections

```bash
python unified_embedder.py list
```

#### Show Collection Details

```bash
python unified_embedder.py info my_collection
```

#### Delete a Collection

```bash
python unified_embedder.py delete my_collection --confirm
```

---

## Supported File Types

| File Type          | Extensions                                                  | Processing Strategy                                     |
| ------------------ | ----------------------------------------------------------- | ------------------------------------------------------- |
| **PDF**            | `.pdf`                                                      | Page-aware extraction with structure preservation       |
| **Word Documents** | `.docx`, `.doc`                                             | Paragraph and heading-based splitting                   |
| **Markdown**       | `.md`, `.markdown`                                          | Header hierarchy-aware chunking                         |
| **JSON**           | `.json`                                                     | Key-based sections, special Postman collection handling |
| **YAML**           | `.yaml`, `.yml`                                             | Top-level key splitting                                 |
| **Code Files**     | `.py`, `.js`, `.java`, `.ts`, `.cpp`, `.go`, `.rs`, `.html` | Language-aware AST-based chunking                       |
| **Plain Text**     | `.txt`                                                      | Paragraph-based splitting                               |

---

## Smart Chunking Strategies

The pipeline automatically selects the optimal chunking strategy based on file type.

### Code Files (Python, JavaScript, Java, etc.)

```python
# Preserves function and class boundaries
def process_data(df):
    return df.dropna()

class DataProcessor:
    def __init__(self):
        pass
```

**Strategy**: Language-aware recursive splitting using LangChain's `Language` enum.

### Markdown Documents

```markdown
# Main Header
Content under main header...

## Subsection
Subsection content stays together...

### Deep Section
Deep content preserved...
```

**Strategy**: Header hierarchy preservation with section metadata.

### JSON/YAML Files

```yaml
database:
  host: localhost
  port: 5432

api:
  endpoint: /v1/data
  timeout: 30
```

**Strategy**: Top-level key-based splitting, Postman collection request extraction.

### PDFs and Word Documents

**Strategy**: Page-aware or section-aware extraction maintaining document structure.

---

## Configuration

### Chunk Size and Overlap

Default settings optimized for OpenAI `text-embedding-3-small` (8191 token limit):

```python
chunk_size = 4000
chunk_overlap = 400
```

Modify in `unified_embedder.py`:

```python
embedder = UnifiedDocumentEmbedder(
    openai_key=openai_key,
    qdrant_url=qdrant_url,
    qdrant_api_key=qdrant_api_key,
    chunk_size=4000,
    chunk_overlap=400
)
```

### Cache Location

Default cache file: `.unified_embedding.meta.json`

To change:

```python
cache_manager = EmbeddingCacheManager(cache_file="custom_cache.json")
```

### Logging Level

Modify in `utils.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## Metadata Schema

Every embedded chunk includes comprehensive metadata:

```json
{
  "filename": "technical_doc.pdf",
  "filepath": "/absolute/path/to/technical_doc.pdf",
  "file_type": ".pdf",
  "file_size": 2458624,
  "created_date": "2025-01-15T10:30:00",
  "modified_date": "2025-01-20T14:22:00",
  "file_hash": "a3f5d8c9e2b1...",
  "embedding_date": "2025-01-21T09:15:00",
  "section_name": "Chapter 3 > Security Protocols",
  "chunk_number": 5,
  "total_chunks": 24,
  "split_method": "recursive",
  "page_number": 12
}
```

This enables:

* Provenance tracking
* Change detection
* Context reconstruction
* Filtering and section-level retrieval

---

## Use Cases

### 1. Technical Documentation RAG System

```bash
python unified_embedder.py embed-multi technical_docs docs/*.md docs/*.pdf docs/api/*.json
```

### 2. Legal Document Analysis

```bash
python unified_embedder.py embed contract_2024.pdf legal_kb
```

### 3. Code Repository Search

```bash
find ./src -name "*.py" -exec python unified_embedder.py add {} codebase_vectors \;
```

### 4. API Collection Management

```bash
python unified_embedder.py embed api_collection.json api_docs
```

### 5. Knowledge Base Updates

```bash
python unified_embedder.py embed-multi knowledge_base updated/*.pdf updated/*.md
```

---

## Advanced Features

### Incremental Update Workflow

The system maintains a local cache (`.unified_embedding.meta.json`) that tracks file hashes.

1. First Run: All files are processed and embedded
2. Subsequent Runs:

   * Compute SHA-256 hash of each file
   * Compare with cached hash
   * Skip unchanged files
   * Process only new/modified files
3. Cache Update: Store new hashes and metadata

---

## Development

### Project Structure

```
unified-embedder/
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── LICENSE
├── unified_embedder.py
├── file_processors.py
├── smart_chunker.py
├── cache_manager.py
├── qdrant_client_wrapper.py
├── utils.py
└── .unified_embedding.meta.json
```

### Adding Custom File Processors

```python
class CustomProcessor(FileProcessor):
    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith('.custom')

    def extract_text(self, file_path: str) -> str:
        with open(file_path, 'r') as f:
            return f.read()

    def extract_structured_sections(self, file_path: str):
        return [{"content": self.extract_text(file_path), "section_name": "Custom Section"}]
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built for the RAG community**

Last Updated: January 2025

---

