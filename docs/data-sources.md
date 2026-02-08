# Data Sources

## PDF Document Corpus

### Overview

| Property | Value |
|----------|-------|
| Source Folders | `kb/{category}__db_inserted/` (one per collection) |
| Total Count | ~761 PDF documents across all collections |
| Language | German |
| Domain | Technical/regulatory documentation |
| Topics | Construction, engineering, nuclear safety, radiation protection |

### Folder Structure

```
kb/
├── GLageKon__db_inserted/     # PDFs for GLageKon collection (~450 docs)
├── NORM__db_inserted/         # PDFs for NORM collection (~50 docs)
├── StrlSch__db_inserted/      # PDFs for StrlSch collection (~100 docs)
├── StrlSchExt__db_inserted/   # PDFs for StrlSchExt collection (~150 docs)
└── insert_data/               # Initial import folder (ignore at runtime)
```

**Note:** `kb/insert_data/` is only used for initial vector DB population and should be ignored at runtime. The `*__db_inserted/` folders contain the actual source PDFs corresponding to each ChromaDB collection.

### Document Naming Convention

Files follow sequential numbering with descriptive names:
```
003_EG_025_K1,_Verw.-_u._Sozialgebäude.pdf
010_StrlSchG.pdf
015_StrlSchV.pdf
```

### Use Case Example

German radiation protection legislation with complex cross-references:
- **StrlSchG** (Strahlenschutzgesetz) ↔ **StrlSchV** (Strahlenschutzverordnung)
- Documents reference each other: "gemäß § 18 StrlSchV", "siehe Abschnitt 3"

---

## ChromaDB Collections

### Pre-populated Collections

| Collection | PDF Folder | Chunk Size | Overlap | ~Docs | Purpose |
|------------|------------|------------|---------|-------|---------|
| `GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000` | `kb/GLageKon__db_inserted/` | 10000 | 2000 | 450 | General layout/construction |
| `NORM__Qwen--Qwen3-Embedding-0.6B--3000--600` | `kb/NORM__db_inserted/` | 3000 | 600 | 50 | Standards/norms |
| `StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600` | `kb/StrlSch__db_inserted/` | 3000 | 600 | 100 | Radiation protection |
| `StrlSchExt__Qwen--Qwen3-Embedding-0.6B--3000--600` | `kb/StrlSchExt__db_inserted/` | 3000 | 600 | 150 | Extended radiation protection |

### Naming Convention

```
{category}__{embedding_model}--{chunk_size}--{overlap}
```

Example breakdown:
- `GLageKon`: Category (General Layout/Construction)
- `Qwen--Qwen3-Embedding-0.6B`: Embedding model used
- `10000--2000`: 10000 char chunks with 2000 char overlap

### Database Location

```
kb/database/
├── GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000/
├── NORM__Qwen--Qwen3-Embedding-0.6B--3000--600/
├── StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600/
└── StrlSchExt__Qwen--Qwen3-Embedding-0.6B--3000--600/
```

### Collection ↔ PDF Folder Mapping

Each ChromaDB collection has a corresponding PDF folder:

| Collection Prefix | PDF Source Folder |
|-------------------|-------------------|
| `GLageKon__*` | `kb/GLageKon__db_inserted/` |
| `NORM__*` | `kb/NORM__db_inserted/` |
| `StrlSch__*` | `kb/StrlSch__db_inserted/` |
| `StrlSchExt__*` | `kb/StrlSchExt__db_inserted/` |

---

## Embedding Model

### Qwen3-Embedding-0.6B

| Property | Value |
|----------|-------|
| Model | Qwen3-Embedding-0.6B |
| Provider | HuggingFace |
| RAM Required | ~1GB |
| Dimensions | 1024 |

### Usage via HuggingFace (as implemented)

The application uses `langchain_huggingface.HuggingFaceEmbeddings` with the model `Qwen/Qwen3-Embedding-0.6B` and runs embeddings on CUDA:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vector = embeddings.embed_query("What are the dose limits?")
```

---

## ChromaDB Client Usage

### Connection

```python
import chromadb

# Connect to persistent storage
client = chromadb.PersistentClient(
    path="kb/database/GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000"
)

# Get collection
collection = client.get_collection("GLageKon")

# Query
results = collection.query(
    query_texts=["Strahlenschutz Grenzwerte"],
    n_results=5,
)
```

### List Collections

```bash
python -c "
import chromadb
c = chromadb.PersistentClient('kb/database/GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000')
print([col.name for col in c.list_collections()])
"
```

### Multi-Collection Search

```python
from src.services.chromadb_client import ChromaDBClient

client = ChromaDBClient(base_path="kb/database")

# Search across multiple collections
results = client.search_all(
    query="Dosisgrenzwerte",
    collections=["GLageKon", "StrlSch", "NORM"],
    top_k=5,
)
```

---

## Document Registry

For reference resolution, the system uses a **document registry** (`kb/document_registry.json`) that maps PDF filenames to human-readable synonyms across all 4 collections.

### document_registry.json

```json
{
  "collections": {
    "StrlSch": {
      "documents": [
        {"filename": "StrlSchG.pdf", "synonyms": ["Strahlenschutzgesetz", "StrlSchG", "StrSchG"]},
        {"filename": "StrlSchV.pdf", "synonyms": ["Strahlenschutzverordnung", "StrlSchV"]},
        {"filename": "AtG.pdf", "synonyms": ["Atomgesetz", "AtG", "Atomic Energy Act"]},
        {"filename": "KTA 1401_2017-11.pdf", "synonyms": ["KTA 1401"]}
      ]
    },
    "NORM": {
      "documents": [
        {"filename": "ICRP_103.pdf", "synonyms": ["ICRP 103", "ICRP Publication 103"]},
        {"filename": "Trinkwasserverordnung.pdf", "synonyms": ["Trinkwasserverordnung", "TrinkwV"]}
      ]
    }
  }
}
```

### Usage in Reference Resolution

```python
from src.agents.tools import load_document_registry, resolve_document_name

# Singleton loader
registry = load_document_registry()

# 3-stage resolution: exact synonym > fuzzy (0.7) > substring
filename, collection_key = resolve_document_name("Strahlenschutzgesetz")
# -> ("StrlSchG.pdf", "StrlSch")

filename, collection_key = resolve_document_name("KTA 1401", collection_hint="StrlSch")
# -> ("KTA 1401_2017-11.pdf", "StrlSch")

filename, collection_key = resolve_document_name("NonexistentDoc")
# -> (None, None)
```

The resolved `(filename, collection_key)` is used by `_vector_search_scoped()` to search within the specific collection and post-filter by document name, enabling precise reference following.

---

## Source Path Resolution

Map ChromaDB collection to source PDF folder:

```python
def resolve_source_directory(collection_name: str) -> str:
    """Map collection name to source PDF folder.

    Each collection has a corresponding *__db_inserted folder.

    Example:
        'GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000'
        → 'kb/GLageKon__db_inserted/'
    """
    # Extract category prefix (before first '__')
    category = collection_name.split("__")[0]
    return f"kb/{category}__db_inserted/"

def resolve_source_path(doc_name: str, collection: str) -> str:
    """Get full path to source document."""
    base_dir = resolve_source_directory(collection)
    return f"{base_dir}{doc_name}"

# Example usage:
# resolve_source_directory("GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000")
# → "kb/GLageKon__db_inserted/"
#
# resolve_source_path("003_EG_025_K1.pdf", "GLageKon__Qwen--...")
# → "kb/GLageKon__db_inserted/003_EG_025_K1.pdf"
```
