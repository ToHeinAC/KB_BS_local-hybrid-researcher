"""ChromaDB client for vector search across collections."""

import logging
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.models.results import VectorResult

logger = logging.getLogger(__name__)


class CollectionNotFoundError(Exception):
    """Raised when a ChromaDB collection is not found."""

    pass


class ChromaDBClient:
    """Client for multi-collection ChromaDB operations."""

    # Known collection name patterns
    COLLECTION_PATTERNS = {
        "GLageKon": "GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000",
        "NORM": "NORM__Qwen--Qwen3-Embedding-0.6B--3000--600",
        "StrlSch": "StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600",
        "StrlSchExt": "StrlSchExt__Qwen--Qwen3-Embedding-0.6B--3000--600",
    }

    # PDF folder mappings
    PDF_FOLDERS = {
        "GLageKon": "GLageKon__db_inserted",
        "NORM": "NORM__db_inserted",
        "StrlSch": "StrlSch__db_inserted",
        "StrlSchExt": "StrlSchExt__db_inserted",
    }

    def __init__(self, base_path: str | Path | None = None):
        """Initialize ChromaDB client.

        Args:
            base_path: Base path to ChromaDB databases. Defaults to settings.
        """
        self.base_path = Path(base_path) if base_path else settings.chromadb_path_resolved
        self._clients: dict[str, chromadb.PersistentClient] = {}
        self._collections: dict[str, Collection] = {}
        self._langchain_stores: dict[str, Chroma] = {}
        self._embedding_cache: dict[str, HuggingFaceEmbeddings] = {}

        logger.info(f"ChromaDB base path: {self.base_path}")

    def _get_embeddings(self, model_name: str) -> HuggingFaceEmbeddings:
        """Get or create cached HuggingFace embeddings for a model.

        Args:
            model_name: HuggingFace model name (e.g., 'Qwen/Qwen3-Embedding-0.6B')

        Returns:
            Cached HuggingFaceEmbeddings instance
        """
        if model_name not in self._embedding_cache:
            logger.info(f"Loading embedding model: {model_name}")
            self._embedding_cache[model_name] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embedding_cache[model_name]

    def _resolve_embedding_model(self, db_name: str) -> str:
        """Resolve embedding model name from database directory name.

        Falls back to settings.default_embedding_model if parsing fails.

        Args:
            db_name: Database directory name

        Returns:
            HuggingFace model name string
        """
        model = self.extract_embedding_model(db_name)
        if model:
            return model
        logger.warning(
            f"Could not extract embedding model from '{db_name}', "
            f"using default: {settings.default_embedding_model}"
        )
        return settings.default_embedding_model

    def _get_client_for_collection(self, collection_key: str) -> chromadb.PersistentClient:
        """Get or create ChromaDB client for a specific collection.

        Args:
            collection_key: Short collection key (e.g., 'GLageKon')

        Returns:
            PersistentClient for the collection
        """
        if collection_key not in self._clients:
            db_name = self.COLLECTION_PATTERNS.get(collection_key)
            if not db_name:
                raise CollectionNotFoundError(f"Unknown collection: {collection_key}")

            # ChromaDB databases have a 'default' subdirectory
            db_path = self.base_path / db_name / "default"
            if not db_path.exists():
                # Try without 'default' for backwards compatibility
                db_path = self.base_path / db_name
                if not db_path.exists():
                    raise CollectionNotFoundError(f"Database not found: {db_path}")

            self._clients[collection_key] = chromadb.PersistentClient(path=str(db_path))
            logger.info(f"Connected to ChromaDB at {db_path}")

        return self._clients[collection_key]

    def _get_collection(self, collection_key: str) -> Collection:
        """Get or retrieve a collection by key.

        Args:
            collection_key: Short collection key (e.g., 'GLageKon')

        Returns:
            ChromaDB Collection
        """
        if collection_key not in self._collections:
            client = self._get_client_for_collection(collection_key)
            collections = client.list_collections()

            if not collections:
                raise CollectionNotFoundError(
                    f"No collections found in database for {collection_key}"
                )

            # Use first (and typically only) collection
            self._collections[collection_key] = collections[0]
            logger.info(
                f"Using collection '{self._collections[collection_key].name}' "
                f"for {collection_key}"
            )

        return self._collections[collection_key]

    def list_available_collections(self) -> list[str]:
        """List all available collection keys."""
        available = []
        for key, db_name in self.COLLECTION_PATTERNS.items():
            db_path = self.base_path / db_name
            if db_path.exists():
                available.append(key)
        return available

    def _get_langchain_store(self, collection_key: str) -> Chroma:
        """Get or create LangChain Chroma store for a collection.

        Args:
            collection_key: Short collection key (e.g., 'GLageKon')

        Returns:
            LangChain Chroma store
        """
        if collection_key not in self._langchain_stores:
            db_name = self.COLLECTION_PATTERNS.get(collection_key)
            if not db_name:
                raise CollectionNotFoundError(f"Unknown collection: {collection_key}")

            # ChromaDB databases have a 'default' subdirectory
            db_path = self.base_path / db_name / "default"
            if not db_path.exists():
                db_path = self.base_path / db_name

            # Get the collection name
            raw_client = chromadb.PersistentClient(path=str(db_path))
            collections = raw_client.list_collections()
            if not collections:
                raise CollectionNotFoundError(f"No collections in {collection_key}")

            collection_name = collections[0].name

            embedding_model = self._resolve_embedding_model(db_name)
            embeddings = self._get_embeddings(embedding_model)

            self._langchain_stores[collection_key] = Chroma(
                persist_directory=str(db_path),
                embedding_function=embeddings,
                collection_name=collection_name,
            )

            logger.info(
                f"Created LangChain Chroma store for {collection_key} "
                f"with embedding model: {embedding_model}"
            )

        return self._langchain_stores[collection_key]

    def search(
        self,
        query: str,
        collection_key: str,
        top_k: int | None = None,
    ) -> list[VectorResult]:
        """Search a specific collection using LangChain Chroma.

        Args:
            query: Search query text
            collection_key: Collection to search (e.g., 'GLageKon')
            top_k: Number of results to return

        Returns:
            List of VectorResult objects
        """
        top_k = top_k or settings.m_chunks_per_query
        store = self._get_langchain_store(collection_key)

        # Use similarity_search_with_score for relevance scores
        results = store.similarity_search_with_score(query, k=top_k)

        vector_results = []
        for doc, score in results:
            metadata = doc.metadata or {}

            # Score is distance, convert to similarity
            similarity = 1 - score if score < 1 else 1 / (1 + score)

            vector_results.append(
                VectorResult(
                    doc_id=metadata.get("id", str(hash(doc.page_content[:50]))),
                    doc_name=metadata.get("source", metadata.get("filename", "unknown")),
                    chunk_text=doc.page_content,
                    page_number=metadata.get("page"),
                    relevance_score=similarity,
                    collection=collection_key,
                    query_used=query,
                )
            )

        logger.debug(f"Found {len(vector_results)} results for query in {collection_key}")
        return vector_results

    def search_all_collections(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, list[VectorResult]]:
        """Search all available collections.

        Args:
            query: Search query text
            top_k: Number of results per collection

        Returns:
            Dict mapping collection_key to list of VectorResult
        """
        results = {}
        for collection_key in self.list_available_collections():
            try:
                results[collection_key] = self.search(query, collection_key, top_k)
            except Exception as e:
                logger.warning(f"Error searching {collection_key}: {e}")
                results[collection_key] = []

        return results

    def search_multi_collection(
        self,
        query: str,
        collection_keys: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[VectorResult]:
        """Search multiple collections and merge results.

        Args:
            query: Search query text
            collection_keys: Collections to search (defaults to all)
            top_k: Number of results per collection

        Returns:
            Merged and sorted list of VectorResult
        """
        collection_keys = collection_keys or self.list_available_collections()
        all_results = []

        for key in collection_keys:
            if key in self.COLLECTION_PATTERNS:
                results = self.search(query, key, top_k)
                all_results.extend(results)

        # Sort by relevance score descending
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_results

    def get_pdf_folder(self, collection_key: str) -> Path:
        """Get the PDF folder path for a collection.

        Args:
            collection_key: Collection key (e.g., 'GLageKon')

        Returns:
            Path to PDF folder
        """
        folder_name = self.PDF_FOLDERS.get(collection_key)
        if not folder_name:
            raise CollectionNotFoundError(f"Unknown collection: {collection_key}")

        # PDF folders are in kb/ parent directory
        kb_path = self.base_path.parent
        return kb_path / folder_name

    def get_collection_stats(self, collection_key: str) -> dict:
        """Get statistics for a collection.

        Args:
            collection_key: Collection to get stats for

        Returns:
            Dict with count and metadata
        """
        collection = self._get_collection(collection_key)
        return {
            "name": collection.name,
            "count": collection.count(),
        }

    def list_database_directories(self) -> list[str]:
        """List all database directories under kb/database/.

        Returns:
            List of database directory names
        """
        if not self.base_path.exists():
            logger.warning(f"Base path does not exist: {self.base_path}")
            return []

        directories = []
        for d in self.base_path.iterdir():
            if d.is_dir():
                # Check if it looks like a ChromaDB database (has chroma files)
                default_dir = d / "default"
                if default_dir.exists() or any(d.glob("*.sqlite3")) or any(d.glob("chroma*")):
                    directories.append(d.name)

        directories.sort()
        logger.info(f"Found {len(directories)} database directories")
        return directories

    def extract_embedding_model(self, db_name: str) -> str | None:
        """Extract embedding model from database name pattern.

        Format: CollectionName__Model--SubModel--ChunkSize--Overlap
        Example: GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000
        Returns: Qwen/Qwen3-Embedding-0.6B

        Args:
            db_name: Database directory name

        Returns:
            Embedding model name in HuggingFace format or None
        """
        parts = db_name.split("__")
        if len(parts) >= 2:
            model_part = parts[1]
            # Split by -- and take first two parts for model name
            model_parts = model_part.split("--")
            if len(model_parts) >= 2:
                return f"{model_parts[0]}/{model_parts[1]}"
            elif len(model_parts) == 1:
                return model_parts[0]
        return None

    def search_by_database_name(
        self,
        query: str,
        db_name: str,
        top_k: int | None = None,
    ) -> list[VectorResult]:
        """Search a specific database by its directory name.

        Args:
            query: Search query text
            db_name: Database directory name (e.g., 'GLageKon__Qwen--...')
            top_k: Number of results to return

        Returns:
            List of VectorResult objects
        """
        top_k = top_k or settings.m_chunks_per_query

        # Create a cache key for this specific database
        cache_key = f"db:{db_name}"

        if cache_key not in self._langchain_stores:
            db_path = self.base_path / db_name
            if not db_path.exists():
                logger.error(f"Database not found: {db_path}")
                return []

            # Try default subdirectory first
            actual_path = db_path / "default" if (db_path / "default").exists() else db_path

            try:
                # Get collection name from the database
                raw_client = chromadb.PersistentClient(path=str(actual_path))
                collections = raw_client.list_collections()
                if not collections:
                    logger.warning(f"No collections in database: {db_name}")
                    return []

                collection_name = collections[0].name

                embedding_model = self._resolve_embedding_model(db_name)
                embeddings = self._get_embeddings(embedding_model)

                self._langchain_stores[cache_key] = Chroma(
                    persist_directory=str(actual_path),
                    embedding_function=embeddings,
                    collection_name=collection_name,
                )
                logger.info(
                    f"Created LangChain Chroma store for database: {db_name} "
                    f"with embedding model: {embedding_model}"
                )

            except Exception as e:
                logger.error(f"Failed to connect to database {db_name}: {e}")
                return []

        store = self._langchain_stores[cache_key]

        try:
            results = store.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            logger.error(f"Search failed for database {db_name}: {e}")
            return []

        vector_results = []
        for doc, score in results:
            metadata = doc.metadata or {}

            # Score is distance, convert to similarity
            similarity = 1 - score if score < 1 else 1 / (1 + score)

            # Extract collection key from db_name
            collection_key = db_name.split("__")[0] if "__" in db_name else db_name

            vector_results.append(
                VectorResult(
                    doc_id=metadata.get("id", str(hash(doc.page_content[:50]))),
                    doc_name=metadata.get("source", metadata.get("filename", "unknown")),
                    chunk_text=doc.page_content,
                    page_number=metadata.get("page"),
                    relevance_score=similarity,
                    collection=collection_key,
                    query_used=query,
                )
            )

        logger.debug(f"Found {len(vector_results)} results for query in {db_name}")
        return vector_results
