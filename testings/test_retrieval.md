# GOAL
Improvements of retrieval quality by testing different constellations via a GUI application

# USER STORIES
- As a tester, in the GUI I want to select the database, the number of chunks (top k) given back, the similarity metric and the llm for chunk synthesis.
- As a tester, in the GUI I want to see the selected database and the embedding model.
- As a tester, I want to be able to explore the selected database, i. e. I want to be able to see the chunks and their respective metadata in the database. I also want to be able to export a single selectable chunk with all its metadata in a text file.
- As a tester, I want to be able to place a search query and receive the top k most similar chunks.
- As a tester, I want to be able to set the similarity metric (cosine, l2, inner product). Also a multiselect for the similarity metric is possible.
- As a tester, I want to get the full chunks with all metadata available as a response to the search query. The chunks text must be fully displayed in the GUI. The sorting must reflect the different similarity metrics, e. g. firstly show the top k results for cosine, then for l2, lastly for inner product.
- As an evaluator, I want to see the scoring of each chunk to understand the sorting by the most relevant chunks.
- As a tester, I want to provide additional context (like HITL context) so the LLM generates 2 extra search queries, and I can see results grouped by each query with full query text and metadata displayed.

# RULES
- all retrieval and chunk synthesis implementations must be identical to the one used in the deep researcher application src/ui/app.py. That is, the vector database selection and the automatical embedding model extraction must be identical. If you need to change the retrieval or chunk synthesis implementation, you must change it in the original implementation for src/ui/app.py, but set the existing solution (e.g. cosine similarity) as the default.
- The app must be testing/test_retrieval.py. Place a hint how to run the app in this file testing/test_retrieval.md

# RUN

```bash
# From project root directory:
uv run streamlit run testings/test_retrieval.py --server.port 8512

# Or with auto-reload for development:
uv run streamlit run testings/test_retrieval.py --server.port 8512 --server.runOnSave true
```

**Port 8512** is used to avoid conflicts with the main application (port 8511).

## Access

Once running, open your browser to:
- **Local URL**: http://localhost:8512
- **Network URL**: Will be displayed in terminal

## Features

### 1. Search Test Tab
- Execute searches with configurable number of results (k)
- Compare multiple similarity metrics side-by-side (cosine, L2, inner product)
- View full chunk text with metadata
- See relevance scores for each result
- **Additional Context (HITL)**: Provide optional context to generate 2 extra LLM-based search queries (uses `TASK_SEARCH_QUERIES_PROMPT` from the research pipeline). Results are grouped by source query with full query text displayed.

### 2. Database Explorer Tab
- Browse all chunks in the selected database
- Pagination support (10/25/50/100 chunks per page)
- View full chunk text and metadata
- Export individual chunks to text files

### 3. Export Tab
- Export search results to JSON format
- Individual chunk export (available in Database Explorer)
- Batch export functionality

## Configuration

Use the sidebar to configure:
- **Database Selection**: Choose from available ChromaDB databases
- **Number of Results (k)**: 1-20 results per query
- **Similarity Metrics**: Select one or more metrics for comparison
- **Language**: German or English (used for LLM query generation)

## Usage Guide

### Testing Retrieval Quality

1. **Select a Database**: Use the sidebar to choose which ChromaDB database to test
   - The embedding model will be automatically extracted and displayed

2. **Configure Retrieval Parameters**:
   - Set `k` (number of results) using the slider
   - Select one or more similarity metrics for comparison

3. **Execute Search**:
   - Enter your search query in the text input
   - Click "Execute Search"
   - Results will be displayed in separate tabs for each selected metric

4. **Analyze Results**:
   - Compare relevance scores across different metrics
   - Review full chunk text and metadata
   - Identify which chunks are being retrieved for your query

### Exploring Database Contents

1. **Navigate to Database Explorer Tab**
2. **Click "Load Chunks from Database"** to retrieve all chunks
3. **Browse Chunks**:
   - Use pagination controls to navigate through chunks
   - View full text and metadata for each chunk
4. **Export Individual Chunks**:
   - Click the export button within any chunk expander
   - Download includes full text and all metadata

### Comparing Configurations

To systematically compare different retrieval configurations:

1. **Baseline Test**:
   - Run search with current default settings
   - Export results for reference

2. **Vary k Parameter**:
   - Test with k=4, k=10, k=20
   - Observe how additional results affect relevance scores

3. **Compare Metrics** (limited by ChromaDB fixed distance function):
   - Note: All metrics currently use the same underlying L2 distance
   - Future enhancement: Create separate collections with different distance functions

4. **Document Findings**:
   - Use the export functionality to save results
   - Compare exports to identify optimal configurations

## Implementation Notes

- Reuses exact retrieval implementation from `src/ui/app.py`
- No LLM synthesis - focuses on raw retrieval quality
- Direct ChromaDB API access for multi-metric testing
- All embedding model handling identical to main application

## Troubleshooting

### Error: "No databases found in kb/database/"

**Cause**: The ChromaDB database directory doesn't exist or is empty.

**Solution**:
1. Verify that `kb/database/` exists in your project root
2. Check that it contains subdirectories with ChromaDB collections
3. Run the main application first to ensure databases are initialized

### Error: "Database not found: [path]"

**Cause**: Selected database name doesn't match any directory.

**Solution**:
1. Check that the database directory exists in `kb/database/`
2. Verify the directory structure (should contain either a `default` subdirectory or ChromaDB files directly)

### Error: "No collections found in database"

**Cause**: The database directory exists but doesn't contain a valid ChromaDB collection.

**Solution**:
1. Verify the database was properly initialized
2. Check for `chroma.sqlite3` file in the database directory or `default` subdirectory
3. Re-run database initialization if needed

### Search returns no results

**Possible causes**:
1. Query embedding model mismatch
2. Collection is empty
3. Query doesn't match any content semantically

**Solutions**:
1. Verify the embedding model in the sidebar matches the collection
2. Use Database Explorer to check if chunks exist
3. Try broader or different query terms

### Performance is slow

**Causes**:
1. Large database with many chunks
2. Multiple metric comparisons
3. Large k value

**Solutions**:
1. Reduce k value for faster searches
2. Test with a single metric first
3. Consider testing on a smaller database subset

# CURRENT STATUS

✅ **Phase 1 Complete**: Core Streamlit app structure
✅ **Phase 2 Complete**: Multi-metric search implementation
✅ **Phase 3 Complete**: Database explorer with pagination
✅ **Phase 4 Complete**: Chunk export functionality

## Known Limitations

1. **Similarity Metrics**: ChromaDB collections have a fixed distance function (usually L2). The current implementation uses the collection's native distance and converts it to similarity scores. True multi-metric comparison would require creating separate collections with different distance functions.

2. **No LLM Synthesis**: This tool focuses purely on retrieval quality. For testing the full pipeline including LLM extraction, use the main application.

## Quick Reference

### Key Files
- **Application**: `testings/test_retrieval.py` - Main Streamlit GUI
- **Documentation**: `testings/test_retrieval.md` - This file
- **Dependencies**: Uses existing `ChromaDBClient` from `src/services/chromadb_client.py`

### Command Cheat Sheet

```bash
# Run the testing tool
uv run streamlit run testings/test_retrieval.py --server.port 8512

# Run with auto-reload (for development)
uv run streamlit run testings/test_retrieval.py --server.port 8512 --server.runOnSave true

# Check syntax
uv run python -m py_compile testings/test_retrieval.py

# Test imports
uv run python -c "import testings.test_retrieval; print('OK')"
```

### Key Features Summary

| Feature | Status | Tab Location |
|---------|--------|--------------|
| Multi-metric search | ✅ Implemented | Search Test |
| Database selection | ✅ Implemented | Sidebar |
| Embedding model display | ✅ Implemented | Sidebar |
| Language selector | ✅ Implemented | Sidebar |
| Additional context (HITL) | ✅ Implemented | Search Test |
| LLM multi-query generation | ✅ Implemented | Search Test |
| Per-query result grouping | ✅ Implemented | Search Test |
| Chunk browsing | ✅ Implemented | Database Explorer |
| Pagination | ✅ Implemented | Database Explorer |
| Single chunk export | ✅ Implemented | Database Explorer |
| Batch JSON export | ✅ Implemented | Export |
| Relevance score display | ✅ Implemented | Search Test |
| Full chunk text view | ✅ Implemented | All tabs |
| Metadata display | ✅ Implemented | All tabs |

# TODO

- [ ] Add CSV export format
- [ ] Add markdown export format
- [ ] Implement batch chunk export from search results
- [ ] Add text search filter in Database Explorer
- [ ] Add metrics comparison visualization (side-by-side view)
- [ ] Add option to create collections with different distance functions for true multi-metric testing
- [ ] Add relevance score distribution histogram
- [ ] Add query history tracking
- [ ] Implement A/B comparison mode for different configurations
