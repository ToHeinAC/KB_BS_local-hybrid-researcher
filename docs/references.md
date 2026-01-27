# References

## LangChain v1.0 Migration

This project targets **LangChain v1.0+** (released October 2025).

**Key migration resources:**
- [What's new in LangChain v1](https://docs.langchain.com/oss/python/releases/langchain-v1)
- [LangChain & LangGraph 1.0 Announcement](https://www.blog.langchain.com/langchain-langgraph-1dot0/)
- [Migration Guide](https://changelog.langchain.com/announcements/langchain-1-0-now-generally-available)

**Key v1.0 changes affecting this project:**
- Agent state must be `TypedDict` (not Pydantic)
- LangGraph is the foundational runtime for stateful agents
- Python 3.10+ required (3.9 dropped)
- `langchain-community` removed from core (use specific packages)

---

## External Implementation Resources

### Primary References

| # | Reference | URL | Extract |
|---|-----------|-----|---------|
| 1 | Local vectorstore researcher | [KB_BS_local-rag-he](https://github.com/ToHeinAC/KB_BS_local-rag-he) | GUI patterns, ChromaDB integration, HITL implementation |
| 2 | ReAct Agent with Ollama | [deepagents_ollama](https://github.com/ToHeinAC/deepagents_ollama) | Local Ollama integration, ReAct pattern |
| 3 | Document File Search | [agentic-file-search](https://github.com/PromtEngineer/agentic-file-search) | Dynamic document navigation, section extraction |
| 4 | Deep Agents | [deepagents](https://github.com/langchain-ai/deepagents) | Original deep research agent pattern |

### Video Tutorials

| Topic | URL |
|-------|-----|
| Agentic File Search | [YouTube](https://www.youtube.com/watch?v=rMADSuus6jg) |

---

## LangGraph Documentation

### Core Concepts

- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/)
- [StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph)
- [Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
- [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#human-in-the-loop)

### Patterns

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state
class AgentState(TypedDict):
    messages: list
    current_step: str

# Create graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("analyze", analyze_node)
graph.add_node("research", research_node)
graph.add_node("synthesize", synthesize_node)

# Add edges
graph.add_edge("analyze", "research")
graph.add_conditional_edges(
    "research",
    should_continue,
    {"continue": "research", "done": "synthesize"}
)
graph.add_edge("synthesize", END)

# Set entry point
graph.set_entry_point("analyze")

# Compile
app = graph.compile()
```

---

## LangChain Ollama Documentation

### Structured Output with json_mode

For Ollama models <30B parameters:

```python
from langchain_ollama import ChatOllama
from pydantic import BaseModel

class Output(BaseModel):
    answer: str
    confidence: float

llm = ChatOllama(model="qwen3:14b")

# Use json_mode for smaller models
structured_llm = llm.with_structured_output(Output, method="json_mode")

result = structured_llm.invoke("What is 2+2?")
# result is an Output instance
```

### References

- [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama/)
- [Structured Output](https://python.langchain.com/docs/how_to/structured_output/)

---

## ChromaDB Documentation

### Persistent Client

```python
import chromadb

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("documents")

# Add documents
collection.add(
    documents=["text1", "text2"],
    ids=["id1", "id2"],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

### References

- [ChromaDB Getting Started](https://docs.trychroma.com/getting-started)
- [LangChain Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/)

---

## Ollama Documentation

### Model Management

```bash
# Pull models
ollama pull qwen3:14b
ollama pull qwen3:8b

# List models
ollama list

# Run interactively
ollama run qwen3:14b

# API endpoint
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b",
  "prompt": "Hello"
}'
```

### References

- [Ollama](https://ollama.ai)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Models](https://ollama.ai/library)

---

## PyMuPDF Documentation

### PDF Text Extraction

```python
import fitz  # PyMuPDF

doc = fitz.open("document.pdf")

for page in doc:
    text = page.get_text()
    print(text)

doc.close()
```

### References

- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Text Extraction](https://pymupdf.readthedocs.io/en/latest/tutorial.html#extracting-text-and-images)

---

## Streamlit Documentation

### Session State

```python
import streamlit as st

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Update state
st.session_state.messages.append({"role": "user", "content": "Hello"})
```

### Safe Exit Button

```python
import streamlit as st
import subprocess

def safe_exit():
    """Kill Streamlit process on current port."""
    port = 8511  # or detect dynamically
    subprocess.run(
        f"lsof -ti:{port} | xargs -r kill -9",
        shell=True,
        capture_output=True
    )

if st.button("Exit"):
    safe_exit()
```

### References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Session State](https://docs.streamlit.io/library/api-reference/session-state)

---

## Integration Pattern

```
Existing RAG Pipeline (KB_BS_local-rag-he)
    ↓
Enhanced with reference-following logic (Rabbithole steps)
    ↓
Wrapped in LangGraph StateGraph (state management + routing)
    ↓
Exposed via Streamlit UI (HITL + results display)
```

---

## Success Criteria

- [ ] System correctly identifies and follows cross-document references
- [ ] ResearchContext grows with each iteration, capturing relationship trees
- [ ] ToDoList dynamically adapts as new information emerges
- [ ] Final report includes precise citations to original sections
- [ ] Quality scores accurately reflect answer completeness
- [ ] System terminates reliably (respects TODO_MAX_ITEMS and max_iterations)
- [ ] HITL clarification improves research focus
- [ ] Performance acceptable for 50-200 page documents
