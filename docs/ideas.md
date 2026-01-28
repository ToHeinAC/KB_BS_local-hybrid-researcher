# Alternative Implementation Ideas for KB_BS_local-hybrid-researcher

## Idea 1: Enhanced Phase 1 - Iterative Query Analysis with Multi-Vector Retrieval

### Overview
This implementation reimagines Phase 1 (Query Analysis + HITL Clarification) as an iterative, multi-vector retrieval approach that continuously refines user research intent through alternating vector database searches and human-in-the-loop interactions. Unlike the current conversation-only HITL, this approach integrates ChromaDB retrieval into each HITL iteration.

### Key Innovation
The BrAIn researcher context becomes more rigorous by having **direct vector database access during the HITL clarification phase**, not just in later research phases. Each iteration:
1. Generates 3 search queries (original + alternatives)
2. Retrieves ~9 document chunks from the selected database
3. Uses retrieval context to inform smarter follow-up questions
4. Refines queries based on user responses + identified knowledge gaps

---

## Implementation Specification

### State Structure (Compatible with `state.py`)

```python
# Existing fields (from current AgentState):
query: str                          # User's original query
query_analysis: dict                # Comprehensive analysis results
query_retrieval: str                # Accumulated retrieval results (appended each iteration)
hitl_active: bool                   # Whether HITL phase is active
hitl_iteration: int                 # Current iteration (0-indexed)
hitl_max_iterations: int            # Default: 5
hitl_conversation_history: list[dict]
hitl_termination_reason: str | None # "user_end", "max_iterations", "convergence"
coverage_score: float               # 0-1 estimate of information coverage
retrieval_history: dict             # Per-iteration retrieval metadata

# New fields for enhanced HITL:
iteration_queries: list[list[str]]   # [[q1, q2, q3], ...] per iteration
knowledge_gaps: list[str]            # Gaps identified from retrieval analysis
retrieval_dedup_ratios: list[float]  # Dedup ratio per iteration for convergence
```

### LangGraph Flow (Node Functions)

```
START
  ↓
hitl_init (existing - sets language, initializes state)
  ↓
┌──────────────── ITERATION LOOP ────────────────┐
│                                                 │
│  hitl_generate_queries (NEW)                   │
│  → Generate 3 queries: original + 2 alternatives│
│  → Uses query + analysis + user feedback        │
│  ↓                                              │
│  hitl_retrieve_chunks (NEW)                    │
│  → vector_search() for each query (k=3 each)   │
│  → Deduplicate against existing query_retrieval │
│  → Append new chunks to query_retrieval         │
│  ↓                                              │
│  hitl_analyze_retrieval (NEW)                  │
│  → LLM analysis of query + query_retrieval      │
│  → Extract: key_concepts, entities, scope, gaps │
│  → Calculate coverage_score                     │
│  ↓                                              │
│  hitl_generate_questions (ENHANCED)            │
│  → Generate 2-3 questions based on gaps         │
│  → Include coverage % in checkpoint             │
│  → END (wait for user response)                 │
│                                                 │
│  ─── user responds or types /end ───            │
│                                                 │
│  hitl_process_response (ENHANCED)              │
│  → Extract insights from user response          │
│  → IF /end or convergence → hitl_finalize      │
│  → ELSE → generate refined queries, loop back   │
│                                                 │
└─────────────────────────────────────────────────┘
  ↓
hitl_finalize (existing - generates research_queries for Phase 2)
  ↓
generate_todo (Phase 2)
```

### Node Implementations

#### `hitl_generate_queries` (NEW)

```python
def hitl_generate_queries(state: AgentState) -> dict:
    """Generate 3 search queries for current iteration."""
    iteration = state.get("hitl_iteration", 0)
    query = state["query"]
    analysis = state.get("query_analysis", {})
    
    if iteration == 0:
        # Initial: original + broader + alternative angle
        queries = generate_alternative_queries_llm(query, {}, iteration)
    else:
        # Refined: based on user feedback + knowledge gaps
        gaps = state.get("knowledge_gaps", [])
        last_response = state.get("hitl_conversation_history", [])[-1].get("content", "")
        queries = generate_refined_queries_llm(query, last_response, gaps)
    
    # Track all queries
    iteration_queries = list(state.get("iteration_queries", []))
    iteration_queries.append(queries)
    
    return {
        "iteration_queries": iteration_queries,
        "messages": [f"Generated {len(queries)} queries for iteration {iteration}"],
    }
```

#### `hitl_retrieve_chunks` (NEW)

```python
def hitl_retrieve_chunks(state: AgentState) -> dict:
    """Execute vector search and deduplicate results."""
    iteration = state.get("hitl_iteration", 0)
    queries = state.get("iteration_queries", [[]])[-1]
    selected_database = state.get("selected_database")
    k_per_query = 3
    
    all_chunks = []
    for q in queries:
        results = vector_search(q, top_k=k_per_query, selected_database=selected_database)
        all_chunks.extend(results)
    
    # Deduplicate against existing retrieval
    existing = state.get("query_retrieval", "")
    unique_chunks, dedup_stats = calculate_dedup_ratio(all_chunks, existing)
    
    # Append to query_retrieval
    formatted = format_chunks_for_state(unique_chunks, queries)
    new_retrieval = existing + "\n\n" + formatted if existing else formatted
    
    # Track dedup ratio for convergence
    dedup_ratios = list(state.get("retrieval_dedup_ratios", []))
    dedup_ratios.append(dedup_stats["dedup_ratio"])
    
    return {
        "query_retrieval": new_retrieval,
        "retrieval_dedup_ratios": dedup_ratios,
        "retrieval_history": {
            **state.get("retrieval_history", {}),
            f"iteration_{iteration}": {
                "queries": queries,
                "new_chunks": dedup_stats["new_count"],
                "duplicates": dedup_stats["dup_count"],
            }
        },
        "messages": [f"Retrieved {dedup_stats['new_count']} new chunks, {dedup_stats['dup_count']} duplicates"],
    }
```

#### `hitl_analyze_retrieval` (NEW)

```python
def hitl_analyze_retrieval(state: AgentState) -> dict:
    """Analyze accumulated retrieval for concepts, gaps, coverage."""
    query = state["query"]
    retrieval = state.get("query_retrieval", "")
    
    # LLM analysis
    analysis = analyze_retrieval_context_llm(query, retrieval)
    
    return {
        "query_analysis": analysis,
        "knowledge_gaps": analysis.get("knowledge_gaps", []),
        "coverage_score": analysis.get("coverage_score", 0.0),
        "messages": [f"Coverage: {analysis.get('coverage_score', 0):.0%}, gaps: {len(analysis.get('knowledge_gaps', []))}"],
    }
```

### Termination Criteria

```python
def should_terminate_hitl(state: AgentState) -> str:
    """Determine if HITL should end."""
    # Hard limit: max iterations
    if state.get("hitl_iteration", 0) >= state.get("hitl_max_iterations", 5):
        return "max_iterations"
    
    # Convergence: high coverage + high dedup ratio
    coverage = state.get("coverage_score", 0.0)
    dedup_ratios = state.get("retrieval_dedup_ratios", [])
    recent_dedup = dedup_ratios[-1] if dedup_ratios else 0.0
    gaps = len(state.get("knowledge_gaps", []))
    
    if coverage >= 0.80 and recent_dedup >= 0.70 and gaps <= 2:
        return "convergence"
    
    return "continue"
```

### LLM Prompt Templates

#### Generate Alternative Queries (Iteration 0)
```
Original user query: "{query}"

Generate 2 alternative search queries:
1. broader_scope: Explores related/contextual information
2. alternative_angle: Explores implications, challenges, or alternatives

Output JSON:
{"original": "{query}", "broader_scope": "...", "alternative_angle": "..."}
```

#### Analyze Retrieval Context
```
User's Research Query: {query}

Retrieved Context (from knowledge base):
{retrieval_text[:3000]}

Perform comprehensive analysis:
1. KEY CONCEPTS: 5-7 core concepts from query + retrieved content
2. ENTITIES: Named entities (organizations, dates, technical terms)
3. SCOPE: Primary focus, secondary topics, boundaries
4. KNOWLEDGE GAPS: Specific missing information (not "more details")
5. COVERAGE: 0-100% considering foundational, intermediate, advanced coverage

Output JSON with: key_concepts, entities, scope, knowledge_gaps, coverage_score
```

#### Generate Refined Queries (Iteration N>0)
```
Original query: "{query}"
User's clarification: "{user_response}"
Identified gaps: {gaps}

Generate 3 refined search queries:
1. Gap-addressing query
2. New concept exploration query
3. Scope clarification query

Output JSON: {"query_1": "...", "query_2": "...", "query_3": "..."}
```

### UI Display Enhancements

```
Iteration 2 of 5
├─ Queries: "Grenzwerte Strahlenexposition", "Strahlenschutzverordnung Dosisgrenzwerte", "Anwendung Grenzwerte Praxis"
├─ Retrieved: 7 new chunks, 2 duplicates skipped
├─ Coverage: 52%
└─ Gaps: 3 identified

Please answer:
1. You mentioned regulatory limits—are you interested in occupational or public exposure limits?
2. Should we focus on specific isotopes or general radiation types?
3. Is emergency response exposure relevant to your query?

(Type /end to finalize and proceed to research)
```

---

## Expected Outcomes

1. **Higher Quality Research**: Retrieval-informed questions lead to more targeted clarification
2. **Progressive Knowledge Building**: Each iteration adds to `query_retrieval` context
3. **Reduced Iterations**: Convergence detection prevents unnecessary loops
4. **Richer Context**: Accumulated knowledge provides better foundation for Phase 2
5. **Measurable Progress**: Coverage % and gap count give user clear feedback

---

## Integration Notes

### Backward Compatibility
- Existing `hitl_state` dict is extended, not replaced
- UI chat panel continues to work with enhanced checkpoint data
- Legacy HITL flow (`hitl_active=False`) unchanged

### Graph Routing
- New nodes inserted between `hitl_init` and `hitl_generate_questions`
- Existing `hitl_process_response` → `hitl_finalize` routing preserved
- Loop-back edge from `hitl_process_response` → `hitl_generate_queries`

### Performance Considerations
- `query_retrieval` capped at ~5000 tokens (sliding window if needed)
- Deduplication uses simple substring matching (semantic optional)
- Parallel retrieval possible but not required for baseline
