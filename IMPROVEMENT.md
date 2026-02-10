# Graded Context Management & Query-Anchored Synthesis

## Executive Summary

This document outlines improvements to prevent **query drift** during deep reference following and ensure the final answer remains anchored to the user's original intent. The core insight: the current system accumulates rich context during HITL and rabbithole phases but fails to leverage this hierarchically in synthesis.

---

## Current Weaknesses Analysis

### 1. Query Drift in Rabbithole Magic

**Problem**: Deep reference following (depth 2) can pursue tangential references without relevance checks until *after* extraction.

**Example Flow**:
```
User Query: "Grenzwerte für berufliche Strahlenexposition"
  → Task: Find dose limits in StrlSchG
    → Chunk mentions "§78 StrlSchG"
      → Rabbithole follows §78 (relevant ✓)
        → §78 mentions "KTA 1401" (tangential?)
          → Follows KTA 1401 (drift begins)
            → KTA 1401 mentions "DIN standards"
              → Now 3 hops from original query
```

**Current Mitigation**: Relevance threshold (0.6) applied *after* extraction - tokens already spent.

### 2. HITL Context Loss

**Problem**: `query_retrieval` (accumulated during HITL iterations) is NOT used in final synthesis.

```python
# In synthesize() node - current implementation
prompt = SYNTHESIS_PROMPT.format(
    original_query=analysis.original_query,
    findings=info_text,  # ← Only research_context, NO hitl findings
    language=analysis.detected_language,
)
```

**Lost Value**:
- HITL retrieval often contains highly relevant chunks discovered during clarification
- User's clarifications refine understanding but aren't referenced in answer
- `hitl_conversation_history` captures intent clarification - unused

### 3. Flat Context Priority

**Problem**: All sources treated equally regardless of origin/confidence.

| Source Type | Current Weight | Should Be |
|-------------|----------------|-----------|
| Direct vector search | 1.0 | 1.0 (highest) |
| Rabbithole depth-1 | 1.0 | 0.8 |
| Rabbithole depth-2 | 1.0 | 0.6 |
| HITL retrieval | 0 (unused) | 0.5 (supporting) |

### 4. Summarization Loses Verbatim Quotes

**Problem**: `INFO_EXTRACTION_PROMPT` condenses chunks, potentially losing critical exact wording.

**Example**:
```
Original: "Der Grenzwert der effektiven Dosis beträgt für beruflich strahlenexponierte
           Personen 20 Millisievert im Kalenderjahr (§ 78 Abs. 1 Nr. 1 StrlSchG)."

Extracted: "Grenzwert 20 mSv/Jahr für beruflich exponierte Personen"
```

The legal precision ("effektiven Dosis", "Kalenderjahr", exact citation) is lost.

### 5. Weak Language Enforcement

**Problem**: Output language controlled only by `{language}` variable in prompt - LLM can still mix languages.

### 6. No Per-Task Structured Summary

**Problem**: Tasks complete without structured summary that preserves:
- Key findings with source attribution
- Verbatim quotes for critical claims
- Relevance score to original query

---

## Proposed Architecture: Graded Context Management

### Core Concept: Context Tiers

```
┌──────────────────────────────────────────────────────────────────┐
│  TIER 1: Primary Context (weight 1.0)                            │
│  ├─ Direct vector search results for current task                │
│  ├─ Highest relevance score chunks (≥0.85)                       │
│  └─ Explicitly mentioned in user's original query                │
├──────────────────────────────────────────────────────────────────┤
│  TIER 2: Secondary Context (weight 0.7)                          │
│  ├─ Rabbithole depth-1 references (direct citations)             │
│  ├─ Medium relevance score chunks (0.6-0.85)                     │
│  └─ Related to HITL-identified entities                          │
├──────────────────────────────────────────────────────────────────┤
│  TIER 3: Tertiary Context (weight 0.4)                           │
│  ├─ Rabbithole depth-2 references                                │
│  ├─ HITL retrieval chunks (query_retrieval)                      │
│  └─ Supporting context (background, definitions)                 │
├──────────────────────────────────────────────────────────────────┤
│  TIER 4: Meta Context (not in synthesis, for validation)         │
│  ├─ hitl_conversation_history (user intent)                      │
│  ├─ knowledge_gaps (what's missing)                              │
│  └─ coverage_score (completeness estimate)                       │
└──────────────────────────────────────────────────────────────────┘
```

### New State Variables

```python
# In AgentState (src/agents/state.py)

# Graded context tracking
primary_context: list[dict]      # Tier 1: Direct, high-relevance findings
secondary_context: list[dict]    # Tier 2: Reference-followed, medium-relevance
tertiary_context: list[dict]     # Tier 3: Deep references, HITL retrieval
hitl_context_summary: str        # Synthesized HITL conversation insights

# Per-task summaries
task_summaries: list[dict]       # Structured summary per completed task
# Structure: {"task_id": int, "summary": str, "key_quotes": list[dict],
#             "sources": list[str], "relevance_to_query": float}

# Query anchor (prevents drift)
query_anchor: dict               # Immutable reference to original intent
# Structure: {"original_query": str, "detected_language": str,
#             "key_entities": list[str], "scope": str,
#             "hitl_refinements": list[str]}

# Verbatim quote tracking
preserved_quotes: list[dict]     # Critical exact wording
# Structure: {"quote": str, "source": str, "page": int,
#             "relevance_reason": str}
```

---

## Implementation Plan

### Phase A: Query Anchor & HITL Context Preservation

#### A.1 Create Immutable Query Anchor

**File**: `src/agents/nodes.py`
**Location**: `hitl_finalize()` node

**Change**: Create `query_anchor` that persists throughout all phases.

```python
# After HITL finalization, create immutable anchor
query_anchor = {
    "original_query": state["query"],
    "detected_language": language,
    "key_entities": result.get("entities", []),
    "scope": result.get("scope", ""),
    "hitl_refinements": [
        msg["content"] for msg in hitl_conversation_history
        if msg["role"] == "user"
    ],
    "created_at": datetime.now().isoformat(),
}
```

#### A.2 Preserve HITL Context for Synthesis

**File**: `src/agents/nodes.py`
**Location**: `hitl_finalize()` node

**Change**: Create `hitl_context_summary` from HITL conversation and retrieval.

```python
# Summarize HITL findings for later use
hitl_context_summary = _summarize_hitl_context(
    query=state["query"],
    conversation=hitl_conversation_history,
    retrieval=state.get("query_retrieval", ""),
    knowledge_gaps=state.get("knowledge_gaps", []),
    language=language,
)
```

**New Prompt** (`src/prompts.py`):

```python
HITL_CONTEXT_SUMMARY_PROMPT = """Summarize the research clarification conversation.

Original Query: "{query}"

Conversation:
{conversation}

Retrieved Context (from knowledge base during clarification):
{retrieval}

Identified Knowledge Gaps:
{gaps}

Create a concise summary (in {language}) covering:
1. User's refined intent and scope
2. Key clarifications from the conversation
3. Most relevant findings from retrieval
4. Remaining gaps to address

This summary will guide the final answer synthesis.

Summary:"""
```

#### A.3 State Updates

**File**: `src/agents/state.py`

```python
# Add to AgentState
query_anchor: dict               # Immutable query reference
hitl_context_summary: str        # Summarized HITL findings
primary_context: list[dict]      # Tier 1 context
secondary_context: list[dict]    # Tier 2 context
tertiary_context: list[dict]     # Tier 3 context
task_summaries: list[dict]       # Per-task structured summaries
preserved_quotes: list[dict]     # Critical verbatim quotes
```

---

### Phase B: Graded Context Classification

#### B.1 Context Classifier

**File**: `src/agents/tools.py`
**New Function**: `classify_context_tier()`

```python
def classify_context_tier(
    chunk: ChunkWithInfo,
    query_anchor: dict,
    depth: int = 0,
    source_type: str = "vector_search"
) -> tuple[int, float]:
    """Classify chunk into context tier with weight.

    Returns:
        (tier, weight): 1-3 tier and 0.0-1.0 weight
    """
    # Tier 1: Direct search, high relevance, matches query entities
    if (source_type == "vector_search" and
        depth == 0 and
        chunk.relevance_score >= 0.85):

        # Bonus if matches query entities
        entity_match = any(
            entity.lower() in chunk.chunk.lower()
            for entity in query_anchor.get("key_entities", [])
        )
        weight = 1.0 if entity_match else 0.95
        return (1, weight)

    # Tier 2: Depth-1 references or medium relevance
    if depth == 1 or (0.6 <= chunk.relevance_score < 0.85):
        weight = 0.7 * chunk.relevance_score
        return (2, weight)

    # Tier 3: Depth-2+ or HITL retrieval
    weight = 0.4 * chunk.relevance_score
    return (3, weight)
```

#### B.2 Integrate into Task Execution

**File**: `src/agents/nodes.py`
**Location**: `execute_task()` node

**Change**: Classify chunks during accumulation.

```python
# After creating chunk_with_info
tier, weight = classify_context_tier(
    chunk=chunk_with_info,
    query_anchor=state["query_anchor"],
    depth=context.metadata.get("current_depth", 0),
    source_type="vector_search"
)

# Add tier metadata
chunk_with_info_dict = chunk_with_info.model_dump()
chunk_with_info_dict["context_tier"] = tier
chunk_with_info_dict["context_weight"] = weight

# Accumulate by tier
if tier == 1:
    primary_context.append(chunk_with_info_dict)
elif tier == 2:
    secondary_context.append(chunk_with_info_dict)
else:
    tertiary_context.append(chunk_with_info_dict)
```

---

### Phase C: Verbatim Quote Preservation

#### C.1 Quote Extraction During Info Extraction

**File**: `src/agents/tools.py`
**Modify**: `extract_info()` function

**Change**: Return both condensed info AND preserved quotes.

```python
def extract_info_with_quotes(
    chunk_text: str,
    query: str,
    query_anchor: dict,
    client: OllamaClient,
) -> dict:
    """Extract info while preserving critical verbatim quotes.

    Returns:
        {"extracted_info": str, "preserved_quotes": list[dict]}
    """
    result = client.generate_structured(
        INFO_EXTRACTION_WITH_QUOTES_PROMPT.format(
            query=query,
            chunk_text=chunk_text,
            key_entities=", ".join(query_anchor.get("key_entities", [])),
        ),
        InfoExtractionWithQuotes
    )
    return result.model_dump()
```

**New Prompt** (`src/prompts.py`):

```python
INFO_EXTRACTION_WITH_QUOTES_PROMPT = """Given this search query: "{query}"
Key entities to look for: {key_entities}

Extract relevant information from this text chunk.

Text chunk:
{chunk_text}

Provide:
1. extracted_info: Condensed relevant passages (same language as chunk)
2. preserved_quotes: List of exact verbatim quotes that are critical for accuracy
   - Legal definitions with exact numbers/thresholds
   - Technical specifications
   - Named regulations with section numbers
   - Statements that should not be paraphrased

Format for preserved_quotes:
[{{"quote": "exact text", "relevance_reason": "why this must be verbatim"}}]

Return as JSON."""
```

**New Pydantic Model** (`src/models/research.py`):

```python
class PreservedQuote(BaseModel):
    quote: str
    relevance_reason: str
    source: str = ""
    page: int = 0

class InfoExtractionWithQuotes(BaseModel):
    extracted_info: str
    preserved_quotes: list[PreservedQuote]
```

---

### Phase D: Per-Task Structured Summary

#### D.1 Task Summary Generation

**File**: `src/agents/nodes.py`
**Location**: After each task completion in `execute_task()`

```python
def _generate_task_summary(
    task: ToDoItem,
    chunks: list[ChunkWithInfo],
    query_anchor: dict,
    client: OllamaClient,
) -> dict:
    """Generate structured summary for completed task."""

    # Collect all preserved quotes from this task's chunks
    task_quotes = []
    for chunk in chunks:
        task_quotes.extend(chunk.get("preserved_quotes", []))

    # Calculate relevance to original query
    relevance_score = _calculate_task_relevance(
        task_text=task.task,
        original_query=query_anchor["original_query"],
        key_entities=query_anchor["key_entities"],
    )

    summary = client.generate_structured(
        TASK_SUMMARY_PROMPT.format(
            task=task.task,
            original_query=query_anchor["original_query"],
            findings=json.dumps([c.extracted_info for c in chunks[:5]]),
            preserved_quotes=json.dumps(task_quotes[:3]),
            language=query_anchor["detected_language"],
        ),
        TaskSummary
    )

    return {
        "task_id": task.id,
        "task_text": task.task,
        "summary": summary.summary,
        "key_findings": summary.key_findings,
        "preserved_quotes": task_quotes,
        "sources": [c.document for c in chunks],
        "relevance_to_query": relevance_score,
    }
```

**New Prompt** (`src/prompts.py`):

```python
TASK_SUMMARY_PROMPT = """Summarize the findings for this research task.

Task: "{task}"
Original Research Query: "{original_query}"

Findings:
{findings}

Critical Quotes (preserve exactly):
{preserved_quotes}

Create a summary (in {language}) that:
1. Directly addresses how this task contributes to answering the original query
2. Includes key facts with source citations
3. Preserves any critical verbatim quotes
4. Notes any gaps or limitations

Return JSON with:
- summary: Concise task summary
- key_findings: List of discrete findings
- gaps: Any identified gaps"""
```

---

### Phase E: Query-Anchored Synthesis

#### E.1 Enhanced Synthesis Prompt

**File**: `src/prompts.py`
**Modify**: `SYNTHESIS_PROMPT`

```python
SYNTHESIS_PROMPT_ENHANCED = """Synthesize research findings into a comprehensive answer.

CRITICAL: Your answer MUST directly address the original query. Do not drift into tangential topics.

## Original Query
"{original_query}"

## User Intent (from clarification)
{hitl_context_summary}

## Primary Findings (highest confidence, use these first)
{primary_findings}

## Supporting Findings (use to add depth)
{secondary_findings}

## Background Context (use only if gaps remain)
{tertiary_findings}

## Critical Verbatim Quotes (include exactly as written when relevant)
{preserved_quotes}

## Task Summaries
{task_summaries}

---

INSTRUCTIONS:
1. Answer ONLY in {language}. Do not mix languages.
2. Begin with a direct answer to the original query.
3. Support claims with citations: [Document.pdf, Page X]
4. Include preserved quotes for legal/technical precision.
5. Acknowledge gaps identified during research.
6. Structure: Overview → Details → Limitations

Provide:
1. summary: Comprehensive answer (strictly in {language})
2. key_findings: List of most important findings
3. query_coverage: How completely the query was answered (0-100)
4. remaining_gaps: Any unanswered aspects"""
```

#### E.2 Modified Synthesis Node

**File**: `src/agents/nodes.py`
**Modify**: `synthesize()` node

```python
def synthesize(state: AgentState) -> AgentState:
    """Synthesize findings with graded context and query anchoring."""

    query_anchor = state["query_anchor"]

    # Prepare tiered findings
    primary_findings = _format_findings(state["primary_context"][:15])
    secondary_findings = _format_findings(state["secondary_context"][:10])
    tertiary_findings = _format_findings(state["tertiary_context"][:5])

    # Collect preserved quotes
    all_quotes = state.get("preserved_quotes", [])
    quotes_text = json.dumps(all_quotes[:10], ensure_ascii=False)

    # Format task summaries
    task_summaries = state.get("task_summaries", [])
    summaries_text = "\n".join([
        f"Task {ts['task_id']}: {ts['summary']}"
        for ts in task_summaries
    ])

    prompt = SYNTHESIS_PROMPT_ENHANCED.format(
        original_query=query_anchor["original_query"],
        hitl_context_summary=state.get("hitl_context_summary", ""),
        primary_findings=primary_findings,
        secondary_findings=secondary_findings,
        tertiary_findings=tertiary_findings,
        preserved_quotes=quotes_text,
        task_summaries=summaries_text,
        language=query_anchor["detected_language"],
    )

    result = client.generate_structured(prompt, SynthesisOutputEnhanced)

    # ... rest of synthesis logic
```

---

### Phase F: Strict Language Enforcement

#### F.1 Language Validation Layer

**File**: `src/services/ollama_client.py`
**New Method**: `generate_structured_with_language()`

```python
def generate_structured_with_language(
    self,
    prompt: str,
    response_model: type[T],
    target_language: str,
    max_retries: int = 2,
) -> T:
    """Generate structured output with language enforcement.

    If output contains significant other-language content, retry with
    stronger language instruction.
    """
    result = self.generate_structured(prompt, response_model)

    # Validate language (simple heuristic)
    if not self._validate_language(result, target_language):
        # Retry with explicit language enforcement
        enforced_prompt = f"""WICHTIG/IMPORTANT: Antworte NUR auf {target_language}.
DO NOT use any other language.

{prompt}"""
        result = self.generate_structured(enforced_prompt, response_model)

    return result

def _validate_language(self, result: BaseModel, target: str) -> bool:
    """Check if result is in target language."""
    text = str(result.model_dump())

    # Simple heuristic: check for common German/English markers
    german_markers = ["der", "die", "das", "und", "ist", "für", "mit"]
    english_markers = ["the", "and", "is", "for", "with", "of", "to"]

    german_count = sum(1 for m in german_markers if f" {m} " in text.lower())
    english_count = sum(1 for m in english_markers if f" {m} " in text.lower())

    if target == "de":
        return german_count >= english_count
    else:
        return english_count >= german_count
```

#### F.2 Language-Specific Output Models

**File**: `src/models/results.py`

```python
class SynthesisOutputEnhanced(BaseModel):
    summary: str = Field(
        ...,
        description="Comprehensive answer in the specified language only"
    )
    key_findings: list[str]
    query_coverage: int = Field(
        ...,
        ge=0,
        le=100,
        description="How completely the query was answered"
    )
    remaining_gaps: list[str] = Field(
        default_factory=list,
        description="Unanswered aspects of the query"
    )
```

---

### Phase G: Pre-Synthesis Relevance Check

#### G.1 Query Relevance Validator

**File**: `src/agents/nodes.py`
**New Node**: `validate_relevance()`

```python
def validate_relevance(state: AgentState) -> AgentState:
    """Validate accumulated context is relevant to original query.

    Runs before synthesis to filter drift.
    """
    query_anchor = state["query_anchor"]

    # Score each context item against original query
    scored_primary = []
    for item in state["primary_context"]:
        relevance = _score_query_relevance(
            text=item["extracted_info"],
            query=query_anchor["original_query"],
            entities=query_anchor["key_entities"],
        )
        if relevance >= 0.5:  # Only keep if relevant
            item["final_relevance"] = relevance
            scored_primary.append(item)

    # Sort by relevance
    scored_primary.sort(key=lambda x: x["final_relevance"], reverse=True)

    # Log drift detection
    original_count = len(state["primary_context"])
    filtered_count = len(scored_primary)
    if filtered_count < original_count * 0.7:
        logger.warning(
            f"Drift detected: filtered {original_count - filtered_count} "
            f"items from primary context"
        )

    return {
        **state,
        "primary_context": scored_primary,
        # Apply similar filtering to secondary/tertiary
    }
```

**New Prompt** (`src/prompts.py`):

```python
RELEVANCE_SCORING_PROMPT = """Score how relevant this text is to answering the query.

Query: "{query}"
Key entities: {entities}

Text: "{text}"

Score from 0-100:
- 100: Directly answers the query
- 75: Provides key supporting information
- 50: Related but tangential
- 25: Only loosely connected
- 0: Irrelevant

Return JSON: {{"relevance_score": N, "reasoning": "brief explanation"}}"""
```

---

## Graph Modifications

### Updated Graph Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: HITL (unchanged)                                          │
│  hitl_init → generate_queries → retrieve → analyze → questions     │
│  → process_response → [loop or finalize]                            │
├─────────────────────────────────────────────────────────────────────┤
│  NEW: hitl_finalize creates:                                        │
│  - query_anchor (immutable)                                         │
│  - hitl_context_summary (for synthesis)                             │
│  - tertiary_context (from query_retrieval)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 2: ToDo Generation (unchanged)                               │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3: Task Execution (MODIFIED)                                 │
│  FOR EACH TASK:                                                     │
│    Vector Search → Extract Info + Quotes → Classify Tier →          │
│    Reference Detection → Reference Resolution (with tier) →         │
│    Accumulate by Tier → Generate Task Summary → Next Task           │
├─────────────────────────────────────────────────────────────────────┤
│  NEW: validate_relevance                                            │
│  Filter accumulated context against query_anchor                    │
│  Log and warn on significant drift                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 4: Synthesis (MODIFIED)                                      │
│  Uses: primary → secondary → tertiary → hitl_context_summary        │
│  Includes: preserved_quotes, task_summaries                         │
│  Enforces: strict language output                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 4b: Quality Check (enhanced)                                 │
│  NEW dimension: query_adherence (0-100)                             │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 5: Source Attribution (unchanged)                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Additions

**File**: `.env` / `src/config.py`

```python
# Graded Context Settings
CONTEXT_TIER1_MAX_ITEMS = 15      # Max primary context items
CONTEXT_TIER2_MAX_ITEMS = 10      # Max secondary context items
CONTEXT_TIER3_MAX_ITEMS = 5       # Max tertiary context items

# Quote Preservation
MAX_PRESERVED_QUOTES = 10         # Max quotes in synthesis prompt
QUOTE_MIN_LENGTH = 20             # Min chars for a quote to preserve

# Relevance Filtering
DRIFT_FILTER_THRESHOLD = 0.5      # Min relevance to keep in final context
DRIFT_WARNING_RATIO = 0.7         # Warn if more than 30% filtered

# Language Enforcement
LANGUAGE_VALIDATION_ENABLED = True
LANGUAGE_RETRY_ON_MISMATCH = True
```

---

## Testing Strategy

### Unit Tests

1. **Context Classification** (`tests/test_graded_context.py`)
   - Test tier assignment for various chunk scenarios
   - Test weight calculation
   - Test entity matching bonus

2. **Quote Preservation** (`tests/test_quote_preservation.py`)
   - Test extraction of legal citations
   - Test technical specification preservation
   - Test quote deduplication

3. **Task Summaries** (`tests/test_task_summaries.py`)
   - Test summary generation
   - Test quote inclusion
   - Test relevance scoring

4. **Language Validation** (`tests/test_language_enforcement.py`)
   - Test language detection
   - Test retry mechanism
   - Test mixed-language handling

### Integration Tests

1. **Full Pipeline with Drift Detection**
   - Inject known tangential references
   - Verify filtering works
   - Check final answer stays on topic

2. **HITL Context Usage**
   - Verify query_retrieval flows to tertiary_context
   - Verify hitl_context_summary appears in synthesis
   - Check user clarifications influence answer

### End-to-End Verification

```bash
# Run with drift-prone query
python -m src.main --query "Was sind die Grenzwerte für Strahlenexposition?" \
    --verbose --log-drift

# Expected output should:
# 1. Stay focused on dose limits
# 2. Include HITL refinements in answer
# 3. Preserve exact legal quotes
# 4. Output entirely in German
```

---

## Implementation Order

| Phase | Files Modified | Estimated Effort | Dependencies |
|-------|----------------|------------------|--------------|
| A | state.py, nodes.py, prompts.py | Medium | None |
| B | tools.py, nodes.py | Medium | Phase A |
| C | tools.py, prompts.py, models/research.py | Medium | Phase A |
| D | nodes.py, prompts.py | Medium | Phase C |
| E | prompts.py, nodes.py | Medium | Phases A-D |
| F | ollama_client.py, models/results.py | Low | None |
| G | nodes.py, prompts.py, graph.py | Medium | Phases A-B |

**Recommended Order**: A → F → B → C → D → G → E

---

## Success Metrics

1. **Query Adherence**: Final answer directly addresses 95%+ of original query
2. **Drift Reduction**: <20% of accumulated context filtered in validation
3. **Quote Preservation**: Critical legal/technical quotes preserved verbatim
4. **Language Purity**: 100% single-language output (no mixing)
5. **HITL Utilization**: hitl_context_summary referenced in 80%+ of answers
6. **Source Traceability**: All claims traceable to tiered sources

---

## Migration Notes

### Backward Compatibility

- Existing `research_context` structure unchanged
- New fields are additive (not replacing existing)
- Old state files will work (new fields default to empty)

### Deprecation Path

After full implementation:
- `query_retrieval` → tertiary_context (structured)
- Flat context accumulation → tiered accumulation
- Basic `SYNTHESIS_PROMPT` → `SYNTHESIS_PROMPT_ENHANCED`

---

## Summary

This improvement addresses the fundamental weakness of query drift by:

1. **Anchoring**: Immutable `query_anchor` preserves original intent
2. **Grading**: Context tiered by confidence/source type
3. **Preserving**: Verbatim quotes for legal/technical precision
4. **Summarizing**: Per-task structured summaries
5. **Validating**: Pre-synthesis relevance check
6. **Enforcing**: Strict language output

The user's original query remains the north star throughout the entire pipeline, with HITL insights and rabbithole discoveries properly weighted and filtered before synthesis.
