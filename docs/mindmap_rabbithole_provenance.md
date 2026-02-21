# Mindmap: Rabbithole Provenance — Context-Aware Reference Following

## 1. Problem: Lost Reference Context

### Root Cause

When a primary chunk (depth 0, Tier 1) contains a reference and that reference is followed to
retrieve a nested chunk (depth 1, Tier 2/3), the nested chunk is evaluated for relevance to the
research query **in isolation**. The surrounding context of the reference in the parent chunk is
permanently discarded after the agentic gate decision.

### Misinterpretation Scenario

```
Primary chunk A (StrlSchV.pdf, page 45, tier=1):
  "[RELEVANT] Annual dose limit is 20 mSv for workers..."
  "[IRRELEVANT] For waste disposal see Kreislaufwirtschaftsgesetz §21"
                                 ↑ reference detected here

→ Agentic gate runs: gets surrounding_window (3000 chars, computed at nodes.py:509)
  Gate decides: FOLLOW (§21 looks radiation-adjacent)

→ resolve_reference_enhanced("§21") → NestedChunk B (KrWG.pdf, page 21):
  "§21 KrWG: radioactive waste must be stored in certified facilities..."

→ B scored against query (dose limits for radiation workers):
  "radioactive" entity match → boosted score → key_finding!

→ Task summary LLM includes B as key finding about dose limits
  User gets: "Radioactive waste must be stored in certified facilities"
  as dose-limit evidence — MISINTERPRETATION
```

### Why This Happened

The `surrounding_window` variable is already computed at **`nodes.py:509`** and passed to the
agentic gate. After the gate decides to follow, this context is never attached to the resulting
nested chunks. The chunks are therefore scored and formatted without knowing *where in the parent
chunk* the reference appeared.

---

## 2. Solution Architecture (4 Layers)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Data Model                                                     │
│  NestedChunk gains 5 provenance fields (all optional, default "")        │
│  parent_document, parent_page, reference_original_text,                  │
│  reference_type, reference_surrounding_context                           │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Context Entry                                                  │
│  create_tiered_context_entry() accepts + stores provenance               │
│  (only when depth > 0 AND parent_document is non-empty)                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 3: execute_task() — Propagation                                   │
│  After agentic gate, attach provenance to each nc before classification  │
│  surrounding_window (already computed for gate) is reused here           │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 4: LLM Scoring + Formatting                                       │
│  _rerank_task_chunks(): passes parent_context to CHUNK_RERANKER_PROMPT   │
│  _format_ranked_findings(): adds "[via ref ...]" + "Parent context:" headers │
│  TASK_SUMMARY_PROMPT_SYSTEM: new rule 2e caps off-topic ref chunks       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Details

### 3.1 `src/models/research.py` — NestedChunk

Added 5 optional provenance fields:

```python
class NestedChunk(BaseModel):
    # existing fields ...
    parent_document: str = ""
    parent_page: int | None = None
    reference_original_text: str = ""   # ref.original_text
    reference_type: str = ""            # ref.type
    reference_surrounding_context: str = ""  # surrounding_window[:500]
```

All fields default to empty / None for backward compatibility with existing tests and
resolution functions that construct `NestedChunk` without provenance context.

### 3.2 `src/agents/tools.py` — create_tiered_context_entry()

Added 5 matching optional kwargs. Provenance keys are written to the entry dict **only** when
`depth > 0` and `parent_document` is non-empty:

```python
if depth > 0 and parent_document:
    entry["parent_document"] = parent_document
    entry["parent_page"] = parent_page
    entry["reference_original_text"] = reference_original_text
    entry["reference_type"] = reference_type
    entry["reference_surrounding_context"] = reference_surrounding_context[:2000]
```

Depth-0 (direct vector search) entries never carry provenance keys, keeping dict size
unchanged for the majority of chunks.

### 3.3 `src/agents/nodes.py` — execute_task()

**Before the `try:` block** (agentic gate), `surrounding_window = ""` is initialized to ensure
it is always defined even if the try block fails before reaching the `get_context_window()` call.

**After `ref.nested_chunks = nested`** and before the classification loop:

```python
for nc in nested:
    nc.parent_document = chunk.document
    nc.parent_page = chunk.page
    nc.reference_original_text = ref.original_text
    nc.reference_type = ref.type
    nc.reference_surrounding_context = surrounding_window[:2000]
```

`surrounding_window` is reused here — it was already computed for the agentic gate (line 509),
so this adds zero LLM calls.

The provenance kwargs are then forwarded to `create_tiered_context_entry(...)` for the nc_entry.

### 3.4 `src/agents/nodes.py` — _format_ranked_findings()

For each chunk with `parent_document`, a provenance line is appended after the standard header:

```
[Rank N | Score N/100 | KrWG.pdf, Page 21]  // reasoning
[via legal_section ref "§21 KrWG" in StrlSchV.pdf, Page 45]
Parent context: "Annual dose limit is 20 mSv. For waste disposal see §21 KrWG...."
<extracted text>
```

The `TASK_SUMMARY_PROMPT_SYSTEM` can now read the parent context and apply rule 2e.

### 3.5 `src/agents/nodes.py` — _rerank_task_chunks()

The reranker now extracts `reference_surrounding_context` from the chunk dict and passes it as
`parent_context` to `CHUNK_RERANKER_PROMPT_HUMAN`. Direct search results get
`"N/A (direct vector search result)"`.

### 3.6 `src/prompts/research.py` — CHUNK_RERANKER_PROMPT_SYSTEM`

Added rule 2 (with old rules 2-4 renumbered to 3-5):

```
2. If parent_context is provided (not "N/A"), this chunk was found by following a reference.
   Assess whether the reference appeared in a relevant part of the parent:
   - If parent_context is clearly off-topic → penalise the score by 20-40 points
   - If parent_context is on-topic → score normally
   Always state this assessment in your reasoning.
```

`CHUNK_RERANKER_PROMPT_HUMAN` now includes:
```
- parent_context: {parent_context}
```

### 3.7 `src/prompts/research.py` — TASK_SUMMARY_PROMPT_SYSTEM`

Added rule 2e to `<processing_rules>`:

```
e. If a passage has a "[via ref ...]" header, check the "Parent context:" line.
   If the parent context is unrelated to original_query, treat the passage as
   supplementary evidence only (cap its effective Score at 49 regardless of its
   own content) and note this in gaps.
```

---

## 4. Data Flow After This Change

```
execute_task()
│
├─ [depth=0] Vector search → chunk A (StrlSchV.pdf, p.45)
│   ├─ extracted_info: "Annual dose limit 20 mSv ... waste disposal §21 KrWG"
│   ├─ detect_references → ref: {type=legal_section, target="§21 KrWG",
│   │                             original_text="Kreislaufwirtschaftsgesetz §21"}
│   ├─ surrounding_window = get_context_window(extracted_info, "Kreislaufwirtschaftsgesetz §21")
│   │   → "...Annual dose limit 20 mSv for workers. For waste disposal see §21 KrWG."
│   ├─ Agentic gate decides: FOLLOW
│   ├─ resolve_reference_enhanced → nested chunk B (KrWG.pdf, p.21)
│   │
│   ├─ ATTACH PROVENANCE to B:
│   │   B.parent_document = "StrlSchV.pdf"
│   │   B.parent_page = 45
│   │   B.reference_original_text = "Kreislaufwirtschaftsgesetz §21"
│   │   B.reference_type = "legal_section"
│   │   B.reference_surrounding_context = "...Annual dose limit 20 mSv...waste §21..."
│   │
│   └─ create_tiered_context_entry(B, tier=2, depth=1, parent_document="StrlSchV.pdf", ...)
│       → secondary_context entry with all 5 provenance keys
│
├─ _rerank_task_chunks()
│   └─ B: parent_context = "...Annual dose limit 20 mSv...waste §21..."
│       → CHUNK_RERANKER_PROMPT: "parent_context is provided; dose limit is on-topic
│          but reference appeared in waste-disposal sentence → off-topic → -30 penalty"
│       → score: 25/100 (was 55/100 before this change)
│
└─ _format_ranked_findings(ranked_chunks)
    └─ B header: "[Rank 4 | Score 25/100 | KrWG.pdf, Page 21]
                  [via legal_section ref "Kreislaufwirtschaftsgesetz §21" in StrlSchV.pdf, Page 45]
                  Parent context: "...Annual dose limit 20 mSv...waste §21..."
                  §21 KrWG: radioactive waste stored in certified facilities"
       → TASK_SUMMARY_PROMPT rule 2e: parent context is off-topic → cap score at 49
         → goes to supplementary / gaps, NOT to key_findings
```

---

## 5. What This Doesn't Change

- **Zero new LLM calls**: `surrounding_window` is already computed (line 509) and reused.
- **Backward compatibility**: All 5 `NestedChunk` fields default to empty/None. Old code that
  constructs `NestedChunk` without provenance fields continues to work.
- **`_vector_search_scoped()`**: not affected — it uses a registry-resolved collection key,
  not depth-based parent context.
- **`resolve_reference_enhanced()` signatures**: unchanged — provenance is attached in
  `execute_task()` after resolution, not inside resolution functions.
- **Depth-0 primary context entries**: never have provenance keys (depth check prevents it).

---

## 6. Key Files Changed

| File | Change |
|------|--------|
| `src/models/research.py` | 5 provenance fields on `NestedChunk` |
| `src/agents/tools.py` | `create_tiered_context_entry()` accepts + conditionally stores provenance |
| `src/agents/nodes.py` | `execute_task()` attaches provenance; `_format_ranked_findings()` renders it; `_rerank_task_chunks()` passes it to prompt |
| `src/prompts/research.py` | `CHUNK_RERANKER_PROMPT_{SYSTEM,HUMAN}` gains `parent_context`; `TASK_SUMMARY_PROMPT_SYSTEM` gains rule 2e |
| `tests/test_provenance.py` | 11 new unit tests |

---

## 7. Test Plan

All tests in `tests/test_provenance.py`:

| Test | What it verifies |
|------|-----------------|
| `test_nested_chunk_default_provenance_fields` | All 5 provenance fields default to empty/None |
| `test_nested_chunk_provenance_fields_set` | NestedChunk stores provenance fields correctly |
| `test_depth0_has_no_provenance` | Depth-0 tiered context entry has no provenance keys |
| `test_depth1_with_parent_document_stores_provenance` | Depth-1 entry with parent stores all 5 keys |
| `test_depth1_without_parent_document_no_provenance` | Depth-1 entry without parent_document omits keys |
| `test_surrounding_context_truncated_to_500` | context is capped at 500 chars in entry dict |
| `test_format_ranked_findings_shows_via_ref_header` | `[via` appears for depth>0 chunks |
| `test_format_ranked_findings_shows_parent_context` | `Parent context:` line appears |
| `test_format_ranked_findings_no_via_for_direct_chunks` | No `[via` for direct search chunks |
| `test_reranker_human_prompt_with_parent_context` | `CHUNK_RERANKER_PROMPT_HUMAN` accepts `parent_context` |
| `test_reranker_human_prompt_with_na_parent_context` | Works with "N/A (direct...)" sentinel |

---

## 8. Mindmap Summary

```
RABBITHOLE PROVENANCE
├── PROBLEM
│   ├── Nested chunk B evaluated without knowing where in parent A its reference appeared
│   ├── Off-topic sentence: "For waste disposal see §21" → followed → scored high
│   └── Result: waste-storage law included in dose-limit report
│
├── ROOT CAUSE
│   └── surrounding_window computed at gate (nodes.py:509) but never attached to nc
│
├── SOLUTION (4 layers, zero new LLM calls)
│   ├── Layer 1: NestedChunk model — 5 provenance fields (optional, backward-compat)
│   ├── Layer 2: create_tiered_context_entry() — stores provenance in dict (depth>0 only)
│   ├── Layer 3: execute_task() — attaches surrounding_window to each nc after resolution
│   └── Layer 4: LLM scoring + formatting
│       ├── _rerank_task_chunks(): parent_context → CHUNK_RERANKER prompt → score penalty
│       ├── _format_ranked_findings(): [via ref ...] + Parent context: headers
│       └── TASK_SUMMARY_PROMPT rule 2e: cap off-topic ref chunks at score 49
│
├── UNCHANGED
│   ├── resolve_reference_enhanced() signature
│   ├── _vector_search_scoped() (registry-driven, no depth context)
│   ├── All depth-0 primary context entries (no provenance keys)
│   └── LLM call count (reuses existing surrounding_window)
│
└── TESTS
    ├── 11 new unit tests in tests/test_provenance.py
    └── 190 existing tests must still pass (no regressions)
```
