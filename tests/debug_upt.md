# Debugging Analysis: Empty `hitl_smry` in Debug State Files

## Issue Description
`hitl_smry` is empty in `state_1hitl.md`, `state_2todo.md`, and `state_3rabbithole.md`,
despite the HITL conversation having completed with user interactions and retrieval data.

The previous analysis (test-order overwrite theory) was **incorrect** — this is a production-path bug, not a test artifact.

## Root Cause Analysis

### Root Cause 1: UI Chat-Based HITL Bypasses `hitl_finalize` Entirely

> [!CAUTION]
> This is the **primary root cause**. `hitl_smry` is _never generated_ when using the chat-based HITL UI.

**Flow (chat-based HITL — the default UI path):**
1. User interacts via `render_chat_hitl()` in `src/ui/app.py:435`
2. On completion, `_start_research_from_hitl(result)` is called (line 439)
3. This function (lines 529–604) builds `initial_state` with:
   - `initial_state["research_queries"] = research_queries` (line 583)
   - `initial_state["phase"] = "generate_todo"` (line 588)
   - `initial_state["hitl_active"] = False` (line 589)
4. The graph's `route_entry_point` sees `research_queries` populated → routes directly to `generate_todo` (line 53 of `graph.py`)
5. **`hitl_finalize` never runs** → `hitl_smry` stays `""` (its initial value from `create_initial_state`)

**Evidence:** `_start_research_from_hitl` never calls `_generate_hitl_summary()` and never sets `hitl_smry` in the initial state. The function only sets `additional_context` (line 584), which is a plain summary without citations.

### Root Cause 2: Termination Paths Drop `hitl_conversation_history`

> [!WARNING]
> Even when using graph-based HITL, termination paths produce a **stale** conversation history.

In `hitl_process_response` (nodes.py:1302–1400), three termination paths exist:

| Path | Line | Updates `hitl_state`? | Updates `hitl_conversation_history`? |
|------|------|-----------------------|--------------------------------------|
| `/end` | 1336 | ❌ No | ❌ No |
| `max_iterations` | 1351 | ✅ Yes (via `process_human_feedback`) | ❌ No |
| `convergence` | 1377 | ✅ Yes (via `process_human_feedback`) | ❌ No |
| `continue` (non-termination) | 1390 | ✅ Yes | ✅ Yes (line 1396) |

When `hitl_finalize` runs, it reads `hitl_conversation_history` from **agent state** (line 1475), not from `hitl_state["conversation_history"]`. The agent state version was last synced by `hitl_generate_questions` (line 1297), missing the final user response processed by `process_human_feedback`.

**Impact:** `_generate_hitl_summary` receives a conversation history that's missing the last round of user feedback.

## Data Flow Diagram

```
Chat-Based HITL (UI)                  Graph-Based HITL
────────────────────                  ──────────────────
render_chat_hitl()                    hitl_init
        │                                 │
        ▼                             hitl_generate_queries ← syncs hitl_conversation_history
_start_research_from_hitl()               │
   sets research_queries              hitl_retrieve_chunks
   sets additional_context                │
   ❌ NEVER sets hitl_smry             hitl_analyze_retrieval
        │                                 │
        ▼                             hitl_generate_questions ← syncs hitl_conversation_history
  route_entry_point                       │
   sees research_queries               END (wait for user)
   → skips to generate_todo               │
        │                             hitl_process_response
        ▼                                │
   generate_todo                      ┌──────┴──────────┐
   reads hitl_smry=""                 │ termination      │ continue
        │                             │ ❌ no sync       │ ✅ syncs history
        ▼                             ▼                  ▼
   state_1hitl.md                 hitl_finalize     hitl_generate_queries
   → hitl_smry is empty           reads stale              │
                                  conversation             ...
                                       │
                                  _generate_hitl_summary
                                  produces summary with
                                  stale/incomplete conv
```

## Implementation Plan

### Fix 1: Generate `hitl_smry` in `_start_research_from_hitl`

**File:** `src/ui/app.py` → `_start_research_from_hitl()`

After building `initial_state`, call `_generate_hitl_summary()` using:
- `query`: `user_query`
- `conversation`: `session.hitl_conversation_history`
- `retrieval`: `session.hitl_state.get("query_retrieval", "")`
- `knowledge_gaps`: `session.hitl_state.get("analysis", {}).get("knowledge_gaps", [])`
- `language`: `hitl_result.get("language", "de")`

Then set `initial_state["hitl_smry"] = hitl_smry`.

### Fix 2: Sync `hitl_conversation_history` on termination paths

**File:** `src/agents/nodes.py` → `hitl_process_response()`

Add `hitl_conversation_history` to all three termination return dicts:

```python
# /end path (line 1336) — also process the last feedback first
"hitl_conversation_history": hitl_state.get("conversation_history", []),

# max_iterations path (line 1351)
"hitl_conversation_history": hitl_state.get("conversation_history", []),

# convergence path (line 1377)
"hitl_conversation_history": hitl_state.get("conversation_history", []),
```

### Fix 3: Build `query_anchor` in `_start_research_from_hitl`

**File:** `src/ui/app.py` → `_start_research_from_hitl()`

The chat-based HITL path also never creates `query_anchor`, which is used downstream by `execute_task` and `classify_context_tier`. Add:
```python
initial_state["query_anchor"] = {
    "original_query": user_query,
    "detected_language": hitl_result.get("language", "de"),
    "key_entities": entities,
    "scope": scope,
    "hitl_refinements": [...],
    "created_at": datetime.now().isoformat(),
}
```

## Verification Plan

### Automated Tests
1. **Existing tests** in `tests/test_agents.py` — verify no regressions:
   ```
   cd /Users/tobiashein/dev/ai/langgraph/KB_BS_local-hybrid-researcher
   python -m pytest tests/test_agents.py -v
   ```
2. **New test**: `test_termination_paths_sync_conversation_history` — verify all 3 termination paths include `hitl_conversation_history` in their return dict
3. **New test**: `test_start_research_from_hitl_sets_hitl_smry` — verify the UI path populates `hitl_smry`

### Manual Verification
1. Enable state dump (`enable_state_dump=true`)
2. Start a research session via the Streamlit UI (chat-based HITL)
3. Complete the HITL conversation
4. After research completes, check:
   - `tests/debugging/state_1hitl.md` → `hitl_smry` should be non-empty
   - `tests/debugging/state_2todo.md` → `hitl_smry` should be non-empty
   - `tests/debugging/state_3rabbithole.md` → `hitl_smry` should be non-empty

---

# Primary Chunk Visibility Fix - Implementation Summary

## Issue Identified

Primary chunks (depth-0 initial retrieval results) were being suppressed in the GUI due to overly aggressive pre-synthesis filtering in `validate_relevance`. The `_score_and_filter_context` function applied strict thresholds without guaranteeing minimum results, unlike `filter_by_relevance` which had backfill logic.

## Root Cause Analysis

1. **Primary Cause**: `_score_and_filter_context` (lines 772-828 in nodes.py) had NO `min_results` parameter
   - Chunks with `final_relevance < 0.5` were completely filtered out
   - No backfill mechanism to preserve at least N chunks per task

2. **Secondary Cause**: Asymmetry between Stage 1 (classification) and Stage 2 (pre-synthesis filtering)
   - Stage 1's `filter_by_relevance` HAS backfill logic
   - Stage 2's `_score_and_filter_context` did NOT have backfill logic

## Implementation

### Files Modified

1. **src/config.py**
   - Added `primary_min_chunks: int = 3`
   - Added `secondary_min_chunks: int = 2`

2. **src/agents/nodes.py**
   - Added `min_results` parameter to `_score_and_filter_context()`
   - Implemented backfill logic mirroring `filter_by_relevance` pattern
   - Updated `validate_relevance()` to use `min_results=settings.primary_min_chunks` for primary context
   - Updated `validate_relevance()` to use `min_results=settings.secondary_min_chunks` for secondary context
   - Added logging for backfill events

3. **src/ui/components/task_rendering.py**
   - Updated `render_chunk_expander()` to display backfilled chunks with ⚠️ badge
   - Added info box showing backfill reason
   - Visual distinction for low-confidence chunks

4. **src/ui/components/results_view.py**
   - Added per-task backfill statistics display
   - Shows count of low-confidence chunks kept for transparency

5. **.env.example**
   - Documented new configuration parameters
   - Added Phase 3.5 section for chunk filtering

### New Test Coverage

**File**: `tests/test_chunk_filtering.py`

**11 comprehensive tests**:
- ✅ Basic filtering above threshold
- ✅ Filtering below threshold
- ✅ Backfill when below min_results
- ✅ Backfilled items sorted by score
- ✅ No backfill when enough results
- ✅ Backfill with zero passed
- ✅ min_results=0 disables backfill
- ✅ Empty input handling
- ✅ Backfill limited by available items
- ✅ Relevance score formula verification
- ✅ Integration placeholder

**All 11 tests passing** ✓
**All 65 existing tests passing** ✓ (no regressions)

## Solution Design

### Backfill Algorithm

```python
if len(scored_items) < min_results and len(all_scored) > 0:
    # Get rejected items
    rejected_items = [item for item in all_scored if item not in scored_items]
    rejected_items.sort(
        key=lambda x: x.get("final_relevance", x.get("context_weight", 0)),
        reverse=True
    )

    needed = min(min_results - len(scored_items), len(rejected_items))
    backfilled = rejected_items[:needed]

    # Mark as backfilled for UI transparency
    for item in backfilled:
        item["backfilled"] = True
        item["backfill_reason"] = f"Below threshold {threshold:.2f}, kept for visibility"

    scored_items.extend(backfilled)
```

### Guaranteed Minimums

- **Primary context**: 3 chunks minimum per task
- **Secondary context**: 2 chunks minimum per task
- **Tertiary context**: No minimum (light filtering only)

### UI Transparency

- Backfilled chunks show ⚠️ emoji in header
- Info box explains: "ℹ️ Below threshold X.XX (score: Y.YY), kept for visibility"
- Per-task statistics: "ℹ️ N Chunk(s) mit geringer Konfidenz (unter Relevanzschwelle, für Transparenz beibehalten)"

## Benefits

✅ **Transparency**: Users see what was actually retrieved, not just what survived filtering
✅ **Trust the Retrieval**: ChromaDB already scored these chunks as relevant
✅ **LLM Filtering Works**: Downstream prompts (`TASK_SUMMARY_PROMPT`) classify irrelevant findings
✅ **Prevents Silent Suppression**: Chunks don't disappear without user awareness
✅ **Visual Distinction**: Low-confidence chunks clearly marked
✅ **Configurable**: Power users can adjust thresholds via .env
✅ **No Regressions**: All existing tests still pass (65/65)

## Configuration

Add to `.env`:

```ini
# =============================================================================
# PHASE 3.5: CHUNK FILTERING
# =============================================================================
# Minimum chunks to keep per task (even if below relevance threshold)
PRIMARY_MIN_CHUNKS=3      # Keep at least 3 primary chunks per task
SECONDARY_MIN_CHUNKS=2    # Keep at least 2 secondary chunks per task
```

## Deployment Readiness

✅ Code complete
✅ Tests passing (11/11 new + 65/65 existing)
✅ Documentation updated
✅ Configuration parameters added
✅ UI enhancements implemented
✅ Backward compatible (backfilled flag is optional)

**Status**: ✅ **READY FOR DEPLOYMENT**
