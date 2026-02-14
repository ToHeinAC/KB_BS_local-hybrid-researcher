"""Annotated copy of src/prompts.py — comprehensive workflow documentation.

Each prompt constant has a docstring explaining:
1. Phase & graph node where it is used
2. Called by: file + function
3. Workflow position: Previous prompt → THIS → Next prompt
4. Input/Output description & how the output is consumed

All prompts MUST be defined in src/prompts.py per project convention.
Never inline prompt strings in node functions or services.
Use template variables for dynamic content.

Every prompt follows a strict 4-section format optimised for small
local LLMs (<=20B parameters):

### Task   – one-sentence imperative
### Input  – enumerated variables
### Rules  – numbered constraints
### Output format – exact JSON / text template
"""

# =============================================================================
# HITL Prompts - Language Detection
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE_DETECTION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_init
# Called by: src/services/hitl_service.py :: _detect_language_llm() (line ~355)
# Workflow: [entry_router] → hitl_init (THIS) → hitl_generate_queries
# Previous: — (first prompt in the workflow)
# Next: ALTERNATIVE_QUERIES_INITIAL_PROMPT (in hitl_generate_queries)
#
# Input: {user_query} — the raw user query string
# Output: Plain text, two-letter language code ("de" or "en")
# Consumed by: Stored as state["detected_language"]; propagated to every
# subsequent prompt via {language} template variable.
#
# Notes: This is one of two prompts WITHOUT {language} enforcement
# (the other is REFERENCE_EXTRACTION_PROMPT), because its
# output is a code, not natural language.
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_DETECTION_PROMPT = """### Task
Detect the language of the user text.

### Input
- user_text: "{user_query}"

### Rules
1. Reply with ONLY a two-letter language code.
2. Supported codes: "de" (German), "en" (English).
3. If uncertain, default to "de".
4. Do NOT output anything else.

### Output format
de"""

# =============================================================================
# HITL Prompts - Follow-up Questions (merged DE + EN)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW_UP_QUESTIONS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_generate_questions
# Called by: src/services/hitl_service.py :: _generate_follow_up_questions_llm() (line ~399)
# Workflow: hitl_analyze_retrieval → hitl_generate_questions (THIS) → END (wait for user)
# Previous: RETRIEVAL_ANALYSIS_PROMPT (in hitl_analyze_retrieval)
# Next: — (graph pauses for user input; resumes at hitl_process_response)
#
# Input: {user_query} — original query
#        {context} — hitl_conversation_history (accumulated Q&A)
#        {retrieval} — query_retrieval text from vector DB
#        {language} — "German" or "English"
# Output: Plain text — 3 numbered follow-up questions
# Consumed by: Displayed to user in Streamlit UI; user's answers drive the
#              next HITL iteration via hitl_process_response.
#
# Notes: Merged from separate DE/EN prompts in Week 4.5.
#        Uses {retrieval} from state to avoid asking about info
#        already present in the knowledge base.
# ─────────────────────────────────────────────────────────────────────────────
FOLLOW_UP_QUESTIONS_PROMPT = """### Task
Generate exactly 3 follow-up questions to clarify the user's research query.

### Input
- user_query: "{user_query}"
- conversation_context: {context}
- knowledge_base_retrieval: {retrieval}

### Rules
1. Write all 3 questions in {language}.
2. Question 1 must clarify terminology or definitions.
3. Question 2 must identify missing or unclear details.
4. Question 3 must narrow the methodological scope or focus.
5. Use the knowledge_base_retrieval to avoid asking about information already available.
6. Output ONLY the 3 numbered questions, no explanations.

### Output format
1. [Question about definition/context]
2. [Question about details]
3. [Question about scope]"""

# =============================================================================
# HITL Prompts - User Feedback Analysis
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# USER_FEEDBACK_ANALYSIS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_finalize
# Called by: src/services/hitl_service.py :: _analyse_user_feedback_llm() (line ~432)
# Workflow: hitl_process_response → hitl_finalize (THIS) → generate_todo
# Previous: REFINED_QUERIES_PROMPT (in hitl_process_response, on loop iterations)
#           or FOLLOW_UP_QUESTIONS_PROMPT (if user typed /end after first iteration)
# Next: KNOWLEDGE_BASE_QUESTIONS_PROMPT (also in hitl_finalize)
#
# Input: {user_query} — original query
#        {context} — full hitl_conversation_history
#        {language} — "German" or "English"
# Output: JSON with entities[], scope, context, refined_query
# Consumed by: Fed into _generate_knowledge_base_questions_llm() as
#              {analysis}; also stored in state["query_analysis"].
#
# Notes: Extracts structured parameters from the free-form HITL
#        conversation to drive research planning.
# ─────────────────────────────────────────────────────────────────────────────
USER_FEEDBACK_ANALYSIS_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for human feedback analysis.

### Task
Analyse the conversation and extract key research directions, parameters and exact terminologiesclarifying the user's query.

### Input
- original_query: "{user_query}"
- conversation_history: {context}
- language: {language}

### Rules
1. Extract named entities, regulations, and technical terms. Preserve exact and precise terminology.
2. Determine the topical scope of the query.
3. Capture any additional context the user provided.
4. Formulate a refined search query incorporating all clarifications.
5. Write all JSON values in {language}.
6. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"entities": ["list of relevant entities/regulations"],
  "scope": "topic area of the query",
  "context": "additional context from conversation",
  "refined_query": "refined search query"}}
```"""

# =============================================================================
# HITL Prompts - Knowledge Base Questions Generation
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE_BASE_QUESTIONS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 → Phase 2 transition (HITL finalization)
# Graph node: hitl_finalize
# Called by: src/services/hitl_service.py :: _generate_knowledge_base_questions_llm() (line ~473)
# Workflow: USER_FEEDBACK_ANALYSIS_PROMPT → THIS → generate_todo
# Previous: USER_FEEDBACK_ANALYSIS_PROMPT (same node, sequential call)
# Next: TODO_GENERATION_PROMPT (in generate_todo node)
#
# Input: {user_query} — original query
#        {context} — hitl_conversation_history
#        {analysis} — JSON output from USER_FEEDBACK_ANALYSIS_PROMPT
#        {max_queries} — number of queries to generate
#        {language} — "German" or "English"
# Output: JSON with research_queries[] and summary
# Consumed by: research_queries stored in state["research_queries"]; used by
#              generate_todo to create the task list. Summary stored in
#              state["additional_context"].
#
# Notes: This is the final HITL prompt. Its output bridges Phase 1
#        (query refinement) to Phase 2 (research planning).
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_QUESTIONS_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for knowledge base questions generation.

### Task
Generate {max_queries} optimised search queries for a knowledge base based on the input below.

### Input
- original_query: "{user_query}"
- conversation_history: {context}
- extracted_analysis: {analysis}
- language: {language}

### Rules
1. Each query must target a different aspect of the original query.
2. Use domain-specific terminology from the extracted_analysis.
3. Queries must be specific enough for vector similarity search.
4. Write all JSON values (queries, summary) in {language}.
5. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"research_queries": ["query_1", "query_2", "..."],
  "summary": "brief summary of the research direction"}}
```"""

# =============================================================================
# HITL Prompts - Alternative Queries Generation
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE_QUERIES_INITIAL_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_generate_queries
# Called by: src/services/hitl_service.py :: generate_alternative_queries_llm() (line ~789)
#            (iteration == 0 branch)
# Workflow: hitl_init → hitl_generate_queries (THIS) → hitl_retrieve_chunks
# Previous: LANGUAGE_DETECTION_PROMPT (in hitl_init, first iteration only)
# Next: — (no LLM prompt in hitl_retrieve_chunks; next LLM prompt is
#        RETRIEVAL_ANALYSIS_PROMPT in hitl_analyze_retrieval)
#
# Input: {query} — original user query
#        {language} — "German" or "English"
# Output: JSON with broader_scope and alternative_angle query strings
# Consumed by: Combined with original query to form 3-query triple
#              [original, broader_scope, alternative_angle] stored in
#              state["iteration_queries"]. Used by hitl_retrieve_chunks
#              for vector DB search.
#
# Notes: Only used on iteration 0. Subsequent iterations use
#        ALTERNATIVE_QUERIES_REFINED_PROMPT instead.
# ─────────────────────────────────────────────────────────────────────────────
ALTERNATIVE_QUERIES_INITIAL_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for alternative queries generation.

### Task
Generate 2 alternative search queries for the given research question.

### Input
- original_query: "{query}"
- language: {language}

### Rules
1. broader_scope: explore related or contextual information.
2. alternative_angle: explore implications, challenges, or alternatives.
3. Both queries must stay anchored to the original query's intent.
4. Write all JSON values in {language}.
5. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"broader_scope": "...", "alternative_angle": "..."}}
```"""

# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE_QUERIES_REFINED_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_generate_queries
# Called by: src/services/hitl_service.py :: generate_alternative_queries_llm() (line ~795)
#            (iteration > 0 branch)
# Workflow: hitl_process_response → hitl_generate_queries (THIS) → hitl_retrieve_chunks
# Previous: REFINED_QUERIES_PROMPT (in hitl_process_response)
# Next: — (no LLM prompt in hitl_retrieve_chunks; next LLM prompt is
#        RETRIEVAL_ANALYSIS_PROMPT in hitl_analyze_retrieval)
#
# Input: {query} — original user query
#        {entities} — entities discovered so far
#        {gaps} — knowledge_gaps from previous retrieval analysis
#        {language} — "German" or "English"
# Output: JSON with broader_scope and alternative_angle query strings
# Consumed by: Same as ALTERNATIVE_QUERIES_INITIAL_PROMPT — forms a 3-query
#              triple for vector search in hitl_retrieve_chunks.
#
# Notes: Uses knowledge gaps and discovered entities to generate more
#        targeted queries in subsequent HITL iterations.
# ─────────────────────────────────────────────────────────────────────────────
ALTERNATIVE_QUERIES_REFINED_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for refined queries generation.

### Task
Generate 2 refined search queries based on research progress.

### Input
- original_query: "{query}"
- entities_found: {entities}
- knowledge_gaps: {gaps}
- language: {language}

### Rules
1. broader_scope: address the identified knowledge gaps.
2. alternative_angle: explore newly discovered concepts.
3. Incorporate entities where relevant.
4. Write all JSON values in {language}.
5. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"broader_scope": "...", "alternative_angle": "..."}}
```"""

# =============================================================================
# HITL Prompts - Retrieval Analysis
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL_ANALYSIS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_analyze_retrieval
# Called by: src/services/hitl_service.py :: analyze_retrieval_context_llm() (line ~839)
# Workflow: hitl_retrieve_chunks → hitl_analyze_retrieval (THIS) → hitl_generate_questions
# Previous: ALTERNATIVE_QUERIES_INITIAL_PROMPT or ALTERNATIVE_QUERIES_REFINED_PROMPT
#           (in hitl_generate_queries, same iteration)
# Next: FOLLOW_UP_QUESTIONS_PROMPT (in hitl_generate_questions)
#
# Input: {query} — original user query
#        {retrieval} — concatenated text from retrieved chunks
#        {language} — "German" or "English"
# Output: JSON with key_concepts[], entities[], scope, knowledge_gaps[],
#         coverage_score (0.0–1.0)
# Consumed by: coverage_score → state["coverage_score"] (convergence check)
#              knowledge_gaps → state["knowledge_gaps"] (convergence + refined queries)
#              Full result passed to FOLLOW_UP_QUESTIONS_PROMPT as context.
#
# Notes: Coverage score is key to convergence detection:
#        coverage >= 0.80 AND dedup >= 0.70 AND gaps <= 2 → finalize.
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVAL_ANALYSIS_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for retrieved context analysis.

### Task
Analyse the retrieved context against the user's research query.

### Input
- user_query: {query}
- retrieved_context: {retrieval}
- language: {language}

### Rules
1. Extract 5-7 core concepts from query and retrieved content.
2. List named entities (organisations, dates, technical terms).
3. State the primary focus area in one sentence.
4. List concrete knowledge gaps (not vague phrases like "more details").
5. Estimate coverage as a decimal 0.00-1.00 considering foundational, intermediate, and advanced coverage.
6. Write all JSON values in {language}.
7. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"key_concepts": ["..."],
  "entities": ["..."],
  "scope": "...",
  "knowledge_gaps": ["..."],
  "coverage_score": 0.00}}
```"""

# =============================================================================
# HITL Prompts - Refined Queries
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# REFINED_QUERIES_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_process_response
# Called by: src/services/hitl_service.py :: generate_refined_queries_llm() (line ~890)
# Workflow: END (user responds) → hitl_process_response (THIS) →
#           hitl_generate_queries (loop) OR hitl_finalize
# Previous: FOLLOW_UP_QUESTIONS_PROMPT (the questions the user just answered)
# Next: ALTERNATIVE_QUERIES_REFINED_PROMPT (if looping via hitl_generate_queries)
#        or USER_FEEDBACK_ANALYSIS_PROMPT (if finalizing via hitl_finalize)
#
# Input: {query} — original user query
#        {user_response} — user's text answer to the follow-up questions
#        {gaps} — current knowledge_gaps
#        {language} — "German" or "English"
# Output: JSON with query_1, query_2, query_3
# Consumed by: Updated in state["iteration_queries"] for the next retrieval
#              loop iteration. Used by hitl_generate_queries → hitl_retrieve_chunks.
#
# Notes: Generates refined queries based on user clarifications.
#        If user typed "/end", this prompt is skipped and hitl_finalize
#        is called directly.
# ─────────────────────────────────────────────────────────────────────────────
REFINED_QUERIES_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for refined queries generation.

### Task
Generate 3 refined search queries incorporating user feedback in order to clarify and narrow down the research direction.

### Input
- original_query: "{query}"
- user_clarification: "{user_response}"
- identified_gaps: {gaps}
- language: {language}

### Rules
1. query_1: address the identified knowledge gaps.
2. query_2: explore new concepts mentioned by the user.
3. query_3: reflect the updated scope after clarification.
4. Write all JSON values in {language}. Preserve exact and precise terminology.
5. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"query_1": "...", "query_2": "...", "query_3": "..."}}
```"""

# =============================================================================
# Research Prompts - ToDo Generation
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# TODO_GENERATION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 2 — Research Planning
# Graph node: generate_todo
# Called by: src/agents/nodes.py :: generate_todo() (line ~120)
# Workflow: hitl_finalize → generate_todo (THIS) → hitl_approve_todo
# Previous: KNOWLEDGE_BASE_QUESTIONS_PROMPT (in hitl_finalize)
# Next: — (no LLM prompt in hitl_approve_todo; next LLM prompt is
#        TASK_SEARCH_QUERIES_PROMPT in execute_task)
#
# Input: {original_query} — original user query
#        {key_concepts} — from query_analysis
#        {entities} — from query_analysis
#        {scope} — from query_analysis
#        {assumed_context} — additional context from HITL
#        {hitl_smry} — citation-aware HITL summary (from hitl_finalize)
#        {num_items} — number of tasks (3–5, max 15)
#        {language} — "German" or "English"
# Output: JSON with items[] — list of {id, task, context} objects
# Consumed by: Stored in state["todo_list"]. Task 0 (original query) is
#              prepended automatically by the node. Displayed to user for
#              HITL approval (add/remove/reorder). After approval,
#              execute_task iterates through these tasks.
#
# Notes: generate_todo() also prepends the original query as Task 0
#        for direct vector search, ensuring broad baseline coverage.
# ─────────────────────────────────────────────────────────────────────────────
TODO_GENERATION_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for to-do list generation.

### Task
Generate a list of {num_items} specific research tasks based on the input below.
The tasks is perfectly executed under the following key ideas:
DO: Given the original query, the key concepts and entities that are identified,
following the scope and considering the already found context, generate a list of {num_items} specific research tasks.
DON'T: You must not generate tasks or task keywords that are not directly related to the query concepts and entities
or that have been excluded, e.g. if a term is excluded by the user, you must not generate tasks related to that term.

### Input
- original_query: "{original_query}"
- key_concepts: {key_concepts}
- entities: {entities}
- scope: {scope}
- context: {assumed_context}
- hitl_findings: {hitl_smry}
- language: {language}

### Rules
1. Each task must be specific, measurable, and focused on finding concrete information. Preserve exact and precise terminology.
2. Each task must relate to the query concepts and entities.
3. Assign sequential integer IDs starting from 1.
4. Write all JSON values (task descriptions, context) in {language}.
5. Use hitl_findings to avoid duplicating already-covered information. Focus tasks on gaps and uncovered aspects.
6. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"items": [
    {{"id": 1, "task": "Find dose limit regulations", "context": "Core query requirement"}},
    {{"id": 2, "task": "...", "context": "..."}}
  ]}}
```"""

# =============================================================================
# Research Prompts - Information Extraction
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# INFO_EXTRACTION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction
# Graph node: execute_task
# Called by: src/agents/tools.py :: extract_info() (line ~98)
# Workflow: execute_task: TASK_SEARCH_QUERIES_PROMPT → vector search →
#           THIS (per chunk) → classify tier → reference detection
# Previous: TASK_SEARCH_QUERIES_PROMPT (in execute_task, same iteration)
# Next: REFERENCE_EXTRACTION_PROMPT (if hybrid ref detection enabled,
#        same task iteration) or TASK_SUMMARY_PROMPT (end of task)
#
# Input: {query} — current task's search query
#        {chunk_text} — raw text chunk from vector DB
#        {language} — "German" or "English"
# Output: Plain text — extracted relevant passages (no JSON wrapping)
# Consumed by: Stored in chunk dict as "extracted_info"; classified into
#              primary/secondary/tertiary context tiers. Fed into
#              TASK_SUMMARY_PROMPT at end of task.
#
# Notes: Simpler variant without quote preservation. Used as fallback
#        when verbatim quote extraction is not needed.
#        See INFO_EXTRACTION_WITH_QUOTES_PROMPT for enhanced version.
# ─────────────────────────────────────────────────────────────────────────────
INFO_EXTRACTION_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for information extraction.

### Task
Extract only the passages relevant to the search query from the text chunk.

### Input
- search_query: "{query}"
- text_chunk: {chunk_text}

### Rules
1. Include all information that answers or relates to the search query.
2. Be concise; omit filler and unrelated sentences.
3. Preserve exact and precise terminology.
4. Output the extracted text directly, no JSON wrapping.

### Output format
Write the extracted relevant passages directly in {language}. Example:
"Die Grenzwerte für die effektive Dosis betragen 20 mSv pro Kalenderjahr..."

Do NOT output any template or placeholder text. Output only the actual extracted passages."""

# ─────────────────────────────────────────────────────────────────────────────
# INFO_EXTRACTION_WITH_QUOTES_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction
# Graph node: execute_task
# Called by: src/agents/tools.py :: extract_info_with_quotes() (line ~133)
# Workflow: execute_task: TASK_SEARCH_QUERIES_PROMPT → vector search →
#           THIS (per chunk) → classify tier → reference detection
# Previous: TASK_SEARCH_QUERIES_PROMPT (in execute_task, same iteration)
# Next: REFERENCE_EXTRACTION_PROMPT (if hybrid ref detection enabled,
#        same task iteration) or TASK_SUMMARY_PROMPT (end of task)
#
# Input: {query} — current task's search query
#        {key_entities} — key entities from query_anchor
#        {chunk_text} — raw text chunk from vector DB
#        {language} — "German" or "English"
# Output: JSON with extracted_info (condensed text) and
#         preserved_quotes[] ({quote, relevance_reason})
# Consumed by: extracted_info → classified into context tiers
#              preserved_quotes → accumulated in state["preserved_quotes"]
#              for verbatim inclusion in final synthesis.
#
# Notes: Enhanced version of INFO_EXTRACTION_PROMPT. Preserves exact
#        legal definitions, technical specs, and regulation references.
#        Added in Week 4 (Graded Context Management).
# ─────────────────────────────────────────────────────────────────────────────
INFO_EXTRACTION_WITH_QUOTES_PROMPT = """
### Role
Within the deep research agentic workflow, you are a master for information extraction.

### Task
Extract relevant information and preserve critical verbatim quotes from the text chunk.

### Input
- search_query: "{query}"
- key_entities: {key_entities}
- text_chunk: {chunk_text}

### Rules
1. extracted_info: condensed relevant passages in {language}. Preserve exact and precise terminology.
2. preserved_quotes: list of exact verbatim quotes that must not be paraphrased.
3. Preserve quotes for: legal definitions with numbers/thresholds, technical specifications with units, named regulations with section numbers.
4. For each quote include the exact text and a brief relevance reason.
5. Return ONLY valid JSON, no extra text.

### Output format
Return ONLY a JSON object with this structure:
```json
{{"extracted_info": "<your condensed extraction here>",
  "preserved_quotes": [
    {{"quote": "<exact verbatim quote from the chunk>", "relevance_reason": "<brief reason>"}}
  ]}}
```
IMPORTANT: Replace all angle-bracket placeholders with actual content from the text chunk. Never output template text literally."""

# =============================================================================
# Research Prompts - Task Summary
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# TASK_SUMMARY_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (end of each task)
# Graph node: execute_task (internal helper)
# Called by: src/agents/nodes.py :: _generate_task_summary() (line ~1376)
# Workflow: [all chunks processed for task] → THIS → next task or
#           validate_relevance (if last task)
# Previous: INFO_EXTRACTION_PROMPT or INFO_EXTRACTION_WITH_QUOTES_PROMPT
#           (per-chunk, same task)
# Next: TASK_SEARCH_QUERIES_PROMPT (if more tasks remain)
#        or RELEVANCE_SCORING_PROMPT / validate_relevance (if last task)
#
# Input: {task} — current task description
#        {original_query} — original user query
#        {primary_findings} — Tier 1 high-confidence findings for this task
#        {secondary_findings} — Tier 2 supporting findings for this task
#        {tertiary_findings} — Tier 3 background context for this task
#        {preserved_quotes} — verbatim quotes from this task
#        {hitl_smry} — HITL findings summary (established context)
#        {language} — "German" or "English"
# Output: JSON with summary, key_findings[], gaps[],
#         relevance_assessment, irrelevant_findings[], relevance_score (0-100)
# Consumed by: Appended to state["task_summaries"]. Included in
#              SYNTHESIS_PROMPT_ENHANCED as {task_summaries}.
#
# Notes: Added in Week 4 (Graded Context). Includes drift detection
#        by asking LLM to identify irrelevant findings.
# ─────────────────────────────────────────────────────────────────────────────
TASK_SUMMARY_PROMPT = """ 
### Role
You are a research task synthesizer inside a deep-research agent.

### GOAL: Synthesize findings for ONE completed research task and assess relevance to the original query.

### Input
- task: "{task}"
- original_query: "{original_query}"
- hitl_findings: {hitl_smry}
- primary_findings (Tier 1 — highest confidence): {primary_findings}
- secondary_findings (Tier 2 — supporting): {secondary_findings}
- tertiary_findings (Tier 3 — background): {tertiary_findings}
- preserved_quotes: {preserved_quotes}

### Rules:
STEP-BY-STEP INSTRUCTIONS
1. Read the task description.
2. Re-read the original_query — every output must serve answering it.
3. Use hitl_findings as established context to understand how this task connects to the original_query.
4. Process findings by tier priority: primary_findings first, then secondary_findings, then tertiary_findings.
   For each finding, decide: does it directly help answer the original_query for this task?
   - YES → include in summary with source citations and exact terminology. Keep relevant passages from the original text.
   - PARTIALLY → include only the directly relevant part with citation. Keep relevant passages from the original text.
   - NO (shares keywords but addresses a different topic) → move to irrelevant_findings.
5. Format each citation as [Filename.pdf, Page N] using the source and page from the finding.
6. If tiers conflict, primary > secondary > tertiary. Note conflicts in gaps.
7. Embed verbatim quotes from preserved_quotes directly inside key_findings using "quote" [Source.pdf, Page N] format. Do not list them separately.
8. Identify gaps: what information is still missing to fully answer the original_query for this task?
9. Write a comprehensive synthesis in {language} that makes use of all information and especially on NEW information from this task. Include literal references to documents, article numbers, section references (e.g., §3 Abs. 2, Anlage 4) exactly as they appear in the source. Reference hitl_findings only when essential context is needed.
10. Write a one-sentence relevance_assessment.
11. Score relevance_score (0-100): how well do the findings answer the original_query?
    - 80-100: findings directly and substantially answer the query
    - 50-79: findings are partially relevant or cover only a subset
    - 20-49: findings are tangentially related
    - 0-19: findings do not address the query at all

IMPORTANT
- Write in {language}.
- Do NOT invent information. If data is missing, say so in gaps.
- Do NOT add preamble, explanation, or markdown fences — output raw JSON only.

### Output format
OUTPUT — Return ONLY this JSON, no other text:
```json
{{"summary": "<your comprehensive context synthesis in {language}>",
  "key_findings": ["<finding with exact terminology and \"verbatim quote\" [Document.pdf, Page N] citation>"],
  "gaps": ["<what is still missing>"],
  "relevance_assessment": "<one sentence>",
  "irrelevant_findings": ["<finding that looks related but does not answer the query>"],
  "relevance_score": 75}}
```
"""

# =============================================================================
# Research Prompts - Synthesis
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 4 — Query-Anchored Synthesis (legacy/fallback mode)
# Graph node: synthesize
# Called by: src/agents/nodes.py :: synthesize() (line ~708, legacy branch)
# Workflow: validate_relevance → synthesize (THIS) → quality_check
# Previous: TASK_SUMMARY_PROMPT (last task) → validate_relevance (no LLM)
# Next: QUALITY_CHECK_PROMPT (in quality_check)
#
# Input: {original_query} — original user query
#        {findings} — flat concatenated research findings
#        {language} — "German" or "English"
# Output: JSON with summary and key_findings[]
# Consumed by: Stored in state["report"]. Passed to quality_check for
#              evaluation, then to attribute_sources for citation linking.
#
# Notes: Legacy prompt used when graded context fields are empty.
#        Superseded by SYNTHESIS_PROMPT_ENHANCED in the graded context
#        workflow (Week 4+). Kept for backward compatibility.
# ─────────────────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """
### Role
You are an expert report writer producing extensive, detailed deep reports from research findings.

### Task
Generate a thorough, structured deep report that answers the original query using ONLY the provided research findings.

### Input
- original_query: "{original_query}"
- research_findings: {findings}

### Rules
REPORT STRUCTURE — the summary field must be a markdown-formatted deep report:
1. Begin with a direct answer to the query (1-2 sentences).
2. Then provide detailed sections covering every relevant aspect found in the research findings.
3. Use markdown headings (####), bullet points, and numbered lists for structure.

CONTENT RULES
- Preserve original wording from source material when possible.
- Include exact levels, figures, numbers, statistics, thresholds, and limits as they appear in the sources.
- Reference specific sections, paragraphs, articles (e.g., "§ 80 StrlSchV", "Anlage 4 Teil B").
- Use direct quotes (in quotation marks) for key definitions, legal text, or critical formulations.
- Cite every claim as [Document.pdf] — never omit the source.
- State explicitly when information is insufficient or contradictory.
- Use ONLY information from the provided findings — no external knowledge.
- Write in {language} only — no mixing.
- Do NOT invent values, numbers, or citations.
- Do NOT add preamble, explanation, or markdown fences — output raw JSON only.

### Output format
Return ONLY this JSON, no other text:
```json
{{"summary": "<extensive structured deep report in {language} with markdown formatting and citations>",
  "key_findings": ["<one key finding with [Document.pdf] citation>"]}}
```"""

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS_PROMPT_ENHANCED
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 4 — Query-Anchored Synthesis (graded context mode)
# Graph node: synthesize
# Called by: src/agents/nodes.py :: synthesize() (line ~733, enhanced branch)
# Workflow: validate_relevance → synthesize (THIS) → quality_check
# Previous: TASK_SUMMARY_PROMPT (last task) → validate_relevance (no LLM)
# Next: QUALITY_CHECK_PROMPT (in quality_check)
#
# Input: {original_query} — original user query
#        {hitl_smry} — citation-aware HITL summary from HITL_SUMMARY_PROMPT
#        {task_summaries} — per-task structured summaries (pre-digested tiered evidence)
#        {language} — "German" or "English"
# Output: JSON with summary, key_findings[], query_coverage (0–100),
#         remaining_gaps[]
# Consumed by: Stored in state["report"]. Passed to quality_check, then
#              to attribute_sources for clickable citation links.
#
# Notes: Primary synthesis prompt since Week 4. Works from pre-digested
#        task summaries only (tiered evidence resolved at task level).
#        Uses generate_structured_with_language() for strict
#        language enforcement with automatic retry.
# ─────────────────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT_ENHANCED = """
### Role
You are an expert report writer producing extensive, detailed deep reports from pre-digested task summaries.

### Task
Generate a thorough, structured deep report that answers the original query using ONLY the provided task summaries and HITL context.

### Input
- original_query: "{original_query}"
- hitl_smry: {hitl_smry}
- task_summaries: {task_summaries}

### Rules
REPORT STRUCTURE — the summary field must be a markdown-formatted deep report:
1. Begin with a direct answer to the query (1-2 sentences).
2. Then provide detailed sections covering every relevant aspect found across all task summaries.
3. Use markdown headings (####), bullet points, and numbered lists for structure.
4. Group related findings thematically — do not just list task summaries sequentially.
5. End with a brief assessment of completeness and any open questions.

CONTENT RULES
- Read hitl_smry for established context and user clarifications — build on it.
- Preserve original wording from source material when possible.
- Include exact levels, figures, numbers, statistics, thresholds, and limits as they appear in the sources.
- Reference specific sections, paragraphs, articles (e.g., "§ 80 StrlSchV", "Anlage 4 Teil B").
- Use direct quotes (in quotation marks) for key definitions, legal text, or critical formulations.
- Cite every claim as [Document.pdf, Page N] — never omit the source.
- Include verbatim quotes from task summaries where they support a finding.
- State explicitly when information is insufficient, contradictory, or missing.
- Use ONLY information from the provided task summaries — no external knowledge.
- Write in {language} only — no mixing.
- Do NOT invent values, numbers, or citations.
- Do NOT add preamble, explanation, or markdown fences — output raw JSON only.

COVERAGE AND GAPS
- Estimate query_coverage (0-100): how completely the original_query is answered.
- Collect remaining_gaps from all task summaries — what is still missing or contradictory.

### Output format
Return ONLY this JSON, no other text:
```json
{{"summary": "<extensive structured deep report in {language} with markdown formatting and [Document.pdf, Page N] citations>",
  "key_findings": ["<one key finding with [Document.pdf, Page N] citation>"],
  "query_coverage": 0,
  "remaining_gaps": ["<one gap or uncertainty>"]}}
```"""

# =============================================================================
# Reference Extraction Prompt (for LLM-based reference detection)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE_EXTRACTION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (reference following)
# Graph node: execute_task (reference detection sub-step)
# Called by: src/agents/tools.py :: extract_references_llm() (line ~626)
# Workflow: execute_task: info extraction → THIS (per chunk) →
#           resolve_reference_enhanced → scoped passage retrieval →
#           convergence check → next chunk or task summary
# Previous: INFO_EXTRACTION_PROMPT or INFO_EXTRACTION_WITH_QUOTES_PROMPT
#           (same chunk, same task)
# Next: TASK_SUMMARY_PROMPT (after all references for this task resolved)
#
# Input: {text} — raw text chunk to scan for references
# Output: JSON with references[] — list of {reference_mention,
#         reference_type, target_document_hint, confidence}
# Consumed by: Each reference is resolved via resolve_reference_enhanced()
#              which uses document_registry.json for scoped vector search.
#              Resolved passages are classified into context tiers.
#
# Notes: This is one of two prompts WITHOUT {language} enforcement
#        (the other is LANGUAGE_DETECTION_PROMPT), because it copies
#        reference mentions verbatim regardless of language.
#        Used in "llm" and "hybrid" extraction modes (not "regex").
#        In "hybrid" mode, results are deduplicated against regex
#        results by type:target key + substring overlap.
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_EXTRACTION_PROMPT = """### Task
Extract all references from the given text and classify each by type.

### Input
- text: {text}

### Rules
1. Classify each reference as one of: legal_section, academic_numbered, academic_shortform, document_mention.
2. legal_section: paragraph/section references (e.g. "§ 133 des Strahlenschutzgesetzes", "Section 5.2").
3. academic_numbered: numbered citations (e.g. "[253]", "[12, 15]").
4. academic_shortform: author-year citations (e.g. "[Townsend79]", "[Mueller2020]").
5. document_mention: named document references (e.g. "Kreislaufwirtschaftsgesetz", "KTA 1401", "ICRP Publication 103").
6. Copy the reference mention verbatim.
7. Provide a best guess for the target document name (empty string if unknown).
8. Set confidence between 0.0 and 1.0.
9. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"references": [
    {{"reference_mention": "exact text", "reference_type": "legal_section", "target_document_hint": "Strahlenschutzgesetz", "confidence": 0.95}}
  ]}}
```"""

# =============================================================================
# Agentic Decision Prompts
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE_DECISION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (reference following gate)
# Graph node: execute_task (agentic gate before resolve_reference_enhanced)
# Called by: src/agents/nodes.py :: execute_task() (reference loop)
# Workflow: detect references → THIS (per ref) → resolve_reference_enhanced
# Previous: INFO_EXTRACTION_PROMPT / INFO_EXTRACTION_WITH_QUOTES_PROMPT
# Next: resolve_reference_enhanced (if follow=true) or skip
#
# Input: {reference_type} — type of detected reference
#        {reference_target} — target text of the reference
#        {document_context} — document the reference was found in
#        {query_anchor} — original query + key entities + scope + current task
#        {language} — "German" or "English"
# Output: JSON with follow (bool) and reason (str)
# Consumed by: execute_task decides whether to call resolve_reference_enhanced
#
# Notes: First agentic decision point. LLM autonomously decides whether
#        following a reference is worth the token budget. Prevents tangential
#        references from diluting context.
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_DECISION_PROMPT = """
### Role
You are a senior decision expert with deep knowledge of methodological best practices to find relevant references.

### Task
Decide whether following this reference is worthwhile for answering the research query.
Best workflow is:
1. Analyse the query_anchor given to you carefully and methodically with respect to scope and current_task based on the original_query and key_entities.
2. Take into account the reference_type, reference_target, and the context in which it was found.
3. With the analysis in 1. and considering 2., decide whether following this reference in source_document is worthwhile.

### Input
- query_anchor: {query_anchor}  (contains: original_query, key_entities, scope, current_task)
- reference_type: "{reference_type}"
- reference_target: "{reference_target}"
- source_document: "{document_context}"
- surrounding_context: "{surrounding_context}"

### Rules
1. Follow if the reference likely contains information directly relevant to the query based on the surrounding_context.
2. Follow if the reference defines a key term, threshold, or regulation mentioned in the query.
3. Skip if the reference is tangential (e.g. general administrative procedures when the query is very specific details).
4. Skip if the surrounding_context indicates the reference is for background reading only.
5. Skip if the reference target is too vague to resolve (e.g. "see above").
6. When uncertain, FOLLOW — skipping a relevant reference is costlier than following a tangential one.
7. Write the reason in {language}.
8. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"follow": true, "reason": "brief explanation"}}
```"""

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY_REMEDIATION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 4 — Quality Assurance (remediation gate)
# Graph node: quality_check (agentic gate after scoring)
# Called by: src/agents/nodes.py :: quality_check() (after scoring)
# Workflow: quality_check scores → THIS → route_after_quality →
#           synthesize (retry) or attribute_sources (accept)
# Previous: QUALITY_CHECK_PROMPT (same node, sequential)
# Next: SYNTHESIS_PROMPT_ENHANCED (if retry) or attribute_sources (if accept)
#
# Input: {quality_scores} — the 5 dimension scores
#        {issues_found} — issues from quality check
#        {original_query} — original user query
#        {language} — "German" or "English"
# Output: JSON with action ("accept" or "retry") and focus_instructions
# Consumed by: quality_check sets phase to "retry_synthesis" if retry,
#              synthesize() appends focus_instructions to prompt on retry.
#
# Notes: Second agentic decision point. LLM evaluates its own output
#        quality and decides whether to re-synthesize with focused guidance.
#        Max 1 retry to prevent infinite loops.
# ─────────────────────────────────────────────────────────────────────────────
QUALITY_REMEDIATION_PROMPT = """### Task
Decide whether a low-quality research synthesis should be retried or accepted as-is.

### Input
- quality_scores: {quality_scores}
- issues_found: {issues_found}
- original_query: "{original_query}"

### Rules
1. Choose "retry" if specific dimensions scored below 50 and targeted improvement instructions can address them.
2. Choose "retry" if citation_correctness is low — this is fixable by re-emphasizing source attribution.
3. Choose "accept" if the overall score is borderline (within 10% of threshold) and issues are minor.
4. Choose "accept" if the issues are fundamental (e.g. insufficient source data) — retrying won't help.
5. If retrying, write specific focus_instructions addressing the weakest dimensions.
6. Write focus_instructions in {language}.
7. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"action": "retry", "focus_instructions": "specific guidance for re-synthesis"}}
```"""

# =============================================================================
# HITL Summary Prompt (citation-aware, for synthesis)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# HITL_SUMMARY_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 → Phase 4 bridge (created in hitl_finalize,
#        consumed in synthesize)
# Graph node: hitl_finalize (called internally by _generate_hitl_summary)
# Called by: src/agents/nodes.py :: _generate_hitl_summary() (line ~1466)
#            invoked from hitl_finalize() at line ~1300
# Workflow: hitl_finalize: USER_FEEDBACK_ANALYSIS_PROMPT →
#           KNOWLEDGE_BASE_QUESTIONS_PROMPT → THIS → generate_todo
# Previous: KNOWLEDGE_BASE_QUESTIONS_PROMPT (same node, sequential)
# Next: TODO_GENERATION_PROMPT (in generate_todo node)
#
# Input: {query} — original user query
#        {conversation} — full hitl_conversation_history
#        {retrieval} — accumulated query_retrieval text (with [doc, p.N] prefixes)
#        {gaps} — remaining knowledge_gaps
#        {language} — "German" or "English"
# Output: Plain text — citation-aware summary with [Source_filename] annotations
# Consumed by: Stored in state["hitl_smry"]. Included in
#              SYNTHESIS_PROMPT_ENHANCED as {hitl_smry} during
#              Phase 4 synthesis. Bridges Phase 1 insights to final answer.
#
# Notes: Added in Week 4, upgraded to citation-aware in Week 5.
#        Preserves source attribution via [Source_filename] annotations
#        so downstream synthesis can trace facts back to documents.
# ─────────────────────────────────────────────────────────────────────────────
HITL_SUMMARY_PROMPT = """
### Task
Summarise the research clarification conversation for use in final synthesis.
Produce a citation-aware summary that preserves source attribution.

### Input
- original_query: "{query}"
- conversation: {conversation}
- retrieved_context: {retrieval}
- knowledge_gaps: {gaps}

### Rules
1. Write the summary in {language}. Preserve exact and precise terminology.
2. After each factual statement, add a `[Source_filename]` citation matching the document name from retrieved_context.
3. Preserve exact numerical values, ranges, and percentages verbatim — never round or paraphrase numbers.
4. Use direct quotes `"..."` for key definitions, legal formulations, and technical terms.
5. Preserve section/paragraph references (e.g., §3 Abs. 2, Anlage 4 Teil B) exactly as they appear in the source.
6. Structure the output into two sections:
   - **PRIMARY**: Findings directly relevant to the original query.
   - **SECONDARY**: Tangential or supporting findings that provide useful background.
7. Cover: user's refined intent, key clarifications, most relevant retrieval findings, remaining gaps.
8. No prefix, suffix, or meta-commentary. Output the summary directly as plain text, no JSON.

### Output format
PRIMARY:
Factual statement with exact values [source_document.pdf]. Key definition: "verbatim quote" [source_document.pdf]. Reference to §3 Abs. 2 specifies threshold of 6 mSv/a [another_doc.pdf].

SECONDARY:
Supporting context with citation [background_doc.pdf]. Additional background detail [other_doc.pdf].

GAPS:
- Remaining gap 1
- Remaining gap 2"""

# =============================================================================
# Research Prompts - Relevance Scoring (Pre-Synthesis Validation)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE_SCORING_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3.5 — Pre-Synthesis Relevance Validation
# Graph node: validate_relevance (intended, not currently active)
# Called by: — IMPORTED but NOT USED in src/agents/nodes.py (line 44)
# Workflow: execute_task (last task) → validate_relevance → synthesize
# Previous: TASK_SUMMARY_PROMPT (last task)
# Next: SYNTHESIS_PROMPT or SYNTHESIS_PROMPT_ENHANCED (in synthesize)
#
# Input: {query} — original query (or query_anchor)
#        {entities} — key entities from query_anchor
#        {text} — context text to score
#        {language} — "German" or "English"
# Output: JSON with relevance_score (0–100) and reasoning
# Consumed by: Would be used by validate_relevance to LLM-score context
#              items before synthesis.
#
# Notes: Currently UNUSED. The validate_relevance node uses
#        _score_and_filter_context() which does simple keyword/entity
#        matching instead of LLM-based scoring, for efficiency.
#        This prompt is reserved for future "high-stakes" filtering
#        where LLM scoring may be worth the latency cost.
# ─────────────────────────────────────────────────────────────────────────────
RELEVANCE_SCORING_PROMPT = """
### Role
You are a relevance scoring assistant that rates the relevance of a given text to answering the query.

### Task
Score how relevant the given text is to answering the query.

### Input
- query: "{query}"
- key_entities: {entities}
- text: "{text}"
- language: {language}

### Rules
1. Score from 0 to 100.
2. 100 = directly answers the query, 75 = key supporting info, 50 = tangential, 25 = loosely connected, 0 = irrelevant.
3. Write the reasoning in {language}. Preserve exact and precise terminology.
4. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"relevance_score": 75, "reasoning": "brief explanation"}}
```"""

# =============================================================================
# Research Prompts - Task Search Queries
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# TASK_SEARCH_QUERIES_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (start of each task)
# Graph node: execute_task
# Called by: src/agents/nodes.py :: execute_task() (line ~271)
# Workflow: process_hitl_todo → execute_task (THIS) → vector search →
#           info extraction → reference detection → task summary
# Previous: TODO_GENERATION_PROMPT (in generate_todo, Phase 2)
#           — or TASK_SUMMARY_PROMPT (if continuing from previous task)
# Next: INFO_EXTRACTION_PROMPT or INFO_EXTRACTION_WITH_QUOTES_PROMPT
#        (per retrieved chunk, same task)
#
# Input: {task} — current task description
#        {original_query} — original user query
#        {hitl_context} — HITL context summary
#        {key_entities} — key entities from query_anchor
#        {language} — "German" or "English"
# Output: JSON with query_1 and query_2 (2 targeted search queries)
# Consumed by: Combined with a 3rd base query (task + original_query
#              concatenation) to form a 3-query set. Each query is sent
#              to vector DB; results deduplicated by doc:page:text key.
#
# Notes: Added in Week 4.5 (Multi-Query Task Execution). The 3rd
#        query is a simple concatenation, not LLM-generated.
#        Uses TaskSearchQueries Pydantic model for structured output.
# ─────────────────────────────────────────────────────────────────────────────
TASK_SEARCH_QUERIES_PROMPT = """

### Role
You are a search query generation assistant that generates 2 targeted vector-DB search queries for a research task.

### Task
Generate 2 targeted vector-DB search queries for a research task.
The tasks is perfectly executed under the following key ideas:
DO: Given the task and under the condition of the original query, acknowledging the hitl context and key entities,
generate 2 targeted vector-DB search queries for a research task.
DON'T: You must not generate tasks that are not closely covered by the research task or the original query.

### Input
- research_task: "{task}"
- original_query: "{original_query}"
- hitl_context: {hitl_context}
- key_entities: {key_entities}
- language: {language}

### Rules
1. query_1: focused query combining the task's core aspects with key entities.
2. query_2: complementary query exploring a related angle or alternative terminology.
3. Both queries must stay anchored to the original user query.
4. Use domain-specific terminology where possible.
5. Write all JSON values in {language}. Preserve exact and precise terminology.
6. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"query_1": "...", "query_2": "..."}}
```"""

# =============================================================================
# Research Prompts - Quality Check
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY_CHECK_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 4 — Quality Assurance
# Graph node: quality_check
# Called by: src/agents/nodes.py :: quality_check() (line ~863)
# Workflow: synthesize → quality_check (THIS) → attribute_sources → END
# Previous: SYNTHESIS_PROMPT or SYNTHESIS_PROMPT_ENHANCED (in synthesize)
# Next: — (no LLM prompt in attribute_sources; last prompt in workflow)
#
# Input: {original_query} — original user query
#        {summary} — the synthesized report from synthesize()
#        {language} — "German" or "English"
# Output: JSON with 5 dimension scores (0–100 each):
#         factual_accuracy, semantic_validity, structural_integrity,
#         citation_correctness, query_relevance, and issues_found[]
# Consumed by: Total score (0–500) stored in state["quality_score"].
#              Issues stored in state["quality_issues"]. Displayed in UI.
#              If total < QUALITY_THRESHOLD (375), a warning is shown.
#
# Notes: query_relevance dimension added in Week 4.5, expanding
#        scoring from 4 to 5 dimensions (max 400 → 500).
#        This is the LAST LLM prompt in the workflow. attribute_sources
#        performs rule-based citation linking without LLM calls.
# ─────────────────────────────────────────────────────────────────────────────
QUALITY_CHECK_PROMPT = """### Task
Evaluate the quality of a research summary against the original query.

### Input
- original_query: "{original_query}"
- summary: {summary}
- language: {language}

### Rules
1. Score each dimension from 0 to 100.
2. factual_accuracy: are claims factually correct?
3. semantic_validity: does it make logical sense?
4. structural_integrity: is it well-organised?
5. citation_correctness: are sources properly cited?
6. query_relevance: does the summary actually answer the original query? 0 if unrelated, 100 if fully answers it.
7. List any issues found. Write issues in {language}. Preserve exact and precise terminology.
8. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"factual_accuracy": 80,
  "semantic_validity": 85,
  "structural_integrity": 75,
  "citation_correctness": 70,
  "query_relevance": 90,
  "issues_found": ["issue 1"]}}
```"""
