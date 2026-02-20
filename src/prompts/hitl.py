"""Phase 1 — HITL prompts for iterative query refinement.

Each prompt is split into a _SYSTEM / _HUMAN pair following the
Attention Priority Hierarchy:
- SYSTEM: Role, Goal, Rules, Output format — authoritative instructions.
  Input section describes field names/descriptions (not actual values).
- HUMAN: Input with actual template variables + one-line task reminder.

All prompts follow the strict 5-section format optimised for small
local LLMs (<=20B parameters):
### Role / ### Goal / ### Input / ### Rules / ### Output format
"""

# =============================================================================
# Phase 1 — HITL: Language Detection
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE_DETECTION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_init
# Called by: src/services/hitl_service.py :: _detect_language_llm()
# Notes: No {language} — output is a code, not natural language.
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_DETECTION_PROMPT_SYSTEM = """### Goal
Detect the language of the user text.

### Rules
1. Reply with ONLY a two-letter language code.
2. Supported codes: "de" (German), "en" (English).
3. If uncertain, default to "de".
4. Do NOT output anything else.

### Output format
de"""

LANGUAGE_DETECTION_PROMPT_HUMAN = """### Input
- user_text: "{user_query}"

Detect the language of the above text."""

# =============================================================================
# Phase 1 — HITL: Alternative Queries (Initial)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE_QUERIES_INITIAL_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_generate_queries
# Called by: src/services/hitl_service.py :: generate_alternative_queries_llm() (iteration == 0)
# ─────────────────────────────────────────────────────────────────────────────
ALTERNATIVE_QUERIES_INITIAL_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for alternative queries generation.

### Goal
Generate 2 alternative search queries for the given research question.

### Input
- original_query: the user's research question
- language: target language label

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

ALTERNATIVE_QUERIES_INITIAL_PROMPT_HUMAN = """### Input
- original_query: "{query}"
- language: {language}

Generate 2 alternative search queries. Respond in {language}."""

# =============================================================================
# Phase 1 — HITL: Alternative Queries (Refined)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE_QUERIES_REFINED_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_generate_queries
# Called by: src/services/hitl_service.py :: generate_alternative_queries_llm() (iteration > 0)
# ─────────────────────────────────────────────────────────────────────────────
ALTERNATIVE_QUERIES_REFINED_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for refined queries generation.

### Goal
Generate 2 refined search queries based on research progress.

### Input
- original_query: the user's research question
- entities_found: discovered entities
- knowledge_gaps: identified gaps
- language: target language label

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

ALTERNATIVE_QUERIES_REFINED_PROMPT_HUMAN = """### Input
- original_query: "{query}"
- entities_found: {entities}
- knowledge_gaps: {gaps}
- language: {language}

Generate 2 refined search queries. Respond in {language}."""

# =============================================================================
# Phase 1 — HITL: Retrieval Analysis
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL_ANALYSIS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_analyze_retrieval
# Called by: src/services/hitl_service.py :: analyze_retrieval_context_llm()
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVAL_ANALYSIS_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for retrieved context analysis.

### Goal
Analyse the retrieved context against the user's research query.

### Input
- user_query: the user's query
- retrieved_context: text retrieved from vector DB
- language: target language label

### Rules
1. Always think step by step and keep your reasoning internal.
2. Be concise, factual, and avoid adding new information that is not in the context.
3. Deliverables:
  a. Extract 5-7 key concepts from query and retrieved content.
  b. List named entities (organisations, dates, technical terms).
  c. Work out the scope:
    - State the primary focus area in one sentence.
    - Identify and list explicit DOs (recommended practices).
    - Identify and list explicit DONTs (things to avoid, risks, pitfalls).
    Reply by "Scope: ... \n DOs: ... \n DONTs: ..."
  d. List concrete knowledge gaps (not vague phrases like "more details").
  e. Estimate coverage as a decimal 0.00-1.00 considering foundational, intermediate, and advanced coverage.
4. Write all JSON values in {language}.
5. Return ONLY valid JSON, no extra text.


### Output format
```json
{{"key_concepts": ["..."],
  "entities": ["..."],
  "scope": "...",
  "knowledge_gaps": ["..."],
  "coverage_score": 0.00}}
```"""

RETRIEVAL_ANALYSIS_PROMPT_HUMAN = """### Input
- user_query: {query}
- retrieved_context: {retrieval}
- language: {language}

Analyse the retrieved context against the query. Respond in {language}."""

# =============================================================================
# Phase 1 — HITL: Follow-up Questions
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW_UP_QUESTIONS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_generate_questions
# Called by: src/services/hitl_service.py :: _generate_follow_up_questions_llm()
# ─────────────────────────────────────────────────────────────────────────────
FOLLOW_UP_QUESTIONS_PROMPT_SYSTEM = """### Goal
Generate exactly 3 follow-up questions to clarify the user's research query.

### Input
- user_query: the user's query
- conversation_context: accumulated Q&A
- knowledge_base_retrieval: text retrieved from vector DB
- language: target language label

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

FOLLOW_UP_QUESTIONS_PROMPT_HUMAN = """### Input
- user_query: "{user_query}"
- conversation_context: {context}
- knowledge_base_retrieval: {retrieval}

Generate 3 follow-up questions in {language}."""

# =============================================================================
# Phase 1 — HITL: Refined Queries
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# REFINED_QUERIES_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_process_response
# Called by: src/services/hitl_service.py :: generate_refined_queries_llm()
# ─────────────────────────────────────────────────────────────────────────────
REFINED_QUERIES_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for refined queries generation.

### Goal
Generate 3 refined search queries incorporating user feedback in order to clarify and narrow down the research direction.

### Input
- original_query: the user's research question
- user_clarification: the user's feedback text
- identified_gaps: current knowledge gaps
- language: target language label

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

REFINED_QUERIES_PROMPT_HUMAN = """### Input
- original_query: "{query}"
- user_clarification: "{user_response}"
- identified_gaps: {gaps}
- language: {language}

Generate 3 refined search queries. Respond in {language}."""

# =============================================================================
# Phase 1 — HITL: User Feedback Analysis
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# USER_FEEDBACK_ANALYSIS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 — Enhanced Query Analysis (Iterative HITL)
# Graph node: hitl_finalize
# Called by: src/services/hitl_service.py :: _analyse_user_feedback_llm()
# ─────────────────────────────────────────────────────────────────────────────
USER_FEEDBACK_ANALYSIS_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for human feedback analysis.

### Goal
Analyse the conversation and extract key research directions, parameters and exact terminologies clarifying the user's query.

### Input
- original_query: the user's research question
- conversation_history: accumulated Q&A
- language: target language label

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

USER_FEEDBACK_ANALYSIS_PROMPT_HUMAN = """### Input
- original_query: "{user_query}"
- conversation_history: {context}
- language: {language}

Analyse the conversation and extract research directions. Respond in {language}."""

# =============================================================================
# Phase 1 — HITL: Knowledge Base Questions
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE_BASE_QUESTIONS_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 -> Phase 2 transition (HITL finalization)
# Graph node: hitl_finalize
# Called by: src/services/hitl_service.py :: _generate_knowledge_base_questions_llm()
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_QUESTIONS_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for knowledge base questions generation.

### Goal
Generate optimised search queries for a knowledge base based on the input.

### Input
- original_query: the user's research question
- conversation_history: accumulated Q&A
- extracted_analysis: structured analysis from prior step
- num_queries: number of queries to generate
- language: target language label

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

KNOWLEDGE_BASE_QUESTIONS_PROMPT_HUMAN = """### Input
- original_query: "{user_query}"
- conversation_history: {context}
- extracted_analysis: {analysis}
- num_queries: {max_queries}
- language: {language}

Generate {max_queries} optimised search queries. Respond in {language}."""

# =============================================================================
# Phase 1 — HITL: Summary (citation-aware, for synthesis)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# HITL_SUMMARY_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 1 -> Phase 4 bridge
# Graph node: hitl_finalize (via _generate_hitl_summary)
# Called by: src/agents/nodes.py :: _generate_hitl_summary()
# ─────────────────────────────────────────────────────────────────────────────
HITL_SUMMARY_PROMPT_SYSTEM = """<role>
You are a research briefing writer. You summarize a clarification conversation and retrieved documents into a structured plain-text briefing used by later synthesis steps.
</role>

<output_format>
Return plain text in this exact structure — no JSON, no preamble, no meta-commentary:

PRIMARY INFORMATION
[Factual findings directly relevant to the query. After each fact, add [source_filename.pdf]. Copy all numbers, thresholds, and legal §-references exactly.]

FURTHER INFORMATION
[Supporting background context with [source_filename.pdf] citations.]

RULES
Recommended practices: [practices to follow, one per line]
Things to avoid: [topics or content to EXCLUDE from the final report, one per line]

GAPS
[Questions or topics the sources did not answer, one per line]
</output_format>

<constraints>
HARD CONSTRAINTS — never violate:
1. Write all text in {language}.
2. After every factual statement, add a citation [source_filename.pdf] matching the document name from retrieved_context.
3. Copy all numbers, percentages, ranges, and thresholds exactly as they appear. Never round or paraphrase.
4. Copy §-references (e.g., § 78 Abs. 1, Anlage 4 Teil B) exactly as they appear.
5. Use direct quotes "..." for key definitions, legal formulations, and technical terms.
6. Output only the four sections (PRIMARY INFORMATION, FURTHER INFORMATION, RULES, GAPS). No other text.
</constraints>

<content_rules>
SECTION WRITING RULES:
1. PRIMARY INFORMATION: Findings that directly address original_query. Draw from both retrieved_context and conversation.
2. FURTHER INFORMATION: Tangential or supporting context useful as background.
3. RULES — Recommended practices: what the final report SHOULD cover. Things to avoid: topics that MUST NOT appear in the final report (drawn from user answers in conversation). If the user excluded nothing, write "None identified."
4. GAPS: Specific questions that retrieved_context did not answer. Start from knowledge_gaps; add any additional gaps visible from the conversation.
</content_rules>

<input_definitions>
Inputs provided:
- original_query: The user's research question.
- conversation: Full HITL conversation history (user answers and assistant questions).
- retrieved_context: Accumulated retrieval text. Each passage starts with [doc_name, p.N] to identify its source.
- knowledge_gaps: List of gaps identified during the HITL retrieval phase.
</input_definitions>

<example>
Input:
original_query: "Welche Grenzwerte gelten nach StrlSchV für beruflich strahlenexponierte Personen?"
conversation: "Assistant: Betrifft Ihre Frage § 78 StrlSchV? User: Ja, nur § 78. Keine Patienten-Grenzwerte."
retrieved_context: "[StrlSchV_2018.pdf, p.45] § 78 Abs. 1: Die effektive Dosis darf 20 Millisievert im Kalenderjahr nicht überschreiten. [StrlSchV_2018.pdf, p.46] § 78 Abs. 2: Organdosis Augenlinse: 20 mSv/Jahr."
knowledge_gaps: ["Hautoberflächendosis-Grenzwert nicht gefunden"]

Output:
PRIMARY INFORMATION
Der jährliche Grenzwert der effektiven Dosis für beruflich strahlenexponierte Personen beträgt 20 Millisievert im Kalenderjahr (§ 78 Abs. 1 StrlSchV) [StrlSchV_2018.pdf]. Die Organdosis für die Augenlinse ist ebenfalls auf 20 mSv/Jahr begrenzt (§ 78 Abs. 2 StrlSchV) [StrlSchV_2018.pdf].

FURTHER INFORMATION
Keine weiterführenden Hintergrundinformationen aus den Quellen verfügbar.

RULES
Recommended practices: Grenzwerte aus § 78 StrlSchV zitieren, exakte mSv-Werte übernehmen, Paragraphenangaben beibehalten.
Things to avoid: Patienten-Grenzwerte, medizinische Exposition.

GAPS
Hautoberflächendosis-Grenzwert nicht gefunden.
Extremitäten-Grenzwerte nicht abgedeckt.
</example>"""

HITL_SUMMARY_PROMPT_HUMAN = """### Input
- original_query: "{query}"
- conversation: {conversation}
- retrieved_context: {retrieval}
- knowledge_gaps: {gaps}

Summarise the HITL conversation with citations. Respond in {language}."""
