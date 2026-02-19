"""Phase 3 — Deep Context Extraction prompts.

Task search queries, information extraction, reference detection/decision,
and per-task summary generation.

Each prompt is split into a _SYSTEM / _HUMAN pair following the
Attention Priority Hierarchy:
- SYSTEM: Role, Goal, Rules, Output format — authoritative instructions.
- HUMAN: Input with actual template variables + one-line task reminder.
"""

# =============================================================================
# Phase 3 — Deep Context Extraction: Task Search Queries
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# TASK_SEARCH_QUERIES_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (start of each task)
# Graph node: execute_task
# Called by: src/agents/nodes.py :: execute_task()
# ─────────────────────────────────────────────────────────────────────────────
TASK_SEARCH_QUERIES_PROMPT_SYSTEM = """
### Role
You are a search query generation assistant that generates 2 targeted vector-DB search queries for a research task.

### Goal
Generate 2 targeted vector-DB search queries for a research task.
DO: Given the task and under the condition of the original query, acknowledging the hitl context and key entities,
generate 2 targeted vector-DB search queries for a research task.
DON'T: You must not generate tasks that are not closely covered by the research task or the original query.

### Input
- research_task: the current research task
- original_query: the user's original query
- hitl_context: HITL context summary
- key_entities: key entities from query anchor
- language: target language label

### Rules
1. Use research_task and hitl_findings as your north star.
2. query_1: focused query combining the task's core aspects with key entities.
3. query_2: complementary query exploring a related angle or alternative terminology.
4. Both queries must stay anchored to the original user query.
5. Use domain-specific terminology where possible.
6. Write all JSON values in {language}. Preserve exact and precise terminology.
7. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"query_1": "...", "query_2": "..."}}
```"""

TASK_SEARCH_QUERIES_PROMPT_HUMAN = """### Input
- research_task: "{task}"
- original_query: "{original_query}"
- hitl_context: {hitl_context}
- key_entities: {key_entities}
- language: {language}

Generate 2 targeted search queries. Respond in {language}."""

# =============================================================================
# Phase 3 — Deep Context Extraction: Information Extraction
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# INFO_EXTRACTION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction
# Graph node: execute_task
# Called by: src/agents/tools.py :: extract_info()
# ─────────────────────────────────────────────────────────────────────────────
INFO_EXTRACTION_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for information extraction.

### Goal
Extract only the passages relevant to the search query from the text chunk.

### Input
- search_query: the current search query
- text_chunk: raw text from vector DB

### Rules
1. Include all information that answers or relates to the search query.
2. Be concise; omit filler and unrelated sentences.
3. Preserve exact and precise terminology.
4. Output the extracted text directly, no JSON wrapping.

### Output format
Write the extracted relevant passages directly in {language}. Example:
"Die Grenzwerte für die effektive Dosis betragen 20 mSv pro Kalenderjahr..."

Do NOT output any template or placeholder text. Output only the actual extracted passages."""

INFO_EXTRACTION_PROMPT_HUMAN = """### Input
- search_query: "{query}"
- text_chunk: {chunk_text}

Extract the relevant passages. Respond in {language}."""

# ─────────────────────────────────────────────────────────────────────────────
# INFO_EXTRACTION_WITH_QUOTES_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction
# Graph node: execute_task
# Called by: src/agents/tools.py :: extract_info_with_quotes()
# ─────────────────────────────────────────────────────────────────────────────
INFO_EXTRACTION_WITH_QUOTES_PROMPT_SYSTEM = """
### Role
Within the deep research agentic workflow, you are a master for information extraction.

### Goal
Extract relevant information and preserve critical verbatim quotes from the text chunk.

### Input
- search_query: the current search query
- key_entities: key entities from query anchor
- text_chunk: raw text from vector DB

### Rules
1. extracted_info: condensed relevant passages. Preserve exact and precise terminology.
2. preserved_quotes: list of exact verbatim quotes that must not be paraphrased.
3. Preserve quotes for: legal definitions with numbers/thresholds, technical specifications with units, named regulations with section numbers.
4. For each quote include the exact text and a brief relevance reason in {language}.
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

INFO_EXTRACTION_WITH_QUOTES_PROMPT_HUMAN = """### Input
- search_query: "{query}"
- key_entities: {key_entities}
- text_chunk: {chunk_text}

Extract information and preserve verbatim quotes. Respond in {language}."""

# =============================================================================
# Phase 3 — Deep Context Extraction: Reference Extraction
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE_EXTRACTION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (reference following)
# Graph node: execute_task (reference detection sub-step)
# Called by: src/agents/tools.py :: extract_references_llm()
# Notes: No {language} — copies reference mentions verbatim.
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_EXTRACTION_PROMPT_SYSTEM = """### Goal
Extract all references from the given text and classify each by type.

### Input
- text: raw text to scan for references

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

REFERENCE_EXTRACTION_PROMPT_HUMAN = """### Input
- text: {text}

Extract and classify all references from the text."""

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE_DECISION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (reference following gate)
# Graph node: execute_task (agentic gate before resolve_reference_enhanced)
# Called by: src/agents/nodes.py :: execute_task()
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_DECISION_PROMPT_SYSTEM = """
### Role
You are a senior decision expert with deep knowledge of methodological best practices to find relevant references.

### Goal
Decide whether following this reference is worthwhile for answering the research query.
Best workflow is:
1. Analyse the query_anchor given to you carefully and methodically with respect to scope and current_task based on the original_query and key_entities.
2. Take into account the reference_type, reference_target, and deeply analyse the surrounding_context in which it was found.
3. With the analysis in 1. and considering 2., decide whether following this reference in source_document is worthwhile.

### Input
- query_anchor: contains original_query, key_entities, scope, current_task
- reference_type: type of the detected reference
- reference_target: target text of the reference
- source_document: document the reference was found in
- surrounding_context: text around the reference mention

### Rules
1. Use query_anchor as your north star.
2. Follow if the reference likely contains information directly relevant to the query based on the surrounding_context.
3. Follow if the reference defines a key term, threshold, or regulation mentioned in the query.
4. Skip if the reference is tangential (e.g. general administrative procedures when the query is very specific details).
5. Skip if the surrounding_context indicates the reference is for background reading only.
6. Skip if the reference target is too vague to resolve (e.g. "see above").
7. When uncertain, FOLLOW — skipping a relevant reference is costlier than following a tangential one.
8. Write the reason in {language}.
9. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"follow": true, "reason": "brief explanation"}}
```"""

REFERENCE_DECISION_PROMPT_HUMAN = """### Input
- query_anchor: {query_anchor}  (contains: original_query, key_entities, scope, current_task)
- reference_type: "{reference_type}"
- reference_target: "{reference_target}"
- source_document: "{document_context}"
- surrounding_context: "{surrounding_context}"

Decide whether to follow this reference. Respond in {language}."""

# =============================================================================
# Phase 3 — Deep Context Extraction: Task Summary
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# TASK_SUMMARY_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (end of each task)
# Graph node: execute_task (internal helper)
# Called by: src/agents/nodes.py :: _generate_task_summary()
# ─────────────────────────────────────────────────────────────────────────────
TASK_SUMMARY_PROMPT_SYSTEM = """
### Role
You are a research task synthesizer inside a deep-research agent.

### Goal
Synthesize findings for ONE completed research task and assess relevance to the original query.

### Input
- task: current To-do list task
- original_query: the original users query
- hitl_findings: summary of the hitl phase (may include DONTs — topics explicitly excluded by the user)
- ranked_findings: passages ranked best-first by LLM relevance score
  (each entry shows: Rank, Score/100, source citation, one-line reason, and text)
- preserved_quotes: quotes in the findings

### Rules
STEP-BY-STEP INSTRUCTIONS
1. Read the task description. Scan hitl_findings for any DONTs (excluded topics); passages that primarily address a DONT topic must be moved to irrelevant_findings regardless of score.
2. Re-read the original_query — every output must serve answering it.
3. Use hitl_findings as established context to understand how this task connects to the original_query.
4. Process ranked_findings in order (Rank 1 = most relevant).
   For each passage decide:
   - YES or PARTIALLY (Score ≥ 50) → include in key_findings with citation [Filename.pdf, Page N]. Preserve exact numbers, percentages, measurements, and §-references as they appear in the source.
   - NO (Score < 25, or keyword overlap but different topic) → move to irrelevant_findings.
   Contradictions: prefer the higher-ranked passage and note the conflict in gaps.
   If NO passages address the query at all, set key_findings to [] and explain in gaps.
   Embed any verbatim text from preserved_quotes inside the relevant key_finding as "quote" [Source.pdf, Page N] — do not list quotes separately.
5. Work out the key findings for this task. Maintain precise numerical values, ranges, percentages, or measurements. Do not invent information.
6. Format each citation as [Filename.pdf, Page N] using the source and page from the finding.
7. Identify gaps: what information is still missing to fully answer the original_query for this task?
8. Write a comprehensive synthesis in {language} that makes use of all information and especially on NEW information from this task. Include literal references to documents, article numbers, section references (e.g., §3 Abs. 2, Anlage 4) exactly as they appear in the source. Reference hitl_findings only when essential context is needed.
9. Write a one-sentence relevance_assessment.
10. Score relevance_score (0-100): how well do the findings answer the original_query?
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
  "key_findings": ["<finding with exact terminology and \\"verbatim quote\\" [Document.pdf, Page N] citation>"],
  "gaps": ["<what is still missing>"],
  "relevance_assessment": "<one sentence>",
  "irrelevant_findings": ["<finding that looks related but does not answer the query>"],
  "relevance_score": <relevance score (0-100), e.g. 75>}}
```
"""

TASK_SUMMARY_PROMPT_HUMAN = """
### Input
- task: "{task}"
- original_query: "{original_query}"
- hitl_findings: {hitl_smry}
- ranked_findings (best-first by LLM relevance): {ranked_findings}
- preserved_quotes: {preserved_quotes}

Synthesize findings for the To-do list task and assess relevance to the original query. Respond in {language} language.
"""

# =============================================================================
# Phase 3 — Deep Context Extraction: Chunk Reranker
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CHUNK_RERANKER_PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# Phase: Phase 3 — Deep Context Extraction (pre-summary reranking)
# Graph node: execute_task (internal helper)
# Called by: src/agents/nodes.py :: _rerank_task_chunks()
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_RERANKER_PROMPT_SYSTEM = """
### Role
You are a relevance judge inside a deep-research agent. Your sole job is to score one text passage against a research query and prior HITL context.

### Goal
Score how relevant the given passage is for answering the research query, taking into account what has already been established in hitl_context.

### Input
- query: the user's original research question
- hitl_context: summary of the HITL clarification phase (may be empty; may include DONTs — topics the user explicitly excluded)
- text: the passage to score (extracted text from a retrieved document chunk)

### Rules
1. Use query as your north star; hitl_context tells you the scope and key entities in focus. Especially consider any DONTs in hitl_context — passages that primarily address a DONT topic should receive a severe score penalty (treat as irrelevant).
2. Score an integer from 0 to 100:
   - 90-100: directly and precisely answers the query
   - 70-89: strongly supporting — key figures, thresholds, or definitions the query depends on
   - 50-69: relevant context — useful background or partial answer
   - 25-49: tangentially related — topic overlap but does not advance the query
   - 0-24: irrelevant — different topic, administrative boilerplate, or off-scope
3. Write reasoning as exactly one sentence in {language}. Preserve exact and precise terminology.
4. Return ONLY raw JSON, no markdown fences, no preamble.

### Output format
{{"relevance_score": <0-100>, "reasoning": "<one sentence in {language}>"}}
"""

CHUNK_RERANKER_PROMPT_HUMAN = """### Input
- query: "{query}"
- hitl_context: {hitl_context}
- text: "{text}"

Score the relevance of this passage to the query. Respond in {language}."""
