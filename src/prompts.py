"""Centralized LLM prompts for the research agent.

All prompts MUST be defined in this file per project convention.
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

USER_FEEDBACK_ANALYSIS_PROMPT = """### Task
Analyse the conversation and extract key research parameters.

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

KNOWLEDGE_BASE_QUESTIONS_PROMPT = """### Task
Generate {max_queries} optimised search queries for a knowledge base.

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

ALTERNATIVE_QUERIES_INITIAL_PROMPT = """### Task
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

ALTERNATIVE_QUERIES_REFINED_PROMPT = """### Task
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

RETRIEVAL_ANALYSIS_PROMPT = """### Task
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

REFINED_QUERIES_PROMPT = """### Task
Generate 3 refined search queries incorporating user feedback.

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

TODO_GENERATION_PROMPT = """### Task
Generate a list of {num_items} specific research tasks for the given query analysis.

### Input
- original_query: "{original_query}"
- key_concepts: {key_concepts}
- entities: {entities}
- scope: {scope}
- context: {assumed_context}
- language: {language}

### Rules
1. Each task must be specific, measurable, and focused on finding concrete information. Preserve exact and precise terminology.
2. Each task must relate to the query concepts and entities.
3. Assign sequential integer IDs starting from 1.
4. Write all JSON values (task descriptions, context) in {language}.
5. Return ONLY valid JSON, no extra text.

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

INFO_EXTRACTION_PROMPT = """### Task
Extract only the passages relevant to the search query from the text chunk.

### Input
- search_query: "{query}"
- text_chunk: {chunk_text}

### Rules
1. Include only information that directly answers or relates to the search query.
2. Be concise; omit filler and unrelated sentences.
3. Write the extracted information in {language}. Preserve exact and precise terminology.
4. Output the extracted text directly, no JSON wrapping.

### Output format
<extracted relevant passages in {language}>"""

INFO_EXTRACTION_WITH_QUOTES_PROMPT = """### Task
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
```json
{{"extracted_info": "condensed relevant text in {language}",
  "preserved_quotes": [
    {{"quote": "exact verbatim text", "relevance_reason": "why this must be preserved"}}
  ]}}
```"""

# =============================================================================
# Research Prompts - Task Summary
# =============================================================================

TASK_SUMMARY_PROMPT = """### Task
Summarise the findings for a completed research task and assess their relevance.

### Input
- task: "{task}"
- original_query: "{original_query}"
- findings: {findings}
- preserved_quotes: {preserved_quotes}

### Rules
1. Write the summary in {language}.
2. Before summarising, discard any finding that is only superficially related to the original query (shares keywords but addresses a different topic).
3. Include key facts with source citations. Preserve exact and precise terminology.
4. Preserve any critical verbatim quotes.
5. Note gaps or limitations.
6. Provide a one-sentence relevance assessment.
7. List findings that seem related but do NOT actually answer the query.
8. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"summary": "concise task summary in {language}",
  "key_findings": ["finding 1", "finding 2"],
  "gaps": ["gap 1"],
  "relevance_assessment": "one-sentence verdict",
  "irrelevant_findings": ["finding that looks related but is not"]}}
```"""

# =============================================================================
# Research Prompts - Synthesis
# =============================================================================

SYNTHESIS_PROMPT = """### Task
Synthesise research findings into a coherent answer to the original query.

### Input
- original_query: "{original_query}"
- research_findings: {findings}

### Rules
1. Write the summary in {language}. Do not mix languages.
2. Include source citations in the format [Document_name.pdf, Page X].
3. Focus on directly answering the query. Preserve exact and precise terminology.
4. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"summary": "comprehensive answer in {language}",
  "key_findings": ["finding 1", "finding 2"]}}
```"""

SYNTHESIS_PROMPT_ENHANCED = """### Task
Synthesise tiered research findings into a comprehensive, query-anchored answer.

### Input
- original_query: "{original_query}"
- hitl_context_summary: {hitl_context_summary}
- primary_findings (highest confidence): {primary_findings}
- secondary_findings (supporting): {secondary_findings}
- tertiary_findings (background): {tertiary_findings}
- preserved_quotes: {preserved_quotes}
- task_summaries: {task_summaries}

### Rules
1. Write ONLY in {language}. Do not mix languages.
2. Before synthesising, discard any finding that shares keywords but addresses a different topic or intent than the original query.
3. If no relevant findings remain after filtering, state clearly that the knowledge base does not contain information to answer this query.
4. Begin with a direct answer to the original query.
5. Prioritise primary findings; use secondary for depth; use tertiary only to fill gaps.
6. Support claims with citations: [Document.pdf, Page X].
7. Include preserved quotes for legal/technical precision.
8. Acknowledge gaps identified during research.
9. Structure: Overview then Details then Limitations.
10. Return ONLY valid JSON, no extra text.

### Output format
```json
{{"summary": "comprehensive answer strictly in {language}",
  "key_findings": ["finding 1", "finding 2"],
  "query_coverage": 75,
  "remaining_gaps": ["unanswered aspect 1"]}}
```"""

# =============================================================================
# Reference Extraction Prompt (for LLM-based reference detection)
# =============================================================================

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
# HITL Context Summary Prompt (for synthesis)
# =============================================================================

HITL_CONTEXT_SUMMARY_PROMPT = """### Task
Summarise the research clarification conversation for use in final synthesis.

### Input
- original_query: "{query}"
- conversation: {conversation}
- retrieved_context: {retrieval}
- knowledge_gaps: {gaps}

### Rules
1. Write the summary in {language}. Preserve exact and precise terminology.
2. Cover: user's refined intent, key clarifications, most relevant retrieval findings, remaining gaps.
3. This summary will guide the final answer synthesis.
4. Output the summary as plain text, no JSON.

### Output format
<concise summary in {language}>"""

# =============================================================================
# Research Prompts - Relevance Scoring (Pre-Synthesis Validation)
# =============================================================================

RELEVANCE_SCORING_PROMPT = """### Task
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

TASK_SEARCH_QUERIES_PROMPT = """### Task
Generate 2 targeted vector-DB search queries for a research task.

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
