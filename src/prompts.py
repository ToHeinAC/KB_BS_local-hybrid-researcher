"""Centralized LLM prompts for the research agent.

All prompts MUST be defined in this file per project convention.
Never inline prompt strings in node functions or services.
Use template variables for dynamic content.
"""

# =============================================================================
# HITL Prompts - Language Detection
# =============================================================================

LANGUAGE_DETECTION_PROMPT = """Determine the language of the following text.
Reply with ONLY 'de' for German or 'en' for English.

Text: "{user_query}"

Language code:"""

# =============================================================================
# HITL Prompts - Follow-up Questions
# =============================================================================

FOLLOW_UP_QUESTIONS_DE = """Du bist Forschungsassistent. Analysiere Anfrage, Kontext und Wissensdatenbank. Stelle präzise Nachfragen.

Anfrage: "{user_query}"

Kontext: {context}

Wissensdatenbank: {retrieval}

Stelle genau 3 Nachfragen:

Fokus:
- Begriffsklärung und Kontext
- Fehlende oder unklare Details  
- Methodischer Umfang/Fokus

Beispiel:
Anfrage: "Nachhaltigkeit und Profit bei Windenergie?"
1. Was bedeutet "Nachhaltigkeit" hier - ökologisch, ökonomisch, sozial?
2. "Profit" für Betreiber oder volkswirtschaftlicher Nutzen?
3. Zeitraum: Aktuelle Situation oder Zukunftsprognose?

Ausgabeformat - genau 3 nummerierte Fragen, keine Erklärungen:
1. [Frage zur Definition/Kontext]
2. [Frage zu Details]
3. [Frage zu Umfang]

Antworte NUR mit den 3 nummerierten Fragen.
"""

FOLLOW_UP_QUESTIONS_EN = """You are a research assistant. Analyze the query, context, and knowledge base. Ask precise questions.

Query: “{user_query}”

Context: {context}

Knowledge base: {retrieval}

Ask exactly 3 questions:

Focus:
- Clarification of terms and context
- Missing or unclear details  
- Methodological scope/focus

Example:
Query: “Sustainability and profit in wind energy?”
1. What does “sustainability” mean here—ecological, economic, social?
2. “Profit” for operators or economic benefit?
3. Time frame: Current situation or future forecast?

Output format - exactly 3 numbered questions, no explanations:
1. [Question about definition/context]
2. [Question about details]
3. [Question about scope]

Answer ONLY with the 3 numbered questions."""

# =============================================================================
# HITL Prompts - User Feedback Analysis
# =============================================================================

USER_FEEDBACK_ANALYSIS_PROMPT = """Analysiere diese Konversation und extrahiere die wichtigsten Informationen.

Ursprüngliche Anfrage: "{user_query}"

Konversationsverlauf:
{context}

Extrahiere und gib als JSON zurück:
{{"entities": ["Liste relevanter Entitäten/Vorschriften"],
"scope": "Themenbereich der Anfrage",
"context": "Zusätzlicher Kontext",
"refined_query": "Verfeinerte Suchanfrage"}}

JSON:"""

# =============================================================================
# HITL Prompts - Knowledge Base Questions Generation
# =============================================================================

KNOWLEDGE_BASE_QUESTIONS_PROMPT = """Basierend auf dieser Konversation, erstelle {max_queries} optimierte Suchanfragen für eine Wissensdatenbank.

Ursprüngliche Anfrage: "{user_query}"

Konversationsverlauf:
{context}

Extrahierte Informationen: {analysis}

Erstelle {max_queries} verschiedene, spezifische Suchanfragen, die verschiedene Aspekte der Anfrage abdecken.
Jede Anfrage sollte auf einen Aspekt fokussiert sein.

Gib als JSON zurück:
{{"research_queries": ["Anfrage 1", "Anfrage 2", ...],
"summary": "Kurze Zusammenfassung der Forschungsrichtung"}}

JSON:"""

# =============================================================================
# HITL Prompts - Alternative Queries Generation
# =============================================================================

ALTERNATIVE_QUERIES_INITIAL_PROMPT = """Generate 2 alternative search queries for this research question.

Original query: "{query}"

Create:
1. broader_scope: A query that explores related/contextual information
2. alternative_angle: A query that explores implications, challenges, or alternatives

Output as JSON:
{{"broader_scope": "...", "alternative_angle": "..."}}

JSON:"""

ALTERNATIVE_QUERIES_REFINED_PROMPT = """Generate 2 refined search queries based on the research progress.

Original query: "{query}"
Entities found: {entities}
Knowledge gaps: {gaps}

Create:
1. broader_scope: A query addressing the identified knowledge gaps
2. alternative_angle: A query exploring newly mentioned concepts

Output as JSON:
{{"broader_scope": "...", "alternative_angle": "..."}}

JSON:"""

# =============================================================================
# HITL Prompts - Retrieval Analysis
# =============================================================================

RETRIEVAL_ANALYSIS_PROMPT = """User's Research Query: {query}

Retrieved Context (from knowledge base):
{retrieval}

Perform comprehensive analysis:
1. KEY CONCEPTS: 5-7 core concepts from query + retrieved content
2. ENTITIES: Named entities (organizations, dates, technical terms)
3. SCOPE: Primary focus area
4. KNOWLEDGE GAPS: Specific missing information (be concrete, not "more details")
5. COVERAGE: 0-100% estimate considering foundational, intermediate, advanced coverage

Output as JSON:
{{"key_concepts": ["..."], "entities": ["..."], "scope": "...", "knowledge_gaps": ["..."], "coverage_score": 0.XX}}

JSON:"""

# =============================================================================
# HITL Prompts - Refined Queries
# =============================================================================

REFINED_QUERIES_PROMPT = """Original query: "{query}"
User's clarification: "{user_response}"
Identified gaps: {gaps}

Generate 3 refined search queries:
1. query_1: Addresses the identified knowledge gaps
2. query_2: Explores new concepts mentioned by the user
3. query_3: Clarifies the updated scope

Output as JSON:
{{"query_1": "...", "query_2": "...", "query_3": "..."}}

JSON:"""

# =============================================================================
# Research Prompts - ToDo Generation
# =============================================================================

TODO_GENERATION_PROMPT = """Generate a research task list for this query analysis.

Original Query: "{original_query}"
Key Concepts: {key_concepts}
Entities: {entities}
Scope: {scope}
Context: {assumed_context}

Generate {num_items} specific, actionable research tasks.
Each task should be:
- Specific and measurable
- Focused on finding concrete information
- Related to the query concepts

Return JSON with "items" array, each item having:
- id: integer starting from 1
- task: string describing the task
- context: string explaining why this task is needed

Example:
{{"items": [{{"id": 1, "task": "Find dose limit regulations", "context": "Core query requirement"}}]}}"""

# =============================================================================
# Research Prompts - Information Extraction
# =============================================================================

INFO_EXTRACTION_PROMPT = """Given this search query: "{query}"

Extract only the relevant passages from this text chunk. Be concise and focus on information that directly answers or relates to the query.

Text chunk:
{chunk_text}

Extracted relevant information (in the same language as the chunk):"""

# =============================================================================
# Research Prompts - Synthesis
# =============================================================================

SYNTHESIS_PROMPT = """Synthesize these research findings into a coherent answer.

Original Query: "{original_query}"

Research Findings:
{findings}

Provide:
1. summary: A comprehensive answer to the query (in {language})
2. key_findings: List of the most important findings

Include source citations in the format [Document_name.pdf, Page X] where applicable."""

# =============================================================================
# Research Prompts - Quality Check
# =============================================================================

QUALITY_CHECK_PROMPT = """Evaluate the quality of this research summary.

Summary:
{summary}

Score each dimension from 0-100:
1. factual_accuracy: Are claims factually correct?
2. semantic_validity: Does it make logical sense?
3. structural_integrity: Is it well-organized?
4. citation_correctness: Are sources properly cited?

Also list any issues_found as a list of strings."""
