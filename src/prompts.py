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

FOLLOW_UP_QUESTIONS_DE = """Du bist ein Forschungsassistent innerhalb des Agentensystems mit der Aufgabe, präzise Fragen zu stellen. 
Du bist in der Lage, tiefe Kontextinformationen aus Konversation mit dem Nutzer und Abfragen der Wissensdatenbank zu extrahieren.
Deine Aufgabe ist es, präzise Fragen zu stellen, um die Anfrage besser zu verstehen und den Raum der Möglichkeiten für korrekte Antworten des Agentensystems bestmöglich zu erforschen..

Der Benutzer hat folgende Anfrage: "{user_query}"

Bisheriger Konversationsverlauf:
{context}

Relevante Informationen aus der Wissensdatenbank:
{retrieval}

Stelle 2-3 präzise Nachfragen, die 
- präzissieren und 
- den Raum der Möglichkeiten für korrekte Antworten des Agentensystems bestmöglich erweitern.
Fokussiere auf:
- Definition, Bedeutung und Kontext von Begriffen der Anfrage
- Lücken in den gefundenen Informationen
- Unklare Aspekte der Anfrage
- Spezifische Details die noch fehlen
- Methodische Klärungen der Nachforschung

Gutes Beispiel:
Anfrage: "Wie hängen Nachhaltigkeit und Profit bie der Windenergieerzeugung zusammen?"
präzise Nachfragen:
- Was bedeutet Nachhaltigkeit bezüglich der Windenergieerzeugung?
- Was ist unter Profit im Zusammenhang mit der Winernergieerzeugung zu verstehen?
- Sind alle Aspekte des Zusammenhangs zwischen Nachhaltigkeit und Profit im Zusammenhang mit der Windenergieerzeugung zu berücksichtigen oder gibt es einen FOkus auf spezifische Aspekte?    

Formatiere als nummerierte Liste. Antworte NUR mit den Fragen:"""

FOLLOW_UP_QUESTIONS_EN = """You are a research assistant within the agent system with the task of asking precise questions. 
You are able to extract in-depth contextual information from conversations with users and queries of the knowledge database.
Your task is to ask precise questions in order to better understand the query and to explore the range of possibilities for correct answers from the agent system in the best possible way.
The user has the following query: “{user_query}”
Previous conversation history:
{context}
Relevant information from the knowledge base:
{retrieval}
Ask 2-3 precise follow-up questions that 
- clarify and 
- expand the range of possibilities for correct answers from the agent system as much as possible.
Focus on:
- Definition, meaning, and context of terms in the query
- Gaps in the information found
- Unclear aspects of the query
- Specific details that are still missing
- Methodological clarifications of the research
Good example:
Query: “How are sustainability and profit related in wind energy production?”
Precise follow-up questions:
- What does sustainability mean in relation to wind energy production?
- What is meant by profit in the context of wind energy production?
- Should all aspects of the relationship between sustainability and profit in relation to wind energy production be taken into account, or is there a focus on specific aspects?
Format as a numbered list. Answer ONLY with the questions:"""

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
