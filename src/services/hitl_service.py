"""Human-In-The-Loop service for interactive research sessions."""

import logging
from datetime import datetime

from src.models.hitl import (
    ClarificationQuestion,
    HITLCheckpoint,
    HITLDecision,
    HITLState,
)
from src.models.query import QueryAnalysis, ToDoList
from src.prompts import (
    ALTERNATIVE_QUERIES_INITIAL_PROMPT,
    ALTERNATIVE_QUERIES_REFINED_PROMPT,
    FOLLOW_UP_QUESTIONS_DE,
    FOLLOW_UP_QUESTIONS_EN,
    KNOWLEDGE_BASE_QUESTIONS_PROMPT,
    LANGUAGE_DETECTION_PROMPT,
    REFINED_QUERIES_PROMPT,
    RETRIEVAL_ANALYSIS_PROMPT,
    USER_FEEDBACK_ANALYSIS_PROMPT,
)
from src.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Singleton Ollama client for HITL service
_ollama_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


class HITLService:
    """Service for managing HITL interactions."""

    def __init__(self, max_questions: int = 3):
        """Initialize HITL service.

        Args:
            max_questions: Maximum clarification questions per round
        """
        self.max_questions = max_questions

    def generate_clarification_questions(
        self,
        analysis: QueryAnalysis,
    ) -> list[ClarificationQuestion]:
        """Generate clarification questions based on query analysis.

        Args:
            analysis: The analyzed query

        Returns:
            List of clarification questions
        """
        questions = []

        # Q1: Scope clarification
        if not analysis.scope or analysis.scope == "":
            questions.append(
                ClarificationQuestion(
                    id="scope",
                    question="Welchen Bereich betrifft Ihre Anfrage?",
                    options=[
                        "Gesetzliche Vorschriften (StrlSchG/StrlSchV)",
                        "Technische Anforderungen",
                        "Genehmigungsverfahren",
                        "Allgemeine Information",
                    ],
                    context="Um die Suche einzugrenzen",
                )
            )

        # Q2: Entity clarification
        if len(analysis.entities) == 0:
            questions.append(
                ClarificationQuestion(
                    id="entities",
                    question="Welche Vorschriften sind relevant?",
                    options=[
                        "Strahlenschutzgesetz (StrlSchG)",
                        "Strahlenschutzverordnung (StrlSchV)",
                        "Konrad-Unterlagen",
                        "Alle durchsuchen",
                    ],
                    context="Um relevante Dokumente zu identifizieren",
                )
            )

        # Q3: Context clarification
        if analysis.clarification_needed and len(analysis.assumed_context) < 2:
            questions.append(
                ClarificationQuestion(
                    id="context",
                    question="Gibt es zusätzlichen Kontext?",
                    options=[],  # Free text
                    context="Zusätzliche Informationen helfen bei der Suche",
                    required=False,
                )
            )

        return questions[: self.max_questions]

    def merge_clarifications(
        self,
        analysis: QueryAnalysis,
        answers: dict[str, str],
    ) -> QueryAnalysis:
        """Merge user answers into query analysis.

        Args:
            analysis: Original query analysis
            answers: User's answers keyed by question ID

        Returns:
            Updated query analysis
        """
        # Create updated analysis
        updated = analysis.model_copy(deep=True)

        # Process scope answer
        if "scope" in answers and answers["scope"]:
            updated.scope = answers["scope"]
            updated.hitl_refinements.append(f"Scope: {answers['scope']}")

        # Process entities answer
        if "entities" in answers and answers["entities"]:
            entity = answers["entities"]
            if entity not in updated.entities:
                updated.entities.append(entity)
            updated.hitl_refinements.append(f"Entity: {entity}")

        # Process context answer
        if "context" in answers and answers["context"]:
            updated.assumed_context.append(answers["context"])
            updated.hitl_refinements.append(f"Context: {answers['context']}")

        # Mark as no longer needing clarification
        updated.clarification_needed = False

        logger.info(f"Merged {len(answers)} clarifications into query analysis")
        return updated

    def create_query_checkpoint(
        self,
        questions: list[ClarificationQuestion],
    ) -> HITLCheckpoint:
        """Create a checkpoint for query clarification.

        Args:
            questions: Clarification questions to present

        Returns:
            HITL checkpoint
        """
        return HITLCheckpoint(
            checkpoint_type="query_clarify",
            content={
                "questions": [q.model_dump() for q in questions],
            },
            requires_approval=True,
            phase="Phase 1: Query Analysis",
            timestamp=datetime.now().isoformat(),
        )

    def create_todo_checkpoint(
        self,
        todo_list: ToDoList,
    ) -> HITLCheckpoint:
        """Create a checkpoint for ToDoList approval.

        Args:
            todo_list: The generated ToDoList

        Returns:
            HITL checkpoint
        """
        return HITLCheckpoint(
            checkpoint_type="todo_approve",
            content={
                "items": [item.model_dump() for item in todo_list.items],
                "max_items": todo_list.max_items,
            },
            requires_approval=True,
            phase="Phase 2: Research Planning",
            timestamp=datetime.now().isoformat(),
        )

    def apply_todo_modifications(
        self,
        original: ToDoList,
        decision: HITLDecision,
    ) -> ToDoList:
        """Apply user modifications to ToDoList.

        Args:
            original: Original ToDoList
            decision: User's decision with modifications

        Returns:
            Modified ToDoList
        """
        if not decision.modifications:
            return original

        modified = original.model_copy(deep=True)

        # Handle item removals
        if "removed_ids" in decision.modifications:
            removed = set(decision.modifications["removed_ids"])
            modified.items = [
                item for item in modified.items if item.id not in removed
            ]

        # Handle item edits
        if "edited_items" in decision.modifications:
            for edited in decision.modifications["edited_items"]:
                for item in modified.items:
                    if item.id == edited["id"]:
                        if "task" in edited:
                            item.task = edited["task"]
                        if "context" in edited:
                            item.context = edited["context"]

        # Handle new items
        if "new_items" in decision.modifications:
            for new_item in decision.modifications["new_items"]:
                modified.add_task(
                    task=new_item.get("task", ""),
                    context=new_item.get("context", ""),
                )

        logger.info(
            f"Applied modifications: {len(modified.items)} items "
            f"(was {len(original.items)})"
        )
        return modified

    def process_checkpoint_response(
        self,
        state: HITLState,
        decision: HITLDecision,
    ) -> HITLState:
        """Process a checkpoint response and update state.

        Args:
            state: Current HITL state
            decision: User's decision

        Returns:
            Updated HITL state
        """
        if state.checkpoint:
            state.add_to_history(state.checkpoint.checkpoint_type, decision)

        state.clear_pending()
        return state

    def should_request_clarification(
        self,
        analysis: QueryAnalysis,
    ) -> bool:
        """Determine if clarification should be requested.

        Args:
            analysis: Query analysis

        Returns:
            Whether to request clarification
        """
        # Request if explicitly marked
        if analysis.clarification_needed:
            return True

        # Request if lacking key information
        if not analysis.scope and not analysis.entities:
            return True

        # Request if query is very short
        if len(analysis.original_query.split()) < 3:
            return True

        return False

    # Chat-style HITL methods (legacy - kept for backwards compatibility)

    def detect_language(self, user_query: str) -> str:
        """Detect the language of the user's query using LLM.

        Args:
            user_query: The user's input query

        Returns:
            Language code ('de' for German, 'en' for English)
        """
        return _detect_language_llm(user_query)

    def generate_follow_up_questions(
        self,
        state: dict,
        language: str = "de",
    ) -> str:
        """Generate follow-up questions for the conversation.

        Args:
            state: Current HITL conversation state
            language: Language for questions ('de' or 'en')

        Returns:
            Follow-up questions as formatted string
        """
        return _generate_follow_up_questions_llm(state, language)

    def analyse_user_feedback(self, state: dict) -> dict:
        """Analyze user feedback for insights.

        Args:
            state: Current HITL conversation state with user responses

        Returns:
            Dict with extracted insights (entities, scope, context)
        """
        return _analyse_user_feedback_llm(state)

    def generate_knowledge_base_questions(
        self,
        state: dict,
        max_queries: int = 5,
    ) -> dict:
        """Generate final research queries from conversation.

        Args:
            state: Current HITL conversation state
            max_queries: Maximum number of search queries to generate

        Returns:
            Dict with research_queries list and summary
        """
        return _generate_knowledge_base_questions_llm(state, max_queries)


# --- Explicit HITL Functions (Reference Pattern) ---


def _detect_language_llm(user_query: str) -> str:
    """Detect query language using LLM."""
    client = get_ollama_client()

    prompt = LANGUAGE_DETECTION_PROMPT.format(user_query=user_query)

    try:
        response = client.generate(prompt)
        lang = response.strip().lower()[:2]
        if lang in ("de", "en"):
            return lang
        return "de"
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "de"


def _generate_follow_up_questions_llm(
    state: dict, language: str = "de", retrieval: str = ""
) -> str:
    """Generate follow-up questions using LLM.

    Args:
        state: Current HITL state with user_query and conversation_history
        language: Language for questions ('de' or 'en')
        retrieval: Retrieved context from vector DB to inform questions

    Returns:
        Follow-up questions as formatted string
    """
    client = get_ollama_client()

    user_query = state.get("user_query", "")
    conversation_history = state.get("conversation_history", [])

    # Build conversation context
    context = ""
    for msg in conversation_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        context += f"{role}: {content}\n"

    # Prepare retrieval context (with fallback message)
    if language == "de":
        retrieval_text = retrieval or "Noch keine Informationen abgerufen."
    else:
        retrieval_text = retrieval or "No information retrieved yet."

    if language == "de":
        prompt = FOLLOW_UP_QUESTIONS_DE.format(
            user_query=user_query, context=context, retrieval=retrieval_text
        )
    else:
        prompt = FOLLOW_UP_QUESTIONS_EN.format(
            user_query=user_query, context=context, retrieval=retrieval_text
        )

    try:
        return client.generate(prompt)
    except Exception as e:
        logger.warning(f"Failed to generate follow-up questions: {e}")
        if language == "de":
            return "1. Welchen Bereich betrifft Ihre Anfrage?\n2. Gibt es spezifische Vorschriften, die relevant sind?"
        return "1. What area does your request concern?\n2. Are there specific regulations that are relevant?"


def _analyse_user_feedback_llm(state: dict) -> dict:
    """Analyze user feedback using LLM."""
    import json

    client = get_ollama_client()

    user_query = state.get("user_query", "")
    conversation_history = state.get("conversation_history", [])

    # Build conversation context
    context = ""
    for msg in conversation_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        context += f"{role}: {content}\n"

    prompt = USER_FEEDBACK_ANALYSIS_PROMPT.format(user_query=user_query, context=context)

    try:
        response = client.generate(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except Exception as e:
        logger.warning(f"Failed to analyze feedback: {e}")

    return {
        "entities": [],
        "scope": "",
        "context": "",
        "refined_query": user_query,
    }


def _generate_knowledge_base_questions_llm(state: dict, max_queries: int = 5) -> dict:
    """Generate research queries using LLM."""
    import json

    client = get_ollama_client()

    user_query = state.get("user_query", "")
    conversation_history = state.get("conversation_history", [])
    analysis = state.get("analysis", {})

    # Build conversation context
    context = ""
    for msg in conversation_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        context += f"{role}: {content}\n"

    prompt = KNOWLEDGE_BASE_QUESTIONS_PROMPT.format(
        max_queries=max_queries,
        user_query=user_query,
        context=context,
        analysis=analysis,
    )

    try:
        response = client.generate(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
            if "research_queries" not in result:
                result["research_queries"] = [user_query]
            if "summary" not in result:
                result["summary"] = f"Recherche zu: {user_query}"
            return result
    except Exception as e:
        logger.warning(f"Failed to generate KB questions: {e}")

    return {
        "research_queries": [user_query],
        "summary": f"Recherche zu: {user_query}",
    }


# --- Four Explicit HITL Functions (Reference App Pattern) ---


def initialize_hitl_state(user_query: str) -> dict:
    """Initialize HITL state for a new conversation.

    Step 1 of the HITL flow. Detects language and sets up conversation state.

    Args:
        user_query: The user's initial research query

    Returns:
        Dict with: user_query, detected_language, conversation_history,
                   human_feedback=[], analysis={}
    """
    language = _detect_language_llm(user_query)

    return {
        "user_query": user_query,
        "detected_language": language,
        "conversation_history": [{"role": "user", "content": user_query}],
        "human_feedback": [],
        "analysis": {},
    }


def process_initial_query(hitl_state: dict) -> dict:
    """Process the initial query and generate follow-up questions.

    Step 2 of the HITL flow. Generates 2-3 preliminary questions.

    Args:
        hitl_state: Current HITL state from initialize_hitl_state()

    Returns:
        Updated state with follow_up_questions field
    """
    language = hitl_state.get("detected_language", "de")
    query_retrieval = hitl_state.get("query_retrieval", "")

    questions = _generate_follow_up_questions_llm(
        hitl_state, language, retrieval=query_retrieval
    )

    # Add assistant message to history
    hitl_state["conversation_history"].append({
        "role": "assistant",
        "content": questions,
    })
    hitl_state["follow_up_questions"] = questions

    return hitl_state


def process_human_feedback(hitl_state: dict, user_response: str) -> dict:
    """Process user feedback and update analysis.

    Step 3 of the HITL flow (repeatable). Analyzes response, updates
    accumulated analysis, and generates refined questions if needed.

    Args:
        hitl_state: Current HITL state
        user_response: User's response to follow-up questions

    Returns:
        Updated state with analysis and optional new follow_up_questions
    """
    # Add user response to history
    hitl_state["conversation_history"].append({
        "role": "user",
        "content": user_response,
    })
    hitl_state["human_feedback"].append(user_response)

    # Analyze accumulated feedback
    analysis = _analyse_user_feedback_llm(hitl_state)
    hitl_state["analysis"] = analysis

    # Generate refined follow-up questions with retrieval context
    language = hitl_state.get("detected_language", "de")
    query_retrieval = hitl_state.get("query_retrieval", "")
    questions = _generate_follow_up_questions_llm(
        hitl_state, language, retrieval=query_retrieval
    )

    hitl_state["conversation_history"].append({
        "role": "assistant",
        "content": questions,
    })
    hitl_state["follow_up_questions"] = questions

    return hitl_state


def finalize_hitl_conversation(hitl_state: dict, max_queries: int = 5) -> dict:
    """Finalize HITL conversation and generate research queries.

    Step 4 of the HITL flow. Called when user types /end.
    Generates optimized research_queries for Phase 2.

    Args:
        hitl_state: Current HITL state with accumulated analysis
        max_queries: Maximum number of research queries to generate

    Returns:
        Dict with: research_queries, summary, analysis, user_query,
                   entities, scope, context
    """
    # Final analysis pass
    analysis = _analyse_user_feedback_llm(hitl_state)
    hitl_state["analysis"] = analysis

    # Generate research queries
    result = _generate_knowledge_base_questions_llm(hitl_state, max_queries)

    # Return complete result for Phase 2 handoff
    return {
        "research_queries": result.get("research_queries", []),
        "summary": result.get("summary", ""),
        "analysis": analysis,
        "user_query": hitl_state.get("user_query", ""),
        "entities": analysis.get("entities", []),
        "scope": analysis.get("scope", ""),
        "context": analysis.get("context", ""),
        "detected_language": hitl_state.get("detected_language", "de"),
        "conversation_history": hitl_state.get("conversation_history", []),
    }


def format_analysis_dict(analysis: dict | str) -> str:
    """Format analysis dict or string into markdown with German labels.

    Args:
        analysis: Dict with entities/scope/context or a string

    Returns:
        Formatted markdown string
    """
    if isinstance(analysis, str):
        return analysis

    if not isinstance(analysis, dict):
        return str(analysis)

    # German label mappings
    label_map = {
        "entities": "Entitaeten",
        "scope": "Umfang",
        "context": "Kontext",
        "refined_query": "Verfeinerte Anfrage",
        "key_concepts": "Schluesselkonzepte",
        "assumed_context": "Angenommener Kontext",
    }

    lines = []
    for key, value in analysis.items():
        if not value:
            continue

        label = label_map.get(key, key.replace("_", " ").title())

        if isinstance(value, list):
            if value:
                items = ", ".join(f"`{v}`" for v in value)
                lines.append(f"**{label}:** {items}")
        elif isinstance(value, str):
            lines.append(f"**{label}:** {value}")

    return "\n".join(lines) if lines else ""


def _deep_search_dict(data: dict | list, target_keys: set[str]) -> dict:
    """Recursively search nested dicts/lists for target keys.

    Args:
        data: Dict or list to search
        target_keys: Set of keys to find

    Returns:
        Dict with found key-value pairs
    """
    result = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if key in target_keys:
                result[key] = value
            elif isinstance(value, (dict, list)):
                nested = _deep_search_dict(value, target_keys)
                result.update(nested)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                nested = _deep_search_dict(item, target_keys)
                result.update(nested)

    return result


def parse_structured_llm_output(final_answer: str | dict) -> tuple[str, str]:
    """Extract thinking and final content from LLM responses.

    Handles JSON strings, nested dicts, and removes <think> blocks.

    Args:
        final_answer: LLM response (string or dict)

    Returns:
        Tuple of (final_content, thinking_content)
    """
    import json
    import re

    thinking_content = ""
    final_content = ""

    # Handle dict input
    if isinstance(final_answer, dict):
        # Search for common content keys
        content_keys = {"content", "text", "response", "answer", "output"}
        found = _deep_search_dict(final_answer, content_keys)
        for key in content_keys:
            if key in found and found[key]:
                final_answer = found[key]
                break
        else:
            # Convert dict to string if no content key found
            final_answer = json.dumps(final_answer, ensure_ascii=False)

    if not isinstance(final_answer, str):
        final_answer = str(final_answer)

    # Extract <think> blocks
    think_pattern = r"<think>(.*?)</think>"
    think_matches = re.findall(think_pattern, final_answer, re.DOTALL)
    if think_matches:
        thinking_content = "\n".join(think_matches)
        # Remove think blocks from output
        final_content = re.sub(think_pattern, "", final_answer, flags=re.DOTALL)
    else:
        final_content = final_answer

    # Clean up whitespace
    final_content = final_content.strip()
    thinking_content = thinking_content.strip()

    # Try to parse as JSON and extract content
    if final_content.startswith("{"):
        try:
            parsed = json.loads(final_content)
            if isinstance(parsed, dict):
                content_keys = {"content", "text", "response", "answer", "output"}
                for key in content_keys:
                    if key in parsed and parsed[key]:
                        final_content = parsed[key]
                        break
        except json.JSONDecodeError:
            pass

    return final_content, thinking_content


# --- Enhanced Phase 1: Multi-Vector Retrieval Helpers ---


def generate_alternative_queries_llm(
    query: str,
    analysis: dict,
    iteration: int,
) -> list[str]:
    """Generate 3 queries: original + broader + alternative angle.

    Args:
        query: Original user query
        analysis: Current query analysis (may be empty on iteration 0)
        iteration: Current HITL iteration

    Returns:
        List of 3 queries [original, broader, alternative]
    """
    import json

    client = get_ollama_client()

    if iteration == 0 or not analysis:
        prompt = ALTERNATIVE_QUERIES_INITIAL_PROMPT.format(query=query)
    else:
        gaps = analysis.get("knowledge_gaps", [])
        entities = analysis.get("entities", [])
        prompt = ALTERNATIVE_QUERIES_REFINED_PROMPT.format(
            query=query, entities=entities, gaps=gaps
        )

    try:
        response = client.generate(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            return [
                query,
                parsed.get("broader_scope", f"{query} Hintergrund"),
                parsed.get("alternative_angle", f"{query} Anwendung"),
            ]
    except Exception as e:
        logger.warning(f"Alternative query generation failed: {e}")

    # Fallback
    return [query, f"{query} Hintergrund", f"{query} Anwendung"]


def analyze_retrieval_context_llm(query: str, retrieval_text: str) -> dict:
    """Analyze accumulated retrieval for gaps, concepts, scope.

    Args:
        query: Original user query
        retrieval_text: Accumulated retrieval text (may be truncated)

    Returns:
        Dict with key_concepts, entities, scope, knowledge_gaps, coverage_score
    """
    import json

    client = get_ollama_client()

    # Truncate retrieval to avoid token limits
    max_chars = 3000
    truncated = retrieval_text[:max_chars] if len(retrieval_text) > max_chars else retrieval_text

    prompt = RETRIEVAL_ANALYSIS_PROMPT.format(query=query, retrieval=truncated)

    try:
        response = client.generate(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            # Normalize coverage_score to 0-1
            score = parsed.get("coverage_score", 0)
            if isinstance(score, (int, float)):
                if score > 1:
                    score = score / 100.0
                parsed["coverage_score"] = min(1.0, max(0.0, score))
            return parsed
    except Exception as e:
        logger.warning(f"Retrieval analysis failed: {e}")

    return {
        "key_concepts": [],
        "entities": [],
        "scope": "",
        "knowledge_gaps": [],
        "coverage_score": 0.0,
    }


def generate_refined_queries_llm(
    query: str,
    user_response: str,
    gaps: list[str],
) -> list[str]:
    """Generate 3 refined queries based on user feedback.

    Args:
        query: Original user query
        user_response: User's response to follow-up questions
        gaps: Identified knowledge gaps

    Returns:
        List of 3 refined queries
    """
    import json

    client = get_ollama_client()

    prompt = REFINED_QUERIES_PROMPT.format(
        query=query, user_response=user_response, gaps=gaps
    )

    try:
        response = client.generate(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            return [
                parsed.get("query_1", query),
                parsed.get("query_2", f"{query} {user_response[:50]}"),
                parsed.get("query_3", query),
            ]
    except Exception as e:
        logger.warning(f"Refined query generation failed: {e}")

    # Fallback: use original + user keywords
    user_words = user_response.split()[:3]
    return [
        query,
        f"{query} {' '.join(user_words)}" if user_words else query,
        query,
    ]


def calculate_dedup_ratio(
    new_chunks: list,
    existing_retrieval: str,
    threshold: float = 0.7,
) -> tuple[list, dict]:
    """Filter duplicates and return unique chunks with stats.

    Uses simple substring matching for deduplication.

    Args:
        new_chunks: List of VectorResult objects
        existing_retrieval: Accumulated retrieval text
        threshold: Similarity threshold for duplicate detection

    Returns:
        Tuple of (unique_chunks, dedup_stats)
        dedup_stats: {"new_count": int, "dup_count": int, "dedup_ratio": float}
    """
    unique_chunks = []
    dup_count = 0

    existing_lower = existing_retrieval.lower() if existing_retrieval else ""

    for chunk in new_chunks:
        # Get chunk text - handle both VectorResult objects and dicts
        if hasattr(chunk, "chunk_text"):
            chunk_text = chunk.chunk_text
        elif isinstance(chunk, dict):
            chunk_text = chunk.get("chunk_text", chunk.get("text", ""))
        else:
            chunk_text = str(chunk)

        # Simple substring match for deduplication
        chunk_lower = chunk_text.lower()[:200]  # Compare first 200 chars

        if chunk_lower in existing_lower:
            dup_count += 1
        else:
            unique_chunks.append(chunk)

    total = len(new_chunks)
    new_count = len(unique_chunks)
    dedup_ratio = dup_count / total if total > 0 else 0.0

    return unique_chunks, {
        "new_count": new_count,
        "dup_count": dup_count,
        "dedup_ratio": dedup_ratio,
    }


def format_chunks_for_state(chunks: list, queries: list[str]) -> str:
    """Format retrieved chunks for state.query_retrieval.

    Args:
        chunks: List of VectorResult objects
        queries: List of queries used for retrieval

    Returns:
        Formatted string for query_retrieval state
    """
    lines = []
    for chunk in chunks:
        # Handle both VectorResult objects and dicts
        if hasattr(chunk, "chunk_text"):
            text = chunk.chunk_text
            doc = getattr(chunk, "doc_name", "unknown")
            page = getattr(chunk, "page_number", 0)
        elif isinstance(chunk, dict):
            text = chunk.get("chunk_text", chunk.get("text", ""))
            doc = chunk.get("doc_name", chunk.get("document", "unknown"))
            page = chunk.get("page_number", chunk.get("page", 0))
        else:
            text = str(chunk)
            doc = "unknown"
            page = 0

        lines.append(f"[{doc}, p.{page}]: {text[:500]}")

    return "\n---\n".join(lines)

