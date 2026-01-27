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

    # Chat-style HITL methods

    def detect_language(self, user_query: str) -> str:
        """Detect the language of the user's query using LLM.

        Args:
            user_query: The user's input query

        Returns:
            Language code ('de' for German, 'en' for English)
        """
        client = get_ollama_client()

        prompt = f"""Determine the language of the following text.
Reply with ONLY 'de' for German or 'en' for English.

Text: "{user_query}"

Language code:"""

        try:
            response = client.generate(prompt)
            lang = response.strip().lower()[:2]
            if lang in ("de", "en"):
                return lang
            # Default to German for radiation protection domain
            return "de"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "de"

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
        client = get_ollama_client()

        user_query = state.get("user_query", "")
        conversation_history = state.get("conversation_history", [])

        # Build conversation context
        context = ""
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            context += f"{role}: {content}\n"

        if language == "de":
            prompt = f"""Du bist ein Forschungsassistent für Strahlenschutz-Dokumentation.
Der Benutzer hat folgende Anfrage: "{user_query}"

Bisheriger Konversationsverlauf:
{context}

Stelle 2-3 präzise Nachfragen, um die Anfrage besser zu verstehen.
Fokussiere auf:
- Relevante Vorschriften (StrlSchG, StrlSchV, etc.)
- Spezifische Anwendungsfälle
- Kontext der Anfrage

Formatiere als nummerierte Liste. Antworte NUR mit den Fragen:"""
        else:
            prompt = f"""You are a research assistant for radiation protection documentation.
The user has the following query: "{user_query}"

Conversation history so far:
{context}

Ask 2-3 precise follow-up questions to better understand the request.
Focus on:
- Relevant regulations (StrlSchG, StrlSchV, etc.)
- Specific use cases
- Context of the request

Format as a numbered list. Reply ONLY with the questions:"""

        try:
            return client.generate(prompt)
        except Exception as e:
            logger.warning(f"Failed to generate follow-up questions: {e}")
            if language == "de":
                return "1. Welchen Bereich betrifft Ihre Anfrage?\n2. Gibt es spezifische Vorschriften, die relevant sind?"
            return "1. What area does your request concern?\n2. Are there specific regulations that are relevant?"

    def analyse_user_feedback(self, state: dict) -> dict:
        """Analyze user feedback for insights.

        Args:
            state: Current HITL conversation state with user responses

        Returns:
            Dict with extracted insights (entities, scope, context)
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

        prompt = f"""Analysiere diese Konversation und extrahiere die wichtigsten Informationen.

Ursprüngliche Anfrage: "{user_query}"

Konversationsverlauf:
{context}

Extrahiere und gib als JSON zurück:
{{"entities": ["Liste relevanter Entitäten/Vorschriften"],
"scope": "Themenbereich der Anfrage",
"context": "Zusätzlicher Kontext",
"refined_query": "Verfeinerte Suchanfrage"}}

JSON:"""

        try:
            response = client.generate(prompt)
            # Try to parse JSON
            import json
            # Find JSON in response
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

        prompt = f"""Basierend auf dieser Konversation, erstelle {max_queries} optimierte Suchanfragen für eine Wissensdatenbank.

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

        try:
            response = client.generate(prompt)
            # Try to parse JSON
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                # Ensure we have the expected structure
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
