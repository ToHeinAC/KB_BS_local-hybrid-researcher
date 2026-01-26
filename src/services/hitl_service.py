"""Human-In-The-Loop service for interactive research sessions."""

import logging
from datetime import datetime

from src.models.hitl import (
    ClarificationQuestion,
    HITLCheckpoint,
    HITLDecision,
    HITLState,
)
from src.models.query import QueryAnalysis, ToDoItem, ToDoList

logger = logging.getLogger(__name__)


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
