"""Human-In-The-Loop (HITL) models."""

from typing import Literal

from pydantic import BaseModel, Field


class ClarificationQuestion(BaseModel):
    """A clarification question for the user."""

    id: str = Field(description="Unique question identifier")
    question: str = Field(description="The question text")
    options: list[str] = Field(
        default_factory=list,
        description="Predefined options if any",
    )
    context: str = Field(
        default="",
        description="Context explaining why this question is asked",
    )
    required: bool = Field(
        default=False,
        description="Whether an answer is required",
    )
    answer: str | None = Field(
        default=None,
        description="User's answer",
    )


class HITLCheckpoint(BaseModel):
    """Checkpoint requiring user validation."""

    checkpoint_type: Literal["query_clarify", "todo_approve", "findings_review", "sources_verify"] = Field(
        description="Type of checkpoint",
    )
    content: dict = Field(
        default_factory=dict,
        description="Content to present to user",
    )
    requires_approval: bool = Field(
        default=True,
        description="Whether user approval is required",
    )
    phase: str = Field(
        default="",
        description="Current phase of the research",
    )
    timestamp: str = Field(
        default="",
        description="When checkpoint was created",
    )


class HITLDecision(BaseModel):
    """User's decision at HITL checkpoint."""

    approved: bool = Field(description="Whether user approved")
    modifications: dict | None = Field(
        default=None,
        description="Any modifications made by user",
    )
    feedback: str | None = Field(
        default=None,
        description="Optional feedback from user",
    )
    skip_reason: str | None = Field(
        default=None,
        description="Reason if user skipped",
    )


class HITLState(BaseModel):
    """State for HITL interactions."""

    pending: bool = Field(
        default=False,
        description="Whether HITL interaction is pending",
    )
    checkpoint: HITLCheckpoint | None = Field(
        default=None,
        description="Current checkpoint if any",
    )
    history: list[dict] = Field(
        default_factory=list,
        description="History of HITL interactions",
    )
    clarification_questions: list[ClarificationQuestion] = Field(
        default_factory=list,
        description="Pending clarification questions",
    )

    def add_to_history(
        self,
        checkpoint_type: str,
        decision: HITLDecision,
    ) -> None:
        """Add a decision to history."""
        self.history.append({
            "checkpoint_type": checkpoint_type,
            "approved": decision.approved,
            "modifications": decision.modifications,
            "feedback": decision.feedback,
        })

    def clear_pending(self) -> None:
        """Clear pending state."""
        self.pending = False
        self.checkpoint = None
        self.clarification_questions = []
