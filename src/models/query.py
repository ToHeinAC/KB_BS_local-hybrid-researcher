"""Query and ToDo list models."""

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Extracted analysis of user's research query."""

    original_query: str = Field(description="The original user query")
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Key concepts extracted from the query",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities (laws, regulations, documents)",
    )
    scope: str = Field(
        default="",
        description="Scope of the research (e.g., regulatory, technical)",
    )
    assumed_context: list[str] = Field(
        default_factory=list,
        description="Assumed context based on query",
    )
    clarification_needed: bool = Field(
        default=False,
        description="Whether clarification is needed from user",
    )
    hitl_refinements: list[str] = Field(
        default_factory=list,
        description="Refinements from HITL clarification loop",
    )
    detected_language: str = Field(
        default="de",
        description="Detected language of the query",
    )


class ToDoItem(BaseModel):
    """A single research task."""

    id: int = Field(description="Unique task identifier")
    task: str = Field(description="Description of the task")
    context: str = Field(
        default="",
        description="Context or rationale for this task",
    )
    completed: bool = Field(
        default=False,
        description="Whether the task is completed",
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description="Sub-tasks if any",
    )


class ToDoList(BaseModel):
    """Research task tracker."""

    items: list[ToDoItem] = Field(
        default_factory=list,
        description="List of research tasks",
    )
    max_items: int = Field(
        default=15,
        description="Maximum allowed tasks",
    )
    current_item_id: int | None = Field(
        default=None,
        description="ID of currently executing task",
    )

    def get_pending_tasks(self) -> list[ToDoItem]:
        """Return tasks that are not completed."""
        return [item for item in self.items if not item.completed]

    def get_next_task(self) -> ToDoItem | None:
        """Return the next pending task."""
        pending = self.get_pending_tasks()
        return pending[0] if pending else None

    def mark_completed(self, task_id: int) -> None:
        """Mark a task as completed."""
        for item in self.items:
            if item.id == task_id:
                item.completed = True
                break

    def add_task(self, task: str, context: str = "") -> ToDoItem | None:
        """Add a new task if under limit."""
        if len(self.items) >= self.max_items:
            return None
        new_id = max((item.id for item in self.items), default=0) + 1
        new_item = ToDoItem(id=new_id, task=task, context=context)
        self.items.append(new_item)
        return new_item


class QueryContext(BaseModel):
    """Accumulated context for query generation from original query + HITL."""

    original_query: str = Field(description="The original user query")
    hitl_conversation: list[str] = Field(
        default_factory=list,
        description="History of HITL interactions",
    )
    user_feedback_analysis: str | None = Field(
        default=None,
        description="Analysis of user feedback",
    )
    detected_language: str = Field(
        default="de",
        description="Detected language",
    )


class QuerySet(BaseModel):
    """Queries generated for a todo-item."""

    todo_item_id: str = Field(description="ID of the associated todo item")
    vector_queries: list[str] = Field(
        default_factory=list,
        description="3-5 queries for vector search",
    )
    doc_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for document search (3x multiplier)",
    )
    web_queries: list[str] = Field(
        default_factory=list,
        description="Queries for web search if enabled",
    )
    iteration: int = Field(
        default=1,
        description="Current iteration number",
    )
    generated_from_critique: bool = Field(
        default=False,
        description="Whether generated from gap analysis",
    )
