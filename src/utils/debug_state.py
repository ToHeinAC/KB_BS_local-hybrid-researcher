"""Debug state dump utility for phase boundary inspection."""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

# Field grouping for markdown output
_FIELD_GROUPS = {
    "Core": [
        "query", "query_analysis", "phase", "iteration",
    ],
    "HITL State": [
        "hitl_active", "hitl_iteration", "hitl_termination_reason",
        "hitl_conversation_history", "hitl_state", "detected_language",
        "coverage_score", "knowledge_gaps", "iteration_queries",
        "retrieval_dedup_ratios", "query_retrieval",
    ],
    "Graded Context": [
        "query_anchor", "hitl_smry",
        "primary_context", "secondary_context", "tertiary_context",
        "preserved_quotes", "task_summaries",
    ],
    "Research Planning": [
        "research_queries", "additional_context", "todo_list",
    ],
    "Task Execution": [
        "current_task_id", "completed_task_ids", "research_context",
    ],
    "Quality & Report": [
        "quality_assessment", "final_report",
    ],
    "Settings & Meta": [
        "selected_database", "k_results", "messages",
    ],
}

# Fields where we show summary stats instead of full content
_SUMMARY_FIELDS = {"research_context", "final_report", "messages"}


def _format_value(key: str, value: object) -> str:
    """Format a state value for markdown display."""
    if value is None:
        return "*None*"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, set):
        return json.dumps(sorted(value), ensure_ascii=False, indent=2)
    if isinstance(value, str):
        if not value:
            return '*empty string*'
        if len(value) > 500:
            return f"{value[:500]}... *({len(value)} chars total)*"
        return value
    if isinstance(value, (list, dict)):
        # Summary-only fields
        if key == "research_context" and isinstance(value, dict):
            queries = value.get("search_queries", [])
            chunk_count = sum(len(q.get("chunks", [])) for q in queries)
            docs = value.get("metadata", {}).get("documents_referenced", [])
            return f"queries: {len(queries)}, chunks: {chunk_count}, docs: {len(docs)}"
        if key == "final_report" and isinstance(value, dict):
            return f"answer length: {len(value.get('answer', ''))}, findings: {len(value.get('findings', []))}"
        if key == "messages" and isinstance(value, list):
            tail = value[-10:] if len(value) > 10 else value
            return json.dumps(tail, ensure_ascii=False, indent=2)
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        if len(text) > 500:
            return f"{text[:500]}... *({len(text)} chars total)*"
        return text
    return str(value)[:500]


def dump_state_markdown(
    state: dict,
    return_dict: dict,
    filepath: str,
    phase_label: str,
) -> None:
    """Write a formatted markdown snapshot of the merged agent state.

    Merges incoming state with return_dict (partial update from the node),
    groups fields by category, and writes to filepath. Never raises.

    Args:
        state: Full agent state before the node's return.
        return_dict: Partial state update returned by the node.
        filepath: Output file path (e.g. "tests/debugging/state_1hitl.md").
        phase_label: Human-readable label for the dump header.
    """
    if not settings.enable_state_dump:
        return

    try:
        merged = {**state, **return_dict}
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# State Dump: {phase_label}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            "",
        ]

        seen_keys: set[str] = set()

        for group_name, fields in _FIELD_GROUPS.items():
            lines.append(f"## {group_name}")
            lines.append("")
            for key in fields:
                seen_keys.add(key)
                value = merged.get(key)
                formatted = _format_value(key, value)
                # Use code block for multi-line values
                if "\n" in formatted and len(formatted) > 80:
                    lines.append(f"### `{key}`")
                    lines.append("```json")
                    lines.append(formatted)
                    lines.append("```")
                else:
                    lines.append(f"- **{key}**: {formatted}")
            lines.append("")

        # Catch any ungrouped keys
        remaining = sorted(set(merged.keys()) - seen_keys)
        if remaining:
            lines.append("## Other Fields")
            lines.append("")
            for key in remaining:
                formatted = _format_value(key, merged.get(key))
                lines.append(f"- **{key}**: {formatted}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("State dump written to %s", filepath)

    except Exception:
        logger.debug("State dump failed (non-fatal)", exc_info=True)
