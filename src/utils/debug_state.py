"""Debug state dump utility for phase boundary inspection."""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.agents.state import create_initial_state
from src.config import settings

logger = logging.getLogger(__name__)


def _format_value(value: object) -> str:
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
            return "*empty string*"
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


def dump_state_markdown(
    state: dict,
    return_dict: dict,
    filepath: str,
    phase_label: str,
) -> None:
    """Write a flat key→value markdown snapshot of the merged agent state.

    Merges incoming state with return_dict (partial update from the node),
    sorts keys alphabetically, and writes each as a ## heading with code block.

    Args:
        state: Full agent state before the node's return.
        return_dict: Partial state update returned by the node.
        filepath: Output file path (e.g. "tests/debugging/state_1hitl.md").
        phase_label: Human-readable label for the dump header.
    """
    if not settings.enable_state_dump:
        return

    try:
        # Build a complete baseline so every AgentState field appears even if
        # no node has written it yet (AgentState is TypedDict, total=False).
        defaults = dict(create_initial_state(""))
        merged = {**defaults, **state, **return_dict}  # live values win over defaults
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# State Dump: {phase_label}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            "",
        ]

        for key in sorted(merged.keys()):
            formatted = _format_value(merged[key])
            lines.append(f"## {key}")
            lines.append("```")
            lines.append(formatted)
            lines.append("```")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("State dump written to %s", filepath)

    except Exception:
        logger.debug("State dump failed (non-fatal)", exc_info=True)
