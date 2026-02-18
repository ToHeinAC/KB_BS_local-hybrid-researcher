"""Prompt constants for all LLM interactions, split by phase.

Re-exports everything for backward compatibility:
    from src.prompts import LANGUAGE_DETECTION_PROMPT_SYSTEM  # still works

Modules:
    hitl.py      - Phase 1: All HITL prompts (18 constants)
    research.py  - Phase 3: Extraction, references, task queries (12 constants)
    synthesis.py - Phase 2 + 4: Todo, synthesis, quality (14 constants)
"""

from src.prompts.hitl import *  # noqa: F401,F403
from src.prompts.research import *  # noqa: F401,F403
from src.prompts.synthesis import *  # noqa: F401,F403
