"""Prompt constants for all LLM interactions, split by phase.

Re-exports everything for backward compatibility:
    from src.prompts import LANGUAGE_DETECTION_PROMPT_SYSTEM  # still works

When OLLAMA_MODEL starts with "gpt-oss", gpt-oss-adapted prompts are
loaded instead of the Qwen3-optimized defaults. The constant names are
identical so consumers do not need any changes.

Modules:
    hitl.py / hitl_gpt.py           - Phase 1: All HITL prompts
    research.py / research_gpt.py   - Phase 3: Extraction, references, task queries
    synthesis.py / synthesis_gpt.py - Phase 2 + 4: Todo, synthesis, quality
"""

from src.config import settings

if settings.model_family == "gpt-oss":
    from src.prompts.hitl_gpt import *  # noqa: F401,F403
    from src.prompts.research_gpt import *  # noqa: F401,F403
    from src.prompts.synthesis_gpt import *  # noqa: F401,F403
else:
    from src.prompts.hitl import *  # noqa: F401,F403
    from src.prompts.research import *  # noqa: F401,F403
    from src.prompts.synthesis import *  # noqa: F401,F403
