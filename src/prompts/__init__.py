"""Prompt constants for all LLM interactions, split by phase.

Dynamic routing via PEP 562 __getattr__: prompt constants are resolved
at access time based on the current ``settings.model_family``, so
switching the model at runtime takes effect immediately.

Usage in consumers::

    from src import prompts
    system = prompts.SYNTHESIS_PROMPT_ENHANCED_SYSTEM.format(language=lang)

Modules:
    hitl.py / hitl_gpt.py           - Phase 1: All HITL prompts
    research.py / research_gpt.py   - Phase 3: Extraction, references, task queries
    synthesis.py / synthesis_gpt.py - Phase 2 + 4: Todo, synthesis, quality
"""

import re as _re

from src.prompts import hitl as _hitl_qwen
from src.prompts import hitl_gpt as _hitl_gpt
from src.prompts import research as _research_qwen
from src.prompts import research_gpt as _research_gpt
from src.prompts import synthesis as _synthesis_qwen
from src.prompts import synthesis_gpt as _synthesis_gpt


def _collect_constants(module):
    """Collect all UPPER_CASE string constants from a module."""
    return {
        name: getattr(module, name)
        for name in dir(module)
        if _re.match(r"^[A-Z][A-Z_0-9]+$", name) and isinstance(getattr(module, name), str)
    }


# Eagerly build dicts from both prompt sets
_qwen_prompts: dict[str, str] = {}
_gptoss_prompts: dict[str, str] = {}

for _mod in (_hitl_qwen, _research_qwen, _synthesis_qwen):
    _qwen_prompts.update(_collect_constants(_mod))

for _mod in (_hitl_gpt, _research_gpt, _synthesis_gpt):
    _gptoss_prompts.update(_collect_constants(_mod))

# Union of all constant names (both sets export identical names)
__all__ = sorted(set(_qwen_prompts) | set(_gptoss_prompts))


def __getattr__(name: str) -> str:
    """Resolve prompt constant dynamically based on current model family."""
    from src.config import settings

    if settings.model_family == "gpt-oss":
        prompts = _gptoss_prompts
    else:
        # "gemma4" and "qwen" both use the Qwen prompt set;
        # /no_think prefix is stripped at OllamaClient level for gemma4
        prompts = _qwen_prompts

    if name in prompts:
        return prompts[name]

    raise AttributeError(f"module 'src.prompts' has no attribute {name!r}")
