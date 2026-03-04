# Coding Standards for prompts

- **All LLM prompts MUST be defined in `src/prompts/` package** (split by phase: `hitl.py`, `research.py`, `synthesis.py`; `__init__.py` re-exports all)
- **All LLM prompts MUST be divided into SYSTEM and HUMAN**, i.e. `SOME_NICE_PROMPT_SYSTEM` and `SOME_NICE_PROMPT_HUMAN` following the Attention Priority Hierarchy:
  - SYSTEM: Role, Output format, Rules — authoritative instructions. Input section describes field names/descriptions (not actual values).
  - HUMAN: Input with actual template variables + one-line task reminder.
  - LLMs process system messages with higher authority and attention weight than user messages. The model is trained to treat the system prompt as the most authoritative instruction layer — it "tries hardest to obey" system-level directives. This means your non-negotiable rules (grounding constraints, anti-hallucination rules, output format) are more strongly enforced when placed in the system prompt vs. burying them in a user message.
- **Every content-bearing prompt MUST include `{language}`** to enforce output language
  - Only exceptions: `LANGUAGE_DETECTION_PROMPT` (outputs code) and `REFERENCE_EXTRACTION_PROMPT` (copies verbatim)
- Optimize prompts for small local LLMs (<=32B parameters); be as clear and specific as you can
- Where appropriate, use short Chain-of-Thought (CoT) reasoning techniques to break down complex tasks into smaller, more manageable steps
- Never inline prompt strings in node functions or services
- Use template variables for dynamic content (e.g., `{query}`, `{context}`)
- Group prompts by category (HITL, Research, Quality)

## SYSTEM Prompt Formats

Three formats co-exist in the codebase. Formats A and B are for Qwen models; Format C is for gpt-oss models. Choose based on model family and prompt complexity.

### Format A — XML tags (preferred for synthesis/summary prompts)

Used by the 5 key synthesis and summary prompts:
`SYNTHESIS_PROMPT_ENHANCED_SYSTEM`, `SYNTHESIS_PROMPT_SYSTEM`, `QUERY_ASSESSMENT_PROMPT_SYSTEM`,
`HITL_SUMMARY_PROMPT_SYSTEM`, `TASK_SUMMARY_PROMPT_SYSTEM`.

Structure (order matters — output format 2nd so LLM sees it early):

```
<role>
One or two sentences: what the model IS and what it DOES. End with "You output valid JSON only." (or "plain text only.").
</role>

<output_format>
Return exactly this JSON — no other text before or after:
{"field": "VALUE", ...}

Field definitions:
- field: description, constraints
</output_format>

<constraints>
HARD CONSTRAINTS — never violate:
1. Numbered absolute rules (no invention, no outside knowledge, language enforcement, etc.)
</constraints>

<content_rules>
WRITING RULES — apply in this order:
1. Numbered procedural rules for how to populate each field.
</content_rules>

<input_definitions>
Inputs provided:
- variable_name: What this input contains and how to use it.
</input_definitions>

<example>
Input: ...
Output: {"field": "realistic domain value", ...}
</example>
```

Key design choices for Format A:
- **Output format is 2nd** (after role, before rules): the LLM anchors on the schema before reading the rules.
- **HARD CONSTRAINTS separated** from writing rules: violations of grounding/language/invention rules vs. stylistic rules.
- **Realistic domain example** using actual radiation-protection terminology, not placeholder text.
- **HUMAN counterpart stays lean**: only `### Input` with actual template variables + one reminder line.

### Format B — Markdown sections (simpler prompts)

Used by most other prompts: `TASK_SEARCH_QUERIES_PROMPT_SYSTEM`, `CHUNK_RERANKER_PROMPT_SYSTEM`,
`RERANKER_PRECISION_PROMPT_SYSTEM`, `RERANKER_RECALL_PROMPT_SYSTEM`,
`REFERENCE_EXTRACTION_PROMPT_SYSTEM`, `REFERENCE_DECISION_PROMPT_SYSTEM`,
`INFO_EXTRACTION_PROMPT_SYSTEM`, `QUALITY_CHECK_PROMPT_SYSTEM`, `QUALITY_REMEDIATION_PROMPT_SYSTEM`,
`TODO_GENERATION_PROMPT_SYSTEM`, `RELEVANCE_SCORING_PROMPT_SYSTEM`, and all HITL phase prompts.

Structure:

```
### Role      ← in SYSTEM half (optional for simple prompts)
### Goal      ← in SYSTEM half
### Input     ← description of fields in SYSTEM; actual values in HUMAN
### Rules     ← in SYSTEM half
### Output format  ← in SYSTEM half
```

### Format C — Harmony format (gpt-oss models)

Used by all `*_gpt.py` prompt files (`hitl_gpt.py`, `research_gpt.py`, `synthesis_gpt.py`).
Loaded automatically when `settings.model_family == "gpt-oss"` via `src/prompts/__init__.py`.

Adaptation rules from Format A/B:

| Original pattern | gpt-oss (Harmony) adaptation |
|---|---|
| `<role>...</role>` XML tags | `# Role` top-level markdown header |
| `### Role` / `### Goal` sections | `# Role` / `# Goal` (single `#`) |
| Separate `<constraints>` + `<content_rules>` | Merged into single `# Rules` with flat numbered list |
| `"Return ONLY valid JSON"` | `"Wrap your JSON between <json> and </json> tags"` |
| `/no_think` directive (Qwen3-specific) | Removed entirely |
| Nested bullet sub-lists (a, b, c) | Flatten to single-level numbered items |
| `{language}` placeholder | Unchanged — kept in all content-bearing prompts |

Structure:

```
# Role
One or two sentences.

# Output format
Wrap your JSON output between <json> and </json> tags:
<json>{"field": "VALUE"}</json>

Field definitions:
- field: description

# Rules
1. Flat numbered rule.
2. Another rule.
```

Key differences from Format A/B:
- **Single `#` headers** instead of `###` or XML tags — better parsed by Harmony-tuned models
- **`<json>` wrapper tags** for structured output — OllamaClient extracts content between tags on parse failure
- **No `/no_think`** — gpt-oss models do not support Qwen3's thinking mode toggle
- **Harmony preamble** injected at runtime by `OllamaClient._prepare_system_prompt()` (not in the prompt file itself)

### Dynamic Prompt Routing (PEP 562)

`src/prompts/__init__.py` uses **PEP 562 `__getattr__`** for runtime-dynamic prompt resolution. Both prompt sets are eagerly loaded into dicts at import time; attribute access resolves dynamically based on the current `settings.model_family`:

```python
# At module load: build lookup dicts from both prompt sets
_qwen_prompts = {name: value for ...}    # from hitl.py, research.py, synthesis.py
_gptoss_prompts = {name: value for ...}  # from hitl_gpt.py, research_gpt.py, synthesis_gpt.py

def __getattr__(name: str) -> str:
    from src.config import settings
    prompts = _gptoss_prompts if settings.model_family == "gpt-oss" else _qwen_prompts
    if name in prompts:
        return prompts[name]
    raise AttributeError(...)
```

**Consumer pattern**: Consumers use `from src import prompts` then `prompts.X` (module-level attribute access), not `from src.prompts import X` (which would bind at import time and miss runtime model changes).

All gpt-oss files export **identical constant names** as their Qwen counterparts (48 total).

### HITL Summary output sections (reference)

`HITL_SUMMARY_PROMPT_SYSTEM` produces **plain text** (not JSON) with this fixed structure:

```
PRIMARY INFORMATION
[Direct facts with [source_filename.pdf] citations]

FURTHER INFORMATION
[Background context with citations]

RULES
Recommended practices: [one per line]
Things to avoid: [one per line — HARD CONSTRAINTS for synthesis]

GAPS
[Unanswered questions, one per line]
```

Downstream prompts (`SYNTHESIS_PROMPT_ENHANCED_SYSTEM`, `TASK_SUMMARY_PROMPT_SYSTEM`) read the
`Things to avoid` section as HARD CONSTRAINTS and move matching passages to `irrelevant_findings`.

## Invocation Pattern

All callers use the message-based `OllamaClient` methods (NOT `generate()` / `generate_structured()`).
Prompt constants are accessed via module-level attribute access for dynamic routing:

```python
from src import prompts

# For plain text output:
system_prompt = prompts.SOME_PROMPT_SYSTEM.format(language=lang_label)
human_prompt = prompts.SOME_PROMPT_HUMAN.format(query=query, language=lang_label)
response = client.generate_messages(system_prompt, human_prompt)

# For structured (Pydantic) output:
result = client.generate_structured_messages(system_prompt, human_prompt, MyModel)

# For structured output with language enforcement + retry:
result = client.generate_structured_messages_with_language(
    system_prompt, human_prompt, MyModel, target_language="de",
)
```

These methods wrap the prompts as `SystemMessage` / `HumanMessage` objects from `langchain_core.messages`, ensuring the LLM receives them with proper role separation.

**Important**: Never use `from src.prompts import X` — this binds at import time and misses runtime model changes. Always use `from src import prompts` + `prompts.X`.
