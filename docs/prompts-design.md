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

Two formats co-exist in the codebase. Choose based on prompt complexity.

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

All callers use the message-based `OllamaClient` methods (NOT `generate()` / `generate_structured()`):

```python
# For plain text output:
system_prompt = SOME_PROMPT_SYSTEM.format(language=lang_label)
human_prompt = SOME_PROMPT_HUMAN.format(query=query, language=lang_label)
response = client.generate_messages(system_prompt, human_prompt)

# For structured (Pydantic) output:
result = client.generate_structured_messages(system_prompt, human_prompt, MyModel)

# For structured output with language enforcement + retry:
result = client.generate_structured_messages_with_language(
    system_prompt, human_prompt, MyModel, target_language="de",
)
```

These methods wrap the prompts as `SystemMessage` / `HumanMessage` objects from `langchain_core.messages`, ensuring the LLM receives them with proper role separation.
