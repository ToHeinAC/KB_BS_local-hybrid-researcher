# Coding Standards for prompts

- **All LLM prompts MUST be defined in `src/prompts.py`**
- **All LLM prompts MUST be divided into SYSTEM and HUMAN**, i.e. `SOME_NICE_PROMPT_SYSTEM` and `SOME_NICE_PROMPT_HUMAN` following the Attention Priority Hierarchy:
  - SYSTEM: Role, Goal, Rules, Output format — authoritative instructions. Input section describes field names/descriptions (not actual values).
  - HUMAN: Input with actual template variables + one-line task reminder.
  - LLMs process system messages with higher authority and attention weight than user messages. The model is trained to treat the system prompt as the most authoritative instruction layer — it "tries hardest to obey" system-level directives. This means your non-negotiable rules (grounding constraints, anti-hallucination rules, output format) are more strongly enforced when placed in the system prompt vs. burying them in a user message.
- **All LLM prompts MUST follow a strict 5-section format**: `### Role`, `### Goal`, `### Input`, `### Rules`, `### Output format`
  - `### Role` and `### Goal` go in the SYSTEM half
  - `### Input` with actual template values goes in the HUMAN half
  - `### Rules` and `### Output format` go in the SYSTEM half
- **Every content-bearing prompt MUST include `{language}`** to enforce output language
  - Only exceptions: `LANGUAGE_DETECTION_PROMPT` (outputs code) and `REFERENCE_EXTRACTION_PROMPT` (copies verbatim)
- Optimize prompts for small local LLMs (<=32B parameters); be as clear and specific as you can
- Where appropriate, use short Chain-of-Thought (CoT) reasoning techniques to break down complex tasks into smaller, more manageable steps
- Never inline prompt strings in node functions or services
- Use template variables for dynamic content (e.g., `{query}`, `{context}`)
- Group prompts by category (HITL, Research, Quality)

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
