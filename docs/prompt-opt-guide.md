# Prompt Optimization Guideline for Local LLMs (≤32B Parameters)

**Target models:** Qwen3:14b (primary), any model ≤32B parameters running via Ollama  
**Purpose:** Hand this guideline + any prompt to Claude Opus 4.6 and request optimization.

***

## 1. Core Constraints of ≤32B Models

Understanding *why* these models need different prompts is essential before optimizing. Smaller LLMs (≤32B parameters) differ from frontier models in several critical ways:[^1][^2]

- **Weaker instruction-following**: They struggle with implicit instructions, nested conditions, and multi-layered rules. Every rule must be stated explicitly and unambiguously.[^3][^2]
- **Smaller effective attention window**: Even with 32K+ context support, attention degrades on long prompts. Critical instructions at the end of a long prompt may be ignored. Place the most important rules first.[^2]
- **Prone to format drift**: Without concrete examples, small models drift from requested output formats (e.g., outputting "4 out of 5 stars" instead of just `4`).[^1][^2]
- **Sensitive to prompt structure**: Markdown headers (`###`) inside system prompts can confuse smaller models into treating them as output formatting rather than instruction delimiters. Use XML tags or plain-text delimiters instead.[^2]
- **Limited abstraction handling**: Terms like "tiered evidence" or "thematic grouping" are too abstract. Operationalize every concept with a concrete definition or example.[^3]

***

## 2. The 12 Optimization Principles

When optimizing any prompt for ≤32B models, apply these principles systematically:

### Principle 1: Use XML Tags as Section Delimiters (Not Markdown Headers)

**Why:** Markdown headers (`###`, `####`) inside system prompts cause ambiguity — the model may interpret them as output formatting instructions. XML tags create unambiguous structure that small LLMs parse reliably.[^2]

```
BAD:  ### Role
      ### Rules  
      ### Output Format

GOOD: <role> ... </role>
      <rules> ... </rules>
      <output_format> ... </output_format>
```

### Principle 2: Front-Load Critical Instructions

**Why:** Due to attention degradation in long contexts, the most important behavioral constraints must appear first. The "primacy effect" is stronger in smaller models — they weight the beginning of the prompt more heavily.[^1][^2]

**Action:** Place role definition → output format → hard constraints → content rules → examples, in that priority order.

### Principle 3: One Instruction Per Line, Numbered When Ordered

**Why:** Smaller models handle flat, numbered lists far better than nested bullets or paragraph-style rules. Each instruction should be a single, self-contained directive.[^4][^1]

```
BAD:  - Your highest priority is to follow original_query and hitl_findings.
        - Treat all things to avoid or pitfalls in hitl_findings as hard constraints. 
        - Do not include any content that matches a things to avoid or pitfalls.

GOOD: 1. Follow original_query and hitl_findings as your top priority.
      2. Treat every item in hitl_findings "things_to_avoid" as a hard constraint — never include that content.
      3. If things_to_avoid conflicts with other source material, things_to_avoid always wins.
```

### Principle 4: Eliminate Vague Qualifiers — Operationalize Everything

**Why:** Words like "extensive," "thorough," "detailed," and "comprehensive" are meaningless to a small LLM — it has no calibration for what "extensive" means. Replace with measurable targets.[^3][^1]

```
BAD:  "Generate a thorough, structured deep report"
GOOD: "Generate a report with 5-15 sections, each containing 2-5 bullet points with citations."

BAD:  "Include exact figures"
GOOD: "Copy numbers, percentages, thresholds, and limits exactly as written in the source. Do not round or paraphrase numerical values."
```

### Principle 5: Provide a Concrete Output Example (Few-Shot)

**Why:** Few-shot examples are the single most impactful technique for small LLMs. A concrete example of the expected output format eliminates ambiguity that instructions alone cannot resolve. For structured output (JSON, YAML), always provide a complete, realistic example — not a schema with placeholders.[^1][^3][^2]

```
BAD:  {"summary": "<your report>", "key_findings": ["<one finding>"]}

GOOD: {"summary": "#### Direct Answer\nThe maximum permissible dose...[Report.pdf, Page 12]\n\n#### Regulatory Framework\n- § 80 StrlSchV defines... [Guidelines.pdf, Page 5]\n- The threshold is 1 mSv/year [Report.pdf, Page 12]", "key_findings": ["The annual dose limit is 1 mSv for public exposure [Report.pdf, Page 12]", "Monitoring is required quarterly per § 80 StrlSchV [Guidelines.pdf, Page 5]"], "query_coverage": 75, "remaining_gaps": ["No source addressed occupational dose limits"]}
```

### Principle 6: Separate "MUST DO" from "MUST NOT DO"

**Why:** Small models handle positive instructions ("do X") and negative constraints ("never do Y") better when they are in separate, clearly labeled blocks rather than interleaved.[^2][^1]

```
<must_do>
1. Cite every claim as [Document.pdf, Page N].
2. Copy numbers exactly from sources.
3. Use direct quotes for legal text.
</must_do>

<must_not>
1. Never invent values, numbers, or citations.
2. Never use external knowledge — only provided task summaries.
3. Never add preamble or explanation outside the JSON.
</must_not>
```

### Principle 7: Define Every Domain-Specific Term

**Why:** Frontier models can infer meaning from context; small models cannot. If you use a term like "tiered evidence," "HITL context," or "query_coverage," define it explicitly.[^3]

```
BAD:  "task_summaries: per-task structured summaries with tiered evidence"

GOOD: "task_summaries: A list of summaries, one per research task. Each summary contains:
       - task_query: what was researched
       - findings: list of facts found
       - sources: list of [Document.pdf, Page N] references
       - confidence: 'high', 'medium', or 'low'"
```

### Principle 8: Use Consistent Delimiter Patterns

**Why:** Mixing delimiters (markdown headers + bullets + numbered lists + XML + code fences) forces the model to context-switch between parsing modes. Pick one primary delimiter system and stick with it.[^2]

**Recommended hierarchy for ≤32B:**
- XML tags for major sections: `<role>`, `<rules>`, `<output_format>`, `<example>`
- Numbered lists for ordered rules within sections
- Bullet points (`-`) only for unordered items within a single rule

### Principle 9: Keep Total Prompt Under 1500 Tokens for Critical System Prompts

**Why:** While Qwen3:14b supports 32K+ context, system prompt effectiveness degrades significantly beyond ~1500 tokens for instruction-following. The user message (with actual data) should consume the bulk of the context window.[^5][^1]

**Action:** If a system prompt exceeds 1500 tokens, split it: move examples and detailed rules to a separate "reference" section in the user message, keeping only role + core constraints + output format in the system prompt.

### Principle 10: Avoid Contradictory Instructions

**Why:** Large models can resolve contradictions via reasoning; small models cannot. Audit every prompt for internal conflicts.[^1]

**Common contradictions to check:**
- "Output raw JSON only" vs. "Use markdown formatting inside the JSON" → Clarify: "The JSON values may contain markdown. The top-level output must be raw JSON with no wrapping."
- "Preserve original wording" vs. "Write in {language} only" → Clarify: "Preserve original wording for quotes. Write all non-quoted text in {language}."
- "Be thorough" vs. "Be concise" → Pick one and operationalize it.

### Principle 11: For Qwen3 Specifically — Control Thinking Mode

**Why:** Qwen3 has a hybrid thinking/non-thinking mode. For structured output tasks (JSON generation), non-thinking mode (`/no_think`) often produces cleaner, more format-compliant output. For complex reasoning tasks, thinking mode is better.[^6][^7][^5]

**Action:**
- For synthesis/report generation: Use thinking mode (`/think`) with `Temperature=0.6, TopP=0.95, TopK=20`
- For structured extraction/classification: Use non-thinking mode (`/no_think`) with `Temperature=0.7, TopP=0.8, TopK=20`
- Never use greedy decoding (temperature=0) with Qwen3 — it causes repetition loops[^5]

### Principle 12: Test with Adversarial Inputs

**Why:** Small models fail silently. After optimization, test each prompt with: (a) minimal input (1 task summary), (b) conflicting sources, (c) missing fields, (d) very long input near context limit.

***

## 3. Optimization Checklist

Use this checklist when reviewing any prompt before handing it to Claude Opus 4.6 for optimization:

```
□ Are section delimiters XML tags (not markdown headers)?
□ Are the top 3 most critical instructions in the first 200 tokens?
□ Is every vague qualifier replaced with a measurable target?
□ Is there at least one complete few-shot output example?
□ Are positive rules ("do") separated from negative rules ("don't")?
□ Is every domain term defined inline?
□ Is the delimiter style consistent throughout?
□ Is the system prompt under ~1500 tokens?
□ Are there zero internal contradictions?
□ Is the Qwen3 thinking mode specified for this task type?
□ Does the output format example match EXACTLY what downstream code expects?
□ Has it been tested with edge cases (empty input, conflicting sources, long input)?
```

***

## 4. Instruction Template for Claude Opus 4.6

Copy-paste this when asking Claude Opus 4.6 to optimize a prompt:

```
I need you to optimize the following system prompt for a local LLM with ≤32B parameters 
(specifically Qwen3:14b running via Ollama).

## Optimization Guideline
[PASTE THE FULL GUIDELINE FROM SECTIONS 1-3 ABOVE]

## Prompt to Optimize
[PASTE THE PROMPT HERE]

## Instructions
1. Analyze the prompt against each of the 12 principles and the checklist.
2. List every violation found, with the specific principle number.
3. Produce the fully rewritten, optimized prompt.
4. Explain each major change and which principle it addresses.
5. Ensure the optimized prompt is under 1500 system-prompt tokens 
   (move overflow to a user-message reference section if needed).
6. Include a complete, realistic few-shot output example.
7. Flag any remaining trade-offs or areas where testing is needed.
```

***

## 5. Applied Example: Optimizing SYNTHESIS_PROMPT_ENHANCED_SYSTEM

### 5.1 Violations Found in the Original Prompt

| # | Principle Violated | Specific Issue |
|---|---|---|
| 1 | P1: XML Tags | Uses `### Role`, `### Goal`, `### Rules` — markdown headers inside system prompt |
| 2 | P2: Front-Loading | Output format is at the very end; role is first (good) but format should be second |
| 3 | P3: One Per Line | Nested bullets under "Your highest priority" create 3-level indentation |
| 4 | P4: No Vague Qualifiers | "thorough," "extensive," "detailed," "deep" — all undefined |
| 5 | P5: Few-Shot Example | Output format shows a schema with placeholders, not a realistic example |
| 6 | P6: Separate DO/DON'T | Positive and negative rules are interleaved in one block |
| 7 | P7: Define Terms | "tiered evidence," "HITL context," "hitl_smry," "query_coverage" undefined |
| 8 | P8: Consistent Delimiters | Mixes markdown headers, numbered lists, bullets, code fences |
| 9 | P9: Token Budget | Prompt is ~600 tokens as system prompt — acceptable, but wastes tokens on vague terms |
| 10 | P10: Contradictions | "output raw JSON only" vs. "markdown formatting" inside JSON; "preserve original wording" vs. "write in {language} only" |
| 11 | P11: Qwen3 Mode | No thinking mode guidance |
| 12 | P12: Adversarial Testing | No guidance for empty/minimal input handling |

### 5.2 Prompt optimization example

#### 5.2.1 Original Prompt
```python
SYNTHESIS_PROMPT_ENHANCED_SYSTEM = """
### Role
You are an expert report writer producing extensive, detailed deep reports from pre-digested task summaries.

### Goal
Generate a thorough, structured deep report that answers the original query using ONLY the provided task summaries and HITL context.

### Input
- original_query: the user's original query
- hitl_smry: citation-aware HITL summary
- task_summaries: per-task structured summaries with tiered evidence

### Rules
REPORT STRUCTURE — the summary field must be a markdown-formatted deep report:
1. Begin with a direct answer to the query (1-2 sentences).
2. Then provide detailed sections covering every relevant aspect found across all task summaries.
3. Use markdown headings (####), bullet points, and numbered lists for structure.
4. Group related findings thematically — do not just list task summaries sequentially.
5. End with a brief assessment of completeness and any open questions.

CONTENT RULES
- Your highest priority is to follow original_query and hitl_findings.
  - Treat all things to avoid or pitfalls in hitl_findings as hard constraints. 
  - Do not include any content that matches a things to avoid or pitfalls in your report item.
  - If things to avoid or pitfalls conflicts with other information, obey the things to avoid or pitfalls.
  - Example: In the case the original_query is about climate and hitl_findings put "current weather" in things to avoid or pitfalls, then do not include current weather in your report.
- Preserve original wording from source material when possible.
- Include exact levels, figures, numbers, statistics, thresholds, and limits as they appear in the sources.
- Reference specific sections, paragraphs, articles (e.g., "§ 80 StrlSchV", "Anlage 4 Teil B").
- Use direct quotes (in quotation marks) for key definitions, legal text, or critical formulations.
- Cite every claim as [Document.pdf, Page N] — never omit the source.
- Include verbatim quotes from task summaries where they support a finding.
- State explicitly when information is insufficient, contradictory, or missing.
- Use ONLY information from the provided task summaries — no external knowledge.
- Write in {language} only — no mixing.
- Do NOT invent values, numbers, or citations.
- Do NOT add preamble, explanation, or markdown fences — output raw JSON only.

COVERAGE AND GAPS
- Estimate query_coverage (0-100): how completely the original_query is answered.
- Collect remaining_gaps from all task summaries — what is still missing or contradictory.

### Output format
Return ONLY this JSON, no other text:
```json
{{"summary": "<your extensive structured deep report in {language} with markdown formatting and [Document.pdf, Page N] citations>",
  "key_findings": ["<one key finding with [Document.pdf, Page N] citation>"],
  "query_coverage": 0,
  "remaining_gaps": ["<one gap or uncertainty>"]}}
```
#### 5.2.2 System Prompt (Optimized)
```python
SYNTHESIS_PROMPT_ENHANCED_SYSTEM = """
<role>
You are a report-writing assistant. You produce structured reports from provided task summaries. You output valid JSON only.
</role>

<output_format>
Return exactly this JSON structure — no other text before or after it:

{{"summary": "MARKDOWN_REPORT_HERE", "key_findings": ["FINDING_1", "FINDING_2"], "query_coverage": INTEGER_0_TO_100, "remaining_gaps": ["GAP_1", "GAP_2"]}}

Field definitions:
- summary: A markdown-formatted report answering the original query. Use #### headings, bullet points, and [Document.pdf, Page N] citations. Aim for 5-15 sections, 2-5 bullets each.
- key_findings: 3-10 most important facts, each with a [Document.pdf, Page N] citation.
- query_coverage: Integer 0-100 estimating how completely the query is answered. 100 = fully answered, 0 = no relevant information found.
- remaining_gaps: List of specific questions or topics that the sources did not answer or where sources contradicted each other.
</output_format>

straints>
HARD CONSTRAINTS (never violate):
1. Use ONLY information from the provided task_summaries and hitl_context. Never use outside knowledge.
2. Every item in hitl_context → things_to_avoid is forbidden. Never include that content in your report, even if sources mention it.
3. If things_to_avoid conflicts with source material, things_to_avoid always wins.
4. Never invent numbers, values, statistics, or citations.
5. Never add text outside the JSON structure — no preamble, no explanation, no code fences.
6. Write all non-quoted text in {language}. Do not mix languages.
</constraints>

tent_rules>
REPORT WRITING RULES (follow in order of priority):
1. Start the summary with a 1-2 sentence direct answer to original_query.
2. Group findings by theme, not by task summary order.
3. Copy numbers, percentages, thresholds, and limits exactly as they appear in sources. Do not round or paraphrase.
4. Use direct quotes (in quotation marks) for legal text, definitions, or critical formulations.
5. Cite every factual claim as [Document.pdf, Page N]. Never omit citations.
6. Reference specific legal sections (e.g., "§ 80 StrlSchV", "Anlage 4 Teil B") when sources mention them.
7. When sources are insufficient or contradictory, state this explicitly in the report and add the topic to remaining_gaps.
</content_rules>

<input_definitions>
You will receive these inputs:
- original_query: The user's question that the report must answer.
- hitl_context: Human-in-the-loop guidance containing:
  - focus_areas: Topics to emphasize.
  - things_to_avoid: Topics that MUST NOT appear in the report (hard constraint).
  - additional_notes: Extra context or clarifications.
- task_summaries: A list of research results. Each contains:
  - task_query: What was researched.
  - findings: Facts discovered, with source references.
  - sources: List of [Document.pdf, Page N] references.
  - confidence: "high", "medium", or "low".
</input_definitions>

<example>
INPUT:
original_query: "What are the dose limits for radiation workers under German law?"
hitl_context: {{"focus_areas": ["occupational dose limits", "StrlSchV"], "things_to_avoid": ["medical exposure limits"], "additional_notes": ""}}
task_summaries: [
  {{"task_query": "occupational dose limits StrlSchV", "findings": ["The annual effective dose limit for occupational exposure is 20 mSv per calendar year (§ 78 Abs. 1 StrlSchV)", "The organ dose limit for the eye lens is 20 mSv per calendar year (§ 78 Abs. 2 StrlSchV)"], "sources": ["StrlSchV_2018.pdf, Page 45", "StrlSchV_2018.pdf, Page 46"], "confidence": "high"}},
  {{"task_query": "dose monitoring requirements", "findings": ["Personal dosimetry is required when annual dose may exceed 1 mSv (§ 66 StrlSchV)"], "sources": ["StrlSchV_2018.pdf, Page 38"], "confidence": "high"}}
]

OUTPUT:
{{"summary": "#### Direct Answer\\nThe annual effective dose limit for occupationally exposed workers under German law is 20 mSv per calendar year, as defined in § 78 Abs. 1 StrlSchV [StrlSchV_2018.pdf, Page 45].\\n\\n#### Occupational Dose Limits\\n- Annual effective dose: 20 mSv per calendar year (§ 78 Abs. 1 StrlSchV) [StrlSchV_2018.pdf, Page 45]\\n- Organ dose limit for eye lens: 20 mSv per calendar year (§ 78 Abs. 2 StrlSchV) [StrlSchV_2018.pdf, Page 46]\\n\\n#### Monitoring Requirements\\n- Personal dosimetry is mandatory when the annual dose may exceed 1 mSv (§ 66 StrlSchV) [StrlSchV_2018.pdf, Page 38]\\n\\n#### Completeness Assessment\\nThe core occupational dose limits are well-covered. Skin dose limits and extremity dose limits were not found in the provided sources.", "key_findings": ["Annual effective dose limit: 20 mSv/year (§ 78 Abs. 1 StrlSchV) [StrlSchV_2018.pdf, Page 45]", "Eye lens dose limit: 20 mSv/year (§ 78 Abs. 2 StrlSchV) [StrlSchV_2018.pdf, Page 46]", "Personal dosimetry required above 1 mSv/year (§ 66 StrlSchV) [StrlSchV_2018.pdf, Page 38]"], "query_coverage": 65, "remaining_gaps": ["Skin dose limits not found in sources", "Extremity dose limits not addressed", "Dose limits for pregnant workers not covered"]}}
</example>
"""
```

### 5.3 Change Log — What Was Changed and Why

| Change | Principle | Rationale |
|---|---|---|
| Replaced all `### Headers` with `<xml_tags>` | P1 | XML tags are unambiguous section delimiters that small LLMs parse reliably without confusing them with output format instructions |
| Moved `<output_format>` to second position (after role) | P2 | Output format is the most critical behavioral constraint after role identity — front-loading it ensures the model sees it during peak attention |
| Flattened all nested bullets into numbered single-level rules | P3 | Eliminates 3-level nesting that small LLMs struggle to track; each rule is self-contained |
| Replaced "thorough, extensive, detailed" with "5-15 sections, 2-5 bullets each" | P4 | Gives the model a concrete, measurable target instead of subjective qualifiers |
| Added complete realistic example with radiation protection domain content | P5 | The single most impactful change — the example shows exact formatting, citation style, markdown-in-JSON structure, and realistic content. Small LLMs pattern-match from examples far more reliably than from abstract instructions |
| Split rules into `straints>` (hard MUST NOT) and `tent_rules>` (DO rules) | P6 | Clear separation prevents the model from confusing positive instructions with negative constraints |
| Added `<input_definitions>` with field-by-field breakdown | P7 | Defines "hitl_context," "things_to_avoid," "task_summaries," "confidence" — no undefined terms remain |
| Consistent use of XML tags (sections) + numbered lists (rules) + bullets (only for unordered items) | P8 | Single delimiter system reduces parsing ambiguity |
| System prompt is ~800 tokens (well under 1500 limit) | P9 | Leaves ample context window for the actual task summaries and user data |
| Resolved "raw JSON only" vs. "markdown formatting" contradiction by clarifying: "markdown inside the summary field value, JSON as the top-level output" | P10 | Eliminates the contradiction that would cause a small model to either strip markdown or wrap JSON in markdown |
| Example uses radiation protection content relevant to the actual use case | P5, P7 | Domain-relevant examples prime the model for the actual domain, reducing off-topic drift |
| Added explicit handling for missing/contradictory sources | P12 | "When sources are insufficient or contradictory, state this explicitly" — guides behavior on adversarial inputs |

### 5.4 Qwen3:14b Runtime Recommendations

For this specific synthesis prompt when running on Qwen3:14b via Ollama:[^6][^5]

- **Mode:** Use thinking mode (`/think`) — report synthesis benefits from step-by-step reasoning before output generation
- **Sampling:** `Temperature=0.6`, `TopP=0.95`, `TopK=20`, `MinP=0`
- **Max output tokens:** Set to at least 8192 (reports can be long)
- **Never use greedy decoding** (`temperature=0`) — causes repetition loops in Qwen3
- **Quantization:** Q4_K_M provides the best quality/speed balance for 14B on 24GB VRAM[^8]
- **Post-processing:** Always validate the JSON output with a parser. Small models occasionally produce trailing text or incomplete JSON. Implement a repair step (strip text outside the outermost `{}`).

***

## 6. Additional Patterns for Agentic Prompts

Beyond the synthesis prompt, agentic researcher systems typically have several other prompt types. Here are pattern-specific tips:

### Planning/Decomposition Prompts
- Use numbered step format: "Step 1: ... Step 2: ..."
- Provide 2-3 example decompositions of different query types
- Limit output to a fixed structure (e.g., JSON array of task objects)
- Keep task descriptions short — small models generate better plans when each task is ≤20 words

### Tool-Calling Prompts
- Qwen3 uses Hermes-style tool format natively. Use `tokenizer.apply_chat_template()` with tool definitions[^9]
- For Ollama, define tools in the API call's `tools` parameter, not in the system prompt text
- Keep tool descriptions under 50 words each — long descriptions confuse Qwen3:14b[^10]
- Test with edge cases: no tool needed, multiple tools needed, tool returns error

### Summarization/Extraction Prompts
- Always provide the exact output schema + one complete example
- For extraction: use non-thinking mode (`/no_think`) for faster, cleaner structured output
- For summarization: use thinking mode (`/think`) for better reasoning
- Cap output length explicitly: "Summarize in 3-5 sentences" not "Summarize briefly"

### Routing/Classification Prompts
- Use non-thinking mode — classification is fast pattern matching, not deep reasoning
- Enumerate ALL valid output values explicitly: "Output exactly one of: 'search', 'summarize', 'answer', 'clarify'"
- Provide one example per category
- Small models are biased toward the first/last option — randomize example order during testing

***

## 7. Anti-Patterns to Flag During Optimization

When reviewing any prompt for ≤32B optimization, flag and fix these common anti-patterns:

| Anti-Pattern | Fix |
|---|---|
| "Be creative / Be thorough / Be detailed" | Replace with measurable targets |
| Nested bullet lists (3+ levels) | Flatten to single-level numbered lists |
| Multiple output formats mentioned ("JSON or markdown") | Pick exactly one format |
| Instructions referencing other instructions ("as mentioned above") | Make each instruction self-contained |
| Placeholders in output examples (`<your text here>`) | Use realistic, domain-specific content |
| Code fences inside prompts (` ```json ``` `) | Remove code fences — show raw format |
| Implicit type expectations ("a number") | Explicit: "an integer between 0 and 100" |
| Long conditional chains ("if X then Y, unless Z, but if W...") | Break into separate numbered rules |
| Mixing languages in instructions | Use one language throughout |
| Using model-specific features without checking support | Test each feature (tool calling, JSON mode) with your specific Ollama model version |

---

## References

1. [Prompt Engineering for Smaller LLMs: Tips for Developers](https://fabwebstudio.com/blog/prompt-engineering-for-smaller-ll-ms-tips-for-developers) - Master prompt engineering techniques for smaller language models to achieve accurate results in on-d...

2. [Prompt engineering for LLMs: Proven techniques to ...](https://superlinear.eu/insights/articles/prompt-engineering-for-llms-techniques-to-improve-quality-optimize-cost-reduce-latency) - Master prompt engineering to improve LLM outputs. Learn structured techniques like XML formatting, f...

3. [[PDF] Prompt Optimization with Expert Priors for Small and Medium-sized ...](https://aclanthology.org/2025.knowledgenlp-1.25.pdf)

4. [Prompting | How-to guides](https://www.llama.com/docs/how-to-guides/prompting/) - Prompt engineering is a technique used in natural language processing (NLP) to improve the performan...

5. [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) - Best Practices · Sampling Parameters: · Adequate Output Length: We recommend using an output length ...

6. [Qwen3 - How to Run & Fine-tune](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) - You can add /think and /no_think to user prompts or system messages to switch the model's thinking m...

7. [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/) - To define the available tools, you can use the MCP configuration file, use the integrated tool of Qw...

8. [Local LLM Deployment on 24GB GPUs: Models & ...](https://intuitionlabs.ai/articles/local-llm-deployment-24gb-gpu-optimization) - This report details deploying LLMs on 24GB GPUs, covering model architectures, VRAM needs, and optim...

9. [Function Calling and Tool Use | QwenLM/Qwen3 | DeepWiki](https://deepwiki.com/QwenLM/Qwen3/3.3-function-calling-and-tool-use) - This document covers the implementation and usage of function calling capabilities in Qwen3 models, ...

10. [Qwen3 not Using Tools in Complex Prompts Unlike QwQ ...](https://huggingface.co/Qwen/Qwen3-235B-A22B/discussions/20) - I previously used QwQ-32B via Qwen-Agent and everything ran smoothly. However, when I use the same p...