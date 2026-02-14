# Rabbithole Deep Search: Implementation Analysis

## 1. Functional Overview

The **Rabbithole Deep Search** is the signature iterative retrieval mechanism of the Local Hybrid Researcher. It is designed to navigate inter-document relationships by following citations and references across the knowledge base, simulating how a human researcher would "dig deeper" into a topic.

### 1.1 The Process Flow
1.  **Initial Retrieval**: Performed during the `execute_task` node. It uses the `TASK_SEARCH_QUERIES_PROMPT` to generate multiple targeted queries from the task description, ensuring high recall from the vector database.
2.  **Extraction & Detection**: Detailed in Section 1.2 below.
3.  **Reference Following**: Detected references are evaluated by the Agentic Gate (Section 1.2C) and resolved via scoped vector search with context windowing.
4.  **Task Summarization**: Findings are synthesized per-task with tiered prioritization (Section 1.2D).

#### Prompt Architecture
All prompts in the system follow a strict 4-section format optimized for local LLMs (<=20B parameters):
1. **Task**: One-sentence imperative.
2. **Input**: Enumerated variables.
3. **Rules**: Numbered constraints (e.g., language enforcement, JSON output).
4. **Output format**: Exact JSON or text template.

### 1.2 Extraction & Detection Deep-Dive
The system employs a multi-step LLM pipeline for each retrieved chunk to ensure maximum information density and structural awareness.

#### A. Information Extraction
The first step is moving from raw chunks to query-relevant kernels.
- **Prompt**: `INFO_EXTRACTION_WITH_QUOTES_PROMPT` (or `INFO_EXTRACTION_PROMPT` as fallback).
- **Function**: Extracts relevant passages and identifies **Preserved Quotes** (verbatim excerpts for legal thresholds or technical specs).
- **Logic**: Condenses the chunk while preserving exact units (e.g., `mSv`, `µGy/h`) and official titles.

#### B. Reference Detection (The "Sensor")
Once filtered for relevance, the system looks for "exits" to other documents.
- **Prompt**: `REFERENCE_EXTRACTION_PROMPT`.
- **Function**: Scans text for specific patterns:
    - `legal_section`: e.g., "§ 12 des StrlSchG"
    - `academic_shortform`: e.g., "[Mueller2022]"
    - `document_mention`: e.g., "ICRP Publication 103"
- **Hybrid Logic**: The system combines LLM detection with high-precision Regex patterns to maximize recall without losing accuracy.

#### C. Agentic Gating (The "Gatekeeper")
Not every reference is worth the token cost of follow-up retrieval. 
- **Prompt**: `REFERENCE_DECISION_PROMPT`.
- **Input**: The detected reference, its source document, the **Query Anchor** (original intent), and a focused **surrounding_context** window.
- **Logic**: The LLM evaluates the reference within a tuned context window (e.g., +/- 400 characters) extracted from the query-relevant info.
- **Optimization**: Windowing ensures the small LLM (Ollama) stays focused on the citation and significantly reduces the token budget per research iteration.
- **Output**: A boolean `follow` decision and a `reason`.

#### D. Task Summarization
After processing all direct and followed chunks for a task, the system synthesizes them.
- **Prompt**: `TASK_SUMMARY_PROMPT`.
- **Logic**: Groups findings by **Tier**, integrates the **Preserved Quotes**, and performs an initial **Drift Analysis** (identifying irrelevant or "keyword-only" findings).

---


## 2. Strengths (Pros)

| Strength | Impact |
| :--- | :--- |
| **Deep Contextual Insight** | Uncovers hidden connections that standard flat RAG (single-pass search) would miss entirely. |
| **Context-Aware Gating** | The **Agentic Gate** uses surrounding text to decide if a reference is actually relevant, significantly reducing noise from tangential citations. |
| **Legal/Technical Precision** | Scoped search ensures citations like "§ 78 StrlSchG" lead exactly to the correct regulation context. |
| **Tiered Prioritization** | Graded context ensures the LLM synthesizer knows which evidence is direct (Primary) and which is supporting (Secondary/Tertiary). |
| **Loop Prevention** | Robust safeguards (visited_refs, depth limits, token budget, convergence detection) prevent infinite loops. |

---

## 3. Weaknesses & Risks

| Weakness | Impact |
| :--- | :--- |
| **High Latency** | Sequential LLM calls (Extraction → Detection → Decision → Nested Extraction) make the process slow, especially on local hardware. |
| **Resource Intensive** | Multiple inferences per task put significant load on GPUs/CPUs. |
| **Extraction Dependency** | If the initial extraction is too aggressive in summarizing, it may "prune" the references needed to trigger the rabbithole. |
| **Registry Maintenance** | Scoped search relies on `document_registry.json`. If this mapping is outdated, reference resolution may fail. |
| **Drift Potential** | Despite filtering, deep recursion (Depth 2+) can still accumulate "near-relevant" fluff that dilutes the final report. |
| **Orchestration Complexity** | The system involves many state shifts and branching paths, making it harder to debug than standard RAG. |

---

## 4. Assessment & Recommendation

The Rabbithole implementation is a sophisticated solution to the "shallow RAG" problem. It excels in domain-specific environments (like legal or regulatory compliance) where citations are the primary way information is structured.

### **Current Status:**
The implementation is "state-of-the-art" for local agents, featuring advanced "Graded Context" and "Agentic Gating" that many commercial systems lack.

### **Key Recommendations:**
1.  **Parallelization**: Future versions should parallelize the "Reference Follow Decisions" and "Nested Extractions" to reduce latency.
2.  **Dynamic Registry**: Implement a tool to auto-update the `document_registry.json` byproduct of the embedding process.
3.  **Cross-Check**: Implement a "Refusal Threshold" where the agent stops the rabbithole early if the first few followed refs yield low relevance scores.
