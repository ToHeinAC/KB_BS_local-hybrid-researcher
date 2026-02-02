# Product Requirements Document (PRD)
## Rabbithole-Agent: Local Hybrid Researcher

**Version:** 2.2.0
**Status:** Implementation Complete (Baseline + Enhanced Phase 1)
**Last Updated:** 2026-02-02

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Feature-Spezifikationen](#2-feature-spezifikationen)
3. [User Stories](#3-user-stories)
4. [Akzeptanzkriterien](#4-akzeptanzkriterien)
5. [Produkt-Flows](#5-produkt-flows)
6. [Technical Requirements](#6-technical-requirements)
7. [Configuration Reference](#7-configuration-reference)
8. [Data Models Reference](#8-data-models-reference)

---

## 1. Executive Summary

### 1.1 Produktvision

Der Rabbithole-Agent ist ein vollständig lokales, datenschutzorientiertes Forschungssystem, das **tiefe Referenzverfolgung** über Dokumentensammlungen hinweg durchführt. Im Gegensatz zu klassischen RAG-Systemen (Retrieval-Augmented Generation) versteht dieser Agent Inter-Dokument-Beziehungen und folgt iterativ Querverweisen - das sogenannte "Rabbithole-Prinzip".

### 1.2 Core Problem Statement

**Klassisches RAG-Problem:**
- Mangel an tiefem kontextuellem Verständnis
- Keine Verfolgung von Inter-Dokument-Beziehungen
- Flache, einmalige Suche ohne iterative Vertiefung
- Keine Qualitätssicherung der generierten Antworten

**Unsere Lösung:**
- Iterative Vertiefung ("in den Rabbithole graben")
- Referenzverfolgung über Dokumentgrenzen hinweg
- Qualitätsbewertung mit 4-dimensionalem Scoring
- Human-In-The-Loop (HITL) an kritischen Entscheidungspunkten

### 1.3 Key Differentiators

| Merkmal | Klassisches RAG | Rabbithole-Agent |
|---------|-----------------|-------------------|
| Suchtiefe | Einmalig, flach | Iterativ, bis zu 2 Ebenen tief |
| Referenzen | Ignoriert | Erkennt und folgt (§, Dokumente, URLs) |
| Nutzerinteraktion | Keine/Minimal | Iteratives HITL mit Konvergenz |
| Qualität | Unbewertet | 4-dimensionales Scoring (0-400) |
| Datenschutz | Oft Cloud-basiert | 100% lokal (Ollama) |
| Kontext | Statisch | Dynamisch mit Wissensstandverfolgung |

### 1.4 Target Users

- **Forscher und Analysten:** Tiefe Dokumentenrecherche über regulatorische Texte
- **Compliance-Beauftragte:** Verfolgung von Querverweisen in Rechtstexten
- **Technische Redakteure:** Identifikation zusammenhängender Dokumentation
- **Datenschutz-bewusste Organisationen:** Lokale Verarbeitung ohne Cloud-Abhängigkeit

---

## 2. Feature-Spezifikationen

Die System-Architektur besteht aus 5 Phasen, detailliert beschrieben in [docs/architecture.md](docs/architecture.md).

### 2.1 Phase 1: Enhanced Query Analysis + Iterative HITL

**Kernfunktion:** Intelligente Anfrageanalyse mit iterativer Nutzerverfeinerung und Echtzeit-Vektorsuche.

#### 2.1.1 Sprachenerkennung

- **Automatische Erkennung:** Deutsch (de) oder Englisch (en)
- **Implementierung:** LLM-basierte Analyse mit `LANGUAGE_DETECTION_PROMPT`
- **Auswirkung:** Alle folgenden Prompts werden sprachspezifisch angepasst
- **Referenz:** `src/services/hitl_service.py`, `src/prompts.py`

#### 2.1.2 Iterativer HITL-Loop mit Multi-Vektor-Retrieval

```
┌──────────────────────────────────────────────────────────────────┐
│  hitl_init → hitl_generate_queries → hitl_retrieve_chunks →      │
│  hitl_analyze_retrieval → hitl_generate_questions → [wait] →     │
│  hitl_process_response → [loop back or hitl_finalize]           │
└──────────────────────────────────────────────────────────────────┘
```

**Node-Beschreibungen:**

| Node | Funktion | Output |
|------|----------|--------|
| `hitl_init` | Initialisierung, Spracherkennung | `detected_language`, `hitl_iteration=0` |
| `hitl_generate_queries` | 3 Suchqueries pro Iteration | `iteration_queries` Liste |
| `hitl_retrieve_chunks` | Vektorsuche mit Deduplizierung | ~9 Chunks, `query_retrieval` |
| `hitl_analyze_retrieval` | LLM-Analyse der Ergebnisse | `key_concepts`, `knowledge_gaps`, `coverage_score` |
| `hitl_generate_questions` | 2-3 kontextuelle Fragen | Follow-up Questions für Nutzer |
| `hitl_process_response` | Nutzerantwort analysieren | Terminierungsprüfung |
| `hitl_finalize` | Forschungsqueries generieren | `research_queries` für Phase 2 |

**Query-Generierung (Iteration 0):**
- `original`: Ursprüngliche Nutzeranfrage
- `broader_scope`: Kontextuelle/verwandte Informationen
- `alternative_angle`: Implikationen/Herausforderungen

**Query-Generierung (Iteration N>0):**
- Verfeinert basierend auf Nutzerfeedback
- Fokussiert auf identifizierte `knowledge_gaps`
- Nutzt akkumulierte `entities` und `scope`

**Referenz:** [docs/architecture.md](docs/architecture.md) - Abschnitt "Iterative HITL Flow"

#### 2.1.3 Konvergenzerkennung

Das System erkennt automatisch, wann ausreichend Kontext gesammelt wurde:

| Bedingung | Schwellwert | Beschreibung |
|-----------|-------------|--------------|
| Coverage Score | ≥ 0.80 | Abdeckung der Nutzeranfrage |
| Dedup Ratio | ≥ 0.70 | Anteil neuer (nicht-duplikater) Chunks |
| Knowledge Gaps | ≤ 2 | Anzahl offener Wissenslücken |

**Terminierungsgründe:**
- `user_end`: Nutzer tippt `/end`
- `max_iterations`: Maximum erreicht (Standard: 5)
- `convergence`: Alle 3 Bedingungen erfüllt

**Referenz:** `src/agents/graph.py` Zeilen 111-122

### 2.2 Phase 2: Research Planning

**Kernfunktion:** Generierung einer strukturierten Aufgabenliste für die Forschung.

#### 2.2.1 ToDoList-Generierung

- **Initiale Aufgaben:** 3-5 Tasks
- **Maximum:** 15 Tasks (konfigurierbar via `TODO_MAX_ITEMS`)
- **Quelle:** `research_queries` aus Phase 1 oder LLM-Generierung
- **Prompt:** `TODO_GENERATION_PROMPT` in `src/prompts.py`

**Task-Struktur:**
```python
class ToDoItem:
    id: str           # Eindeutige ID (z.B. "task_1")
    task: str         # Spezifische, messbare Aufgabe
    context: str      # Warum diese Aufgabe wichtig ist
    completed: bool   # Erledigungsstatus
```

#### 2.2.2 Nutzer-Approval-Workflow

1. **Checkpoint erstellen:** `hitl_approve_todo` Node
2. **Anzeige:** ToDoList im UI mit Bearbeitungsoptionen
3. **Nutzeraktionen:**
   - Aufgaben genehmigen
   - Aufgaben entfernen (`removed_ids`)
   - Aufgaben bearbeiten (`edited_items`)
   - Neue Aufgaben hinzufügen (`new_items`)
4. **Anwendung:** `process_hitl_todo` wendet Modifikationen an

**Referenz:** [docs/agent-design.md](docs/agent-design.md) - Abschnitt "HITL Integration Points"

### 2.3 Phase 3: Deep Context Extraction (Rabbithole Magic)

**Kernfunktion:** Tiefe Kontextextraktion durch iterative Referenzverfolgung.

Dies ist das Herzstück des Systems, detailliert beschrieben in [docs/rabbithole-magic.md](docs/rabbithole-magic.md).

#### 2.3.1 Ausführungsfluss pro Task

```
Vector Search → Extract Info → Detect Refs → Follow Refs →
Filter by Relevance → Update ToDoList → Loop until done
```

**Schritt-für-Schritt:**

1. **Vector Search:** Top-k Ergebnisse (Standard k=4)
2. **Chunk Processing:** LLM extrahiert relevante Passagen
3. **Reference Detection:** Regex-Patterns für §, Dokumente, URLs
4. **Reference Resolution:** Rekursiv, tiefenkontrolliert
5. **Relevance Filtering:** Schwellwert 0.6, Zwei-Stufen-Scoring
6. **Context Accumulation:** Hinzufügen zu `research_context`
7. **Task Completion:** Markieren, nächsten Task finden

#### 2.3.2 Referenztypen

| Typ | Pattern-Beispiele | Auflösungsstrategie |
|-----|-------------------|---------------------|
| Section | `§123`, `gemäß §45 StrlSchV` | Sektionssuche im Dokument |
| Document | `siehe Dokument ABC`, `EU 2024/123` | Dokument-Mapping in ChromaDB |
| External | `https://...` | Markierung für Web-Suche (optional) |

**Regex-Patterns:** Definiert in `src/agents/tools.py`

#### 2.3.3 Tiefenverfolgung

```
Depth 0: Initiale Suche
  └─ Depth 1: Erste Referenzebene
    └─ Depth 2: Zweite Referenzebene (MAXIMUM)
```

- **Standard-Tiefe:** 2 (konfigurierbar via `REFERENCE_FOLLOW_DEPTH`)
- **Reset:** Nach jedem abgeschlossenen Task auf 0

#### 2.3.4 Loop-Prävention

- **visited_refs Set:** Speichert "type:target" Keys
- **MAX_ITERATIONS_PER_TASK:** Standard 3
- **Terminierung:** Kein aktueller Task, alle Tasks erledigt, oder Tiefenlimit erreicht

**Referenz:** [docs/rabbithole-magic.md](docs/rabbithole-magic.md) - Vollständige Algorithmus-Dokumentation

### 2.4 Phase 4: Synthesis + Quality Assurance

**Kernfunktion:** Zusammenführung der Forschungsergebnisse mit Qualitätsbewertung.

#### 2.4.1 Synthese

- **Input:** Alle `SearchQueryResult` aus Phase 3
- **Prozess:** LLM aggregiert Erkenntnisse
- **Output:** `SynthesisOutput` mit `summary` und `key_findings`
- **Prompt:** `SYNTHESIS_PROMPT` in `src/prompts.py`

**Sprachhandhabung:** Synthese erfolgt in der erkannten Nutzersprache

#### 2.4.2 Qualitätsbewertung (4-Dimensional)

| Dimension | Gewichtung | Beschreibung |
|-----------|------------|--------------|
| Factual Accuracy | 0-100 | Faktische Korrektheit der Aussagen |
| Semantic Validity | 0-100 | Semantische Kohärenz |
| Structural Integrity | 0-100 | Strukturelle Vollständigkeit |
| Citation Correctness | 0-100 | Korrektheit der Quellenangaben |

**Gesamtscore:** 0-400 Punkte
**Schwellwert:** 300 (Standard, konfigurierbar via `QUALITY_THRESHOLD`)

**Bei Nicht-Bestehen:**
- `issues_found` Liste wird generiert
- Optional: Reflection-Loop (max 1 Iteration via `MAX_REFLECTIONS`)

**Referenz:** [docs/implementation.md](docs/implementation.md) - Phase 4 Details

### 2.5 Phase 5: Source Attribution

**Kernfunktion:** Generierung zitierbarerer Quellenangaben mit klickbaren Links.

#### 2.5.1 Quellenerfassung

Für jede verwendete Quelle werden erfasst:
- `doc_id`: Dokumentenidentifikator
- `page_number`: Seitenzahl
- `relevance_score`: Relevanzwert (0-1)
- `collection`: ChromaDB-Collection
- `category`: Dokumentkategorie (GLageKon, NORM, StrlSch, StrlSchExt)

#### 2.5.2 Pfadauflösung

**Collection → Ordner Mapping:**

| Collection | Quellordner |
|------------|-------------|
| `GLageKon__*` | `kb/GLageKon__db_inserted/` |
| `NORM__*` | `kb/NORM__db_inserted/` |
| `StrlSch__*` | `kb/StrlSch__db_inserted/` |
| `StrlSchExt__*` | `kb/StrlSchExt__db_inserted/` |

**Implementierung:** `source_path_resolver()` in `src/services/`

#### 2.5.3 Final Report Struktur

```python
class FinalReport:
    query: str                    # Ursprüngliche Anfrage
    answer: str                   # Synthetisierte Antwort
    findings: List[Finding]       # Erkenntnisse mit Evidenz
    sources: List[LinkedSource]   # Quellen mit Pfaden
    quality_score: int            # Gesamtqualität (0-400)
    quality_breakdown: dict       # 4-dimensionale Aufschlüsselung
    metadata: dict                # Ausführungsmetadaten
```

**Referenz:** [docs/data-sources.md](docs/data-sources.md) - Collection Mapping

---

## 3. User Stories

### 3.1 Forschungsanfrage stellen

**US-01: Einfache Forschungsanfrage**
> Als **Forscher** möchte ich **eine Frage zu regulatorischen Dokumenten stellen**, damit **ich fundierte Antworten mit Quellenangaben erhalte**.

**US-02: Datenbankauswahl**
> Als **Forscher** möchte ich **eine spezifische Dokumentensammlung auswählen können**, damit **meine Suche auf relevante Dokumente beschränkt wird**.

**US-03: Mehrsprachige Unterstützung**
> Als **deutschsprachiger Nutzer** möchte ich **Fragen auf Deutsch stellen und Antworten auf Deutsch erhalten**, damit **die Kommunikation in meiner Arbeitssprache erfolgt**.

### 3.2 Iterative Verfeinerung

**US-04: Kontextuelle Nachfragen**
> Als **Forscher** möchte ich **Nachfragen zum Verständnis meiner Anfrage erhalten**, damit **das System meine Intention besser verstehen kann**.

**US-05: Echtzeit-Retrieval-Feedback**
> Als **Forscher** möchte ich **sehen, welche Dokumente bei der Verfeinerung gefunden werden**, damit **ich den Fortschritt der Kontextsammlung nachvollziehen kann**.

**US-06: Vorzeitige Beendigung**
> Als **Forscher** möchte ich **die Verfeinerungsphase jederzeit mit `/end` beenden können**, damit **ich bei ausreichendem Kontext schneller zur Forschung übergehen kann**.

**US-07: Konvergenz-Information**
> Als **Forscher** möchte ich **den Coverage-Score und offene Wissenslücken sehen**, damit **ich einschätzen kann, wie gut das System meine Anfrage versteht**.

### 3.3 Aufgabenverwaltung

**US-08: Aufgabenliste überprüfen**
> Als **Forscher** möchte ich **die generierten Forschungsaufgaben vor der Ausführung sehen**, damit **ich die Forschungsrichtung kontrollieren kann**.

**US-09: Aufgaben bearbeiten**
> Als **Forscher** möchte ich **Aufgaben bearbeiten, hinzufügen oder entfernen können**, damit **die Forschung genau meinen Bedürfnissen entspricht**.

**US-10: Fortschrittsanzeige**
> Als **Forscher** möchte ich **den aktuellen Fortschritt der Aufgabenbearbeitung sehen**, damit **ich weiß, wie weit die Forschung ist**.

### 3.4 Forschungsergebnisse

**US-11: Quellenangaben**
> Als **Forscher** möchte ich **für jede Aussage die Quelldokumente und Seitenzahlen sehen**, damit **ich Aussagen verifizieren kann**.

**US-12: Qualitätsbewertung**
> Als **Forscher** möchte ich **eine Qualitätsbewertung der Antwort sehen**, damit **ich das Vertrauen in die Ergebnisse einschätzen kann**.

**US-13: Klickbare Quellenlinks**
> Als **Forscher** möchte ich **auf Quellen klicken können, um das Originaldokument zu öffnen**, damit **ich schnell zur Primärquelle navigieren kann**.

### 3.5 Datenschutz und Kontrolle

**US-14: Lokale Verarbeitung**
> Als **datenschutzbewusster Nutzer** möchte ich **dass alle Daten lokal verarbeitet werden**, damit **keine sensiblen Informationen das System verlassen**.

**US-15: Sicheres Beenden**
> Als **Nutzer** möchte ich **die Anwendung sicher beenden können**, damit **keine Prozesse im Hintergrund weiterlaufen**.

**US-16: Konfigurierbare Parameter**
> Als **Administrator** möchte ich **Systemparameter über Umgebungsvariablen konfigurieren können**, damit **ich das System an unsere Infrastruktur anpassen kann**.

### 3.6 Referenzverfolgung

**US-17: Automatische Referenzerkennung**
> Als **Forscher** möchte ich **dass Querverweise in Dokumenten automatisch erkannt werden**, damit **zusammenhängende Informationen nicht übersehen werden**.

**US-18: Tiefe Kontextextraktion**
> Als **Forscher** möchte ich **dass das System Referenzen bis zu 2 Ebenen tief verfolgt**, damit **ich ein vollständiges Bild der Zusammenhänge erhalte**.

---

## 4. Akzeptanzkriterien

### 4.1 Phase 1: Iterative HITL

| ID | Kriterium | Bedingung |
|----|-----------|-----------|
| AC-1.1 | Spracherkennung | System erkennt Deutsch oder Englisch korrekt in >95% der Fälle |
| AC-1.2 | Query-Generierung | Pro Iteration werden exakt 3 Suchqueries generiert |
| AC-1.3 | Chunk-Retrieval | ~9 Chunks pro Iteration werden abgerufen (3 pro Query) |
| AC-1.4 | Deduplizierung | Bereits abgerufene Chunks werden nicht erneut zurückgegeben |
| AC-1.5 | Konvergenz | System terminiert automatisch bei coverage ≥0.8 AND dedup ≥0.7 AND gaps ≤2 |
| AC-1.6 | Max Iterationen | System terminiert spätestens nach 5 Iterationen (konfigurierbar) |
| AC-1.7 | User End | `/end` Eingabe beendet HITL-Phase sofort |
| AC-1.8 | Follow-up Questions | 2-3 kontextuelle Fragen werden pro Iteration generiert |

### 4.2 Phase 2: Research Planning

| ID | Kriterium | Bedingung |
|----|-----------|-----------|
| AC-2.1 | Task-Generierung | 3-5 initiale Tasks werden generiert |
| AC-2.2 | Task-Limit | Maximale Task-Anzahl beträgt 15 |
| AC-2.3 | HITL Checkpoint | System pausiert für Nutzergenehmigung vor Forschungsstart |
| AC-2.4 | Task-Bearbeitung | Nutzer kann Tasks hinzufügen, entfernen und bearbeiten |
| AC-2.5 | Modifikationsanwendung | Alle Nutzermodifikationen werden korrekt angewendet |

### 4.3 Phase 3: Deep Context Extraction

| ID | Kriterium | Bedingung |
|----|-----------|-----------|
| AC-3.1 | Vector Search | Top-k Ergebnisse werden abgerufen (Standard k=4) |
| AC-3.2 | Referenzerkennung | §-Verweise, Dokumentverweise und URLs werden erkannt |
| AC-3.3 | Referenzverfolgung | Referenzen werden bis zu 2 Ebenen tief verfolgt |
| AC-3.4 | Loop-Prävention | Bereits besuchte Referenzen werden nicht erneut verfolgt |
| AC-3.5 | Relevanzfilterung | Nur Chunks mit Relevanz ≥0.6 werden einbezogen |
| AC-3.6 | Task-Iteration | Maximal 3 Iterationen pro Task |
| AC-3.7 | Kontextakkumulation | Alle relevanten Informationen werden zu research_context hinzugefügt |

### 4.4 Phase 4: Synthesis + Quality

| ID | Kriterium | Bedingung |
|----|-----------|-----------|
| AC-4.1 | Synthese | Alle Erkenntnisse werden zu einer kohärenten Antwort zusammengefasst |
| AC-4.2 | Sprachkonsistenz | Synthese erfolgt in der erkannten Nutzersprache |
| AC-4.3 | Qualitätsdimensionen | Alle 4 Dimensionen werden bewertet (0-100 jeweils) |
| AC-4.4 | Qualitätsschwelle | Gesamtscore ≥300 erforderlich (konfigurierbar) |
| AC-4.5 | Issue-Reporting | Bei Qualitätsproblemen wird `issues_found` Liste generiert |

### 4.5 Phase 5: Source Attribution

| ID | Kriterium | Bedingung |
|----|-----------|-----------|
| AC-5.1 | Quellenerfassung | Alle verwendeten Quellen werden mit doc_id und Seite erfasst |
| AC-5.2 | Pfadauflösung | Quellpfade werden korrekt zu KB-Ordnern aufgelöst |
| AC-5.3 | Final Report | Report enthält query, answer, findings, sources, quality_score |
| AC-5.4 | Zitierbarkeit | Jede Aussage kann zur Quelle zurückverfolgt werden |

### 4.6 Systemweite Kriterien

| ID | Kriterium | Bedingung |
|----|-----------|-----------|
| AC-6.1 | Lokale Ausführung | Keine externen API-Aufrufe (außer optional Web-Suche) |
| AC-6.2 | Fehlerbehandlung | System gibt verständliche Fehlermeldungen bei Problemen |
| AC-6.3 | Session-Persistenz | HITL-Konversation kann nach Unterbrechung fortgesetzt werden |
| AC-6.4 | Safe Exit | Anwendung kann sauber beendet werden ohne hängende Prozesse |
| AC-6.5 | Konfigurierbarkeit | Alle Hauptparameter über .env konfigurierbar |

---

## 5. Produkt-Flows

### 5.1 Hauptworkflow (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER STARTS SESSION                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Enhanced Query Analysis                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  User Input: Query + Optional Database Selection                │ │
│  │                          │                                      │ │
│  │                          ▼                                      │ │
│  │  ┌─────────────────────────────────────────────────────────────┐│ │
│  │  │  ITERATIVE HITL LOOP (0-5 iterations)                       ││ │
│  │  │  • Generate 3 search queries                                 ││ │
│  │  │  • Retrieve ~9 chunks from vector DB                        ││ │
│  │  │  • Analyze retrieval (concepts, gaps, coverage)             ││ │
│  │  │  • Generate 2-3 follow-up questions                         ││ │
│  │  │  • Wait for user response                                   ││ │
│  │  │  • Check: /end? max_iterations? convergence?                ││ │
│  │  └─────────────────────────────────────────────────────────────┘│ │
│  │                          │                                      │ │
│  │                          ▼                                      │ │
│  │  Output: research_queries[], coverage_score, query_analysis     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Research Planning                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  • Generate ToDoList (3-5 tasks from research_queries)          │ │
│  │  • Create HITL checkpoint                                       │ │
│  │  • User reviews/edits tasks                                     │ │
│  │  • Apply modifications                                          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  Output: Approved ToDoList                                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Deep Context Extraction (Rabbithole)                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  FOR EACH TASK:                                                 │ │
│  │    ┌───────────────────────────────────────────────────────────┐│ │
│  │    │  1. Vector search (top-k results)                         ││ │
│  │    │  2. Extract relevant info from chunks                     ││ │
│  │    │  3. Detect references (§, documents, URLs)                ││ │
│  │    │  4. Follow references (depth 0→1→2)                       ││ │
│  │    │  5. Filter by relevance (≥0.6)                            ││ │
│  │    │  6. Update research_context                               ││ │
│  │    │  7. Mark task complete, get next                          ││ │
│  │    └───────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  Output: research_context with all findings                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: Synthesis + Quality Assurance                              │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  1. Synthesize findings → summary + key_findings                │ │
│  │  2. Quality check (4 dimensions × 100 = 400 max)               │ │
│  │  3. If score < 300: generate issues_found                       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  Output: synthesis + quality_assessment                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: Source Attribution                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  1. Collect all sources (doc_id, page, relevance)              │ │
│  │  2. Resolve paths to source PDFs                               │ │
│  │  3. Generate clickable links                                    │ │
│  │  4. Build FinalReport                                           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  Output: FinalReport with answer, findings, sources, quality         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DISPLAY RESULTS                              │
│  • Answer in user's language                                         │
│  • Key findings with evidence                                        │
│  • Quality score breakdown                                           │
│  • Clickable source links                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Iterative HITL Flow (Detail)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HITL_INIT                                     │
│  • Initialize hitl_iteration = 0                                     │
│  • Detect language (de/en)                                           │
│  • Set hitl_active = true                                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HITL_GENERATE_QUERIES                             │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  IF iteration == 0:                                              ││
│  │    query_1 = original query                                      ││
│  │    query_2 = broader_scope (context/related)                     ││
│  │    query_3 = alternative_angle (implications)                    ││
│  │  ELSE:                                                           ││
│  │    query_1 = refined based on user feedback                      ││
│  │    query_2 = gap-focused query                                   ││
│  │    query_3 = concept-focused query                               ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HITL_RETRIEVE_CHUNKS                              │
│  • Execute 3 vector searches (3 chunks each)                         │
│  • Deduplicate against accumulated query_retrieval                   │
│  • Track dedup_ratio for convergence detection                       │
│  • Append new chunks to query_retrieval                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   HITL_ANALYZE_RETRIEVAL                             │
│  LLM analyzes retrieval context:                                     │
│  • Extract key_concepts (5-7)                                        │
│  • Identify entities                                                 │
│  • Determine scope                                                   │
│  • Find knowledge_gaps                                               │
│  • Estimate coverage_score (0-1)                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  HITL_GENERATE_QUESTIONS                             │
│  Generate 2-3 follow-up questions informed by:                       │
│  • Current coverage_score                                            │
│  • Identified knowledge_gaps                                         │
│  • Retrieved context (query_retrieval)                               │
│  • Conversation history                                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      USER RESPONSE                                   │
│  User sees:                                                          │
│  • Generated follow-up questions                                     │
│  • Current coverage score                                            │
│  • Knowledge gaps                                                    │
│  • Option to type /end                                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 HITL_PROCESS_RESPONSE                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  CHECK TERMINATION:                                              ││
│  │    • User typed /end? → user_end → FINALIZE                      ││
│  │    • hitl_iteration ≥ max? → max_iterations → FINALIZE           ││
│  │    • coverage ≥0.8 AND dedup ≥0.7 AND gaps ≤2?                   ││
│  │          → convergence → FINALIZE                                 ││
│  │    • ELSE → increment iteration → LOOP BACK                       ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
            │                                      │
            │ (terminate)                          │ (continue)
            ▼                                      ▼
┌───────────────────────────┐      ┌───────────────────────────────────┐
│     HITL_FINALIZE         │      │   BACK TO HITL_GENERATE_QUERIES   │
│  • Generate research_     │      │   (iteration += 1)                │
│    queries from context   │      └───────────────────────────────────┘
│  • Set hitl_active=false  │
│  • Record termination     │
│    reason                 │
└───────────────────────────┘
            │
            ▼
    [PHASE 2: Research Planning]
```

### 5.3 Rabbithole Reference-Following Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  START TASK EXECUTION                                                │
│  current_task = todo_list.get_next_task()                           │
│  current_depth = 0                                                   │
│  visited_refs = set()                                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VECTOR SEARCH                                                       │
│  query = task.description + key_concepts                             │
│  chunks = chromadb.search(query, k=4)                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FOR EACH CHUNK:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  1. EXTRACT INFO                                                 ││
│  │     extracted = llm.extract_relevant_passages(chunk, query)     ││
│  │                                                                  ││
│  │  2. DETECT REFERENCES                                            ││
│  │     refs = detect_references(extracted)                          ││
│  │     → Section refs: §123, gemäß §45 StrlSchV                     ││
│  │     → Document refs: siehe Dokument ABC                          ││
│  │     → External refs: https://...                                 ││
│  │                                                                  ││
│  │  3. RELEVANCE FILTER                                             ││
│  │     IF chunk.relevance_score < 0.6: SKIP                        ││
│  │                                                                  ││
│  │  4. ADD TO CONTEXT                                               ││
│  │     research_context.add(ChunkWithInfo)                         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FOLLOW REFERENCES (if depth < 2)                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  FOR EACH detected_reference:                                    ││
│  │    ref_key = f"{ref.type}:{ref.target}"                          ││
│  │                                                                  ││
│  │    IF ref_key IN visited_refs: SKIP (loop prevention)           ││
│  │                                                                  ││
│  │    visited_refs.add(ref_key)                                     ││
│  │    current_depth += 1                                            ││
│  │                                                                  ││
│  │    RESOLVE REFERENCE:                                            ││
│  │    ┌───────────────────────────────────────────────────────────┐││
│  │    │  IF type == "section":                                    │││
│  │    │    nested_chunks = search_section_in_doc(target)          │││
│  │    │  ELIF type == "document":                                 │││
│  │    │    nested_chunks = search_document(target)                │││
│  │    │  ELIF type == "external":                                 │││
│  │    │    mark_for_web_search(target) [if enabled]               │││
│  │    └───────────────────────────────────────────────────────────┘││
│  │                                                                  ││
│  │    RECURSIVELY process nested_chunks (depth check)               ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TASK COMPLETION CHECK                                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  IF sufficient_info_gathered OR max_iterations_reached:         ││
│  │    task.completed = true                                         ││
│  │    current_depth = 0  (reset for next task)                      ││
│  │    current_task = todo_list.get_next_task()                      ││
│  │                                                                  ││
│  │  IF no more tasks:                                               ││
│  │    → PROCEED TO PHASE 4                                          ││
│  │  ELSE:                                                           ││
│  │    → LOOP BACK TO VECTOR SEARCH                                  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Fehlerbehandlung Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  ERROR DETECTION                                                     │
│  • LLM timeout/failure                                               │
│  • ChromaDB connection error                                         │
│  • JSON parsing failure                                              │
│  • Ollama service unavailable                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ERROR HANDLING STRATEGIES                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  LLM Timeout:                                                    ││
│  │    • Retry with exponential backoff (tenacity)                   ││
│  │    • Fall back to qwen3:8b if qwen3:14b fails                    ││
│  │    • Add warning to state.warnings                               ││
│  │                                                                  ││
│  │  JSON Parse Error:                                               ││
│  │    • Fall back to default values                                 ││
│  │    • Log error and continue                                      ││
│  │                                                                  ││
│  │  ChromaDB Error:                                                 ││
│  │    • Display error message to user                               ││
│  │    • Suggest checking database path                              ││
│  │                                                                  ││
│  │  Critical Error:                                                 ││
│  │    • Set state.error with message                                ││
│  │    • Route to END node                                           ││
│  │    • Display user-friendly error in UI                           ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RECOVERY OPTIONS                                                    │
│  • User can retry query                                              │
│  • User can modify parameters                                        │
│  • User can select different database                                │
│  • Safe exit always available                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Technical Requirements

### 6.1 Framework Requirements

| Komponente | Anforderung | Begründung |
|------------|-------------|------------|
| Python | ≥3.10 | LangChain v1.0 Requirement |
| LangChain | ≥1.0.0 | TypedDict State, neue API |
| LangGraph | ≥1.0.0 | StateGraph, Checkpointing |
| Pydantic | ≥2.0.0 | Structured Output, Validation |

**Referenz:** [docs/references.md](docs/references.md) - LangChain v1.0 Migration

### 6.2 LLM Requirements

| Modell | Verwendung | Kontext |
|--------|------------|---------|
| qwen3:14b | Primär | 128K Token (OLLAMA_NUM_CTX) |
| qwen3:8b | Fallback | 128K Token |
| Qwen3-Embedding-0.6B | Embeddings | 1024 Dimensionen |

**Structured Output:**
```python
llm.with_structured_output(Model, method="json_mode")
```

**Referenz:** [docs/agent-design.md](docs/agent-design.md) - Ollama Structured Outputs

### 6.3 Vector Database

| Aspekt | Spezifikation |
|--------|---------------|
| System | ChromaDB (persistent) |
| Pfad | `./kb/database` |
| Collections | 4 (GLageKon, NORM, StrlSch, StrlSchExt) |
| Chunk-Größe | 3K-10K je nach Collection |
| Overlap | 600-2000 je nach Collection |

**Referenz:** [docs/data-sources.md](docs/data-sources.md) - ChromaDB Collections

### 6.4 State Management

**LangGraph Requirement:**
- Agent State MUSS `TypedDict` verwenden (nicht Pydantic)
- Pydantic für Tool-Inputs/Outputs
- Serialisierung zu/von Dicts

**Referenz:** [docs/data-models.md](docs/data-models.md) - Vollständige State-Definition

### 6.5 UI Requirements

| Aspekt | Spezifikation |
|--------|---------------|
| Framework | Streamlit ≥1.28.0 |
| Port | 8511 (konfigurierbar) |
| Session State | HITL-Konversation, Workflow-Phase |
| Caching | `@st.cache_resource` für Services |

---

## 7. Configuration Reference

Vollständige Konfigurationsdetails in [docs/configuration.md](docs/configuration.md).

### 7.1 Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| **Ollama** |||
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama Service URL |
| `OLLAMA_MODEL` | `qwen3:14b` | Primäres LLM |
| `OLLAMA_FALLBACK_MODEL` | `qwen3:8b` | Fallback LLM |
| `OLLAMA_NUM_CTX` | `131072` | Kontextfenster (128K) |
| `OLLAMA_SAFE_LIMIT` | `0.9` | Token-Limit (90%) |
| **ChromaDB** |||
| `CHROMADB_PATH` | `./kb/database` | Datenbankpfad |
| **HITL** |||
| `MAX_CLARIFICATION_QUESTIONS` | `3` | Fragen pro Iteration |
| **ToDo** |||
| `TODO_MAX_ITEMS` | `15` | Maximum Tasks |
| `INITIAL_TODO_ITEMS` | `5` | Initiale Tasks |
| **Suche** |||
| `M_CHUNKS_PER_QUERY` | `4` | Chunks pro Query |
| `REFERENCE_FOLLOW_DEPTH` | `2` | Max Referenztiefe |
| `REFERENCE_RELEVANCE_THRESHOLD` | `0.6` | Min Relevanz |
| **Qualität** |||
| `QUALITY_THRESHOLD` | `300` | Min Qualitätsscore |
| `ENABLE_QUALITY_CHECKER` | `true` | Qualitätsprüfung |

### 7.2 Settings-Klasse

```python
class Settings(BaseSettings):
    """Lädt Konfiguration aus .env mit Defaults"""

    ollama_model: str = "qwen3:14b"
    chromadb_path: str = "./kb/database"
    reference_follow_depth: int = 2
    quality_threshold: int = 300
    # ... weitere Felder

    class Config:
        env_file = ".env"
```

---

## 8. Data Models Reference

Vollständige Modell-Definitionen in [docs/data-models.md](docs/data-models.md).

### 8.1 Kernmodelle

| Modell | Zweck | Schlüsselfelder |
|--------|-------|-----------------|
| `QueryAnalysis` | Analysierte Anfrage | `original_query`, `key_concepts`, `entities`, `scope` |
| `ToDoItem` | Einzelner Task | `id`, `task`, `context`, `completed` |
| `ToDoList` | Task-Container | `items`, `add_task()`, `get_next_task()` |
| `ChunkWithInfo` | Chunk mit Extraktion | `chunk`, `extracted_info`, `references`, `relevance_score` |
| `DetectedReference` | Erkannte Referenz | `type`, `target`, `nested_chunks` |
| `VectorResult` | Suchergebnis | `doc_id`, `chunk_text`, `page_number`, `relevance_score` |
| `QualityAssessment` | Qualitätsbewertung | 4 Dimensionen, `issues_found` |
| `FinalReport` | Endergebnis | `answer`, `findings`, `sources`, `quality_score` |

### 8.2 HITL-Spezifische Modelle

| Modell | Zweck |
|--------|-------|
| `HITLCheckpoint` | Checkpoint-Typ und Inhalt |
| `HITLDecision` | Nutzerentscheidung |
| `AlternativeQueriesOutput` | Query-Alternativen |

### 8.3 AgentState Felder (Auswahl)

```python
class AgentState(TypedDict):
    # Kern
    query: str
    research_context: dict
    final_report: dict

    # HITL Enhanced
    hitl_iteration: int
    coverage_score: float
    knowledge_gaps: List[str]
    query_retrieval: str

    # Referenzverfolgung
    visited_refs: set
    current_depth: int
```

---

## Anhang

### A. Glossar

| Begriff | Definition |
|---------|------------|
| HITL | Human-In-The-Loop - Nutzereinbindung an Entscheidungspunkten |
| Rabbithole | Iterative Tiefensuche durch Referenzverfolgung |
| Convergence | Automatische Terminierung bei ausreichender Informationsabdeckung |
| Dedup Ratio | Verhältnis neuer zu bereits gesehener Chunks |
| Coverage Score | Schätzung der Abdeckung der Nutzeranfrage (0-1) |

### B. Referenzen

- [CLAUDE.md](CLAUDE.md) - Projekt-Übersicht und Quick Start
- [docs/architecture.md](docs/architecture.md) - Systemarchitektur
- [docs/agent-design.md](docs/agent-design.md) - Agent-Patterns
- [docs/data-models.md](docs/data-models.md) - Pydantic-Modelle
- [docs/data-sources.md](docs/data-sources.md) - Datenquellen
- [docs/configuration.md](docs/configuration.md) - Konfiguration
- [docs/implementation.md](docs/implementation.md) - Implementierung
- [docs/rabbithole-magic.md](docs/rabbithole-magic.md) - Referenzverfolgung
- [docs/references.md](docs/references.md) - Externe Ressourcen
