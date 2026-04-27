# Agentic AI on Databricks — April 2026

Personal study repository for the **Databricks Academy Agentic AI** course series, completed April 2026.

---

## Repository Structure

```
agentic_ai_dbx_apr_2026/
├── README.md
├── week-1-building-retrieval-agents-on-databricks-en_us-1.0.1/
├── week-2-building-single-agent-applications-on-databricks-en_us-1.0.1/
└── week-3-genai-agent-evaluation-on-databricks-en_us-1.0.0/
```

---

## Week 1 — Building Retrieval Agents on Databricks

> Folder: `week-1-building-retrieval-agents-on-databricks-en_us-1.0.1/`

**Course overview:** Parse unstructured documents, build vector search solutions, and deploy production-ready retrieval agents using MLflow and Agent Bricks.

### Prerequisites
- Intermediate Python, basic SQL
- Familiarity with Databricks, Unity Catalog, LLMs, and MLflow

### Technical Requirements
- Databricks **Serverless Compute** (environment version 5)
- Access to `ai_parse_document()` (Beta), Mosaic AI Agent Bricks (Beta)
- Pre-created Vector Search endpoint, Foundation Model APIs

### Modules

| Module | Topic | Notebooks |
|--------|-------|-----------|
| **01** | Foundations of Retrieval Agents | 1.1 Lecture — Beyond Prompts: Retrieval Agents and Context Engineering |
| **02** | Document Parsing and Chunking | 2.1 Lecture, 2.2 Demo — Parse Documents, 2.3 Demo — Clean/Transform/Chunk, 2.4 Lab |
| **03** | Vector Search for Retrieval | 3.1 Lecture — Embeddings & Vector Search, 3.2 Demo, 3.3 Lab |
| **04** | Building and Logging Retrieval Agents | 4.1 Lecture — MLflow & Agent Dev, 4.2 Demo, 4.3 Lab |
| **05** | Agent Bricks | 5.1 Lecture — Knowledge Assistant, 5.2 Demo |

---

## Week 2 — Building Single-Agent Applications on Databricks

> Folder: `week-2-building-single-agent-applications-on-databricks-en_us-1.0.1/`

**Course overview:** Build single-agent applications using Unity Catalog functions as tools, MLflow for tracing and reproducibility, LangChain, and Agent Bricks.

### Prerequisites
- Intermediate Python (decorators, OOP, package management), basic SQL
- Familiarity with Databricks, Unity Catalog, Delta Lake, LLMs, and MLflow

### Technical Requirements
- Databricks **Serverless Compute** (version 5+)

### Modules

| Module | Topic | Notebooks |
|--------|-------|-----------|
| **M01** | Foundations of Agents | 1.0–1.1 Lectures, 1.2 Demo, 1.3 Lab — UC Functions as Agent Tools |
| **M02** | Building Single Agents | 2.0 Lecture, 2.1 Demo (LangChain), 2.2 Lab |
| **M03** | Reproducible Agents | 3.0 Lecture, 3.1–3.2 Demos (MLflow tracing & tagging), 3.3 Lab |
| **M04** | Production-Ready Agents with Agent Bricks | 4.0 Lecture, 4.1 Lab — Multi-Agent Supervisor |

### Key Libraries

| Package | Purpose |
|---------|---------|
| `databricks-langchain` | LangChain integration for Databricks |
| `langgraph >= 1.1.5` | Agent graph orchestration |
| `mlflow-skinny[databricks]` | Experiment tracking & model registry |
| `databricks-agents` | Agent deployment & evaluation |
| `unitycatalog-ai[databricks]` | UC functions as agent tools |
| `databricks-sdk` | Databricks workspace SDK |

---

## Week 3 — GenAI Agent Evaluation on Databricks

> Folder: `week-3-genai-agent-evaluation-on-databricks-en_us-1.0.0/`

**Course overview:** Systematically evaluate AI agents using MLflow's evaluation framework, addressing the unique challenges of non-deterministic AI systems. Covers built-in judges, guideline judges, custom judges, offline vs. online evaluation strategies, MLflow tracing, and human feedback collection from developers and SMEs.

### Prerequisites
- Basic Python (OOP, lambda functions, JSON handling), basic SQL (joins, aggregations)
- Databricks workspace, Unity Catalog, serverless compute, MLflow tracing and model registry
- Familiarity with LLMs, prompt engineering, tool-calling agents, and AI evaluation concepts

### Technical Requirements
- Databricks **Serverless Compute** (version 5+)
- MLflow experiment tracking and model registry
- Databricks model serving endpoints

### Modules

| Module | Topic | Notebooks |
|--------|-------|-----------|
| **M01** | AI Agent Evaluation Fundamentals | 1.1 Lecture — The Challenge of Evaluating AI Agents, 1.2 Demo — Agent Setup, 1.3 Lecture — MLflow's Evaluation Framework |
| **M02** | Built-In and Guideline Judges | 2.1 Lecture — Types of Evaluation Judges, 2.2 Demo — Using MLflow Built-In Judges, 2.3 Demo — Guideline Judges with MLflow, 2.4 Lab — Applying Agent Evaluation, 2.5 Demo — Custom Judges with MLflow |
| **M03** | Custom Judges and Human Feedback | 3.1 Lecture — Offline vs. Online Evaluation Strategies, 3.2 Lecture — Best Practices and Practical Application, 3.3 Lab — Developer and SME Feedback with MLflow |

---

## License

&copy; 2026 Databricks, Inc. All rights reserved.  
Apache, Apache Spark, Spark, and related marks are trademarks of the [Apache Software Foundation](https://www.apache.org/).
