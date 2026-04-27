# Building Single-Agent Applications on Databricks

> **Databricks Academy** course materials — Week 2  
> Version: 1.0.1

## Overview

This course provides hands-on training for building single-agent applications on the Databricks Data Intelligence Platform. Topics covered include:

- Creating AI agents that leverage **Unity Catalog (UC) functions** as tools
- Implementing tracing and monitoring with **MLflow**
- Deploying agents using **LangChain** and **Agent Bricks**
- Managing the complete agent lifecycle: tool creation → testing → production deployment

## Prerequisites

- Intermediate Python (decorators, OOP, package management)
- Basic SQL and Unity Catalog concepts (catalogs, schemas, governance)
- Familiarity with Databricks workspace, Delta Lake, and Jupyter-style notebooks
- Basic understanding of LLMs, prompt engineering, and MLflow

## Runtime Requirement

All demo and lab notebooks must be run on **Databricks Serverless** compute (version 5+).

---

## Course Modules

| Module | Topic | Notebooks |
|--------|-------|-----------|
| **M01** | Foundations of Agents | Lectures, Demo, Lab — UC Functions as Agent Tools |
| **M02** | Building Single Agents | Lecture, Demo (LangChain), Lab |
| **M03** | Reproducible Agents | MLflow tracing, tagging, reproducibility demo & lab |
| **M04** | Production-Ready Agents with Agent Bricks | Lecture, Lab — Multi-Agent Supervisor |

### M01 — Foundations of Agents
| File | Type |
|------|------|
| `1.0 Lecture - Foundations of AI Agents and Tools on Databricks.py` | Lecture |
| `1.1 Lecture - Unity Catalog Functions as Agent Tools.py` | Lecture |
| `1.2 Demo - Building UC Functions as Agent Tools with AI Playground.py` | Demo |
| `1.3 Lab - Building AI Agent Tools with Unity Catalog Functions.py` | Lab |

### M02 — Building Single Agents
| File | Type |
|------|------|
| `2.0 Lecture - Authoring Single AI Agents with Databricks Mosaic AI Agent Framework.py` | Lecture |
| `2.1 Demo - Building Single Agents with LangChain.py` | Demo |
| `2.2 Lab - Building A LangChain Agent.py` | Lab |

### M03 — Reproducible Agents
| File | Type |
|------|------|
| `3.0 Lecture - Building Agents on Databricks with MLflow.py` | Lecture |
| `3.1 Demo - Tracing Single Agents with MLflow.py` | Demo |
| `3.2 Demo - Tagging & Reproducible Agents.py` | Demo |
| `3.3 Lab - Building Reproducible Agents.py` | Lab |

### M04 — Production-Ready Agents with Agent Bricks
| File | Type |
|------|------|
| `4.0 Lecture - Single Agents with Agent Bricks.py` | Lecture |
| `4.1 Lab - Building a Single Agent with Agent Bricks: Multi-Agent Supervisor.py` | Lab |

---

## Project Structure

```
week-2-building-single-agent-applications-on-databricks-en_us-1.0.1/
├── README.md
└── building-single-agent-applications-on-databricks-en_us-1.0.1/
    └── Building Single-Agent Applications on Databricks/
        ├── AGENDA.py
        ├── M01 - Foundations of Agents/
        ├── M02 - Building Single Agents/
        ├── M03 - Reproducible Agents/
        ├── M04 - Production-Ready Agents with Agent Bricks/
        └── Includes/
            ├── config/                  # YAML configs per setup (e.g. setup-common-with-tools.yaml)
            ├── tools/                   # UC function definitions
            ├── _lib/
            │   ├── setup_orchestrator.py   # Main setup entry point (setup_demo_environment)
            │   ├── catalog_utils.py        # Catalog/schema resolution per user
            │   ├── config_loader.py        # YAML config loader
            │   ├── agent_manager.py        # Agent registration & deployment
            │   ├── artifacts_manager.py
            │   ├── experiment_manager.py
            │   └── ...
            └── Classroom-Setup-*.py     # Per-notebook setup scripts
```

## Setup

Each notebook runs a classroom setup script that calls `setup_demo_environment()`:

```python
from Includes import setup_demo_environment

env = setup_demo_environment(config_path="../Includes/config/setup-common-with-tools.yaml")
catalog_name = env["catalog_name"]   # Dynamically built: {prefix}_{username}
schema_name  = env["schema_name"]    # From YAML: "airbnb_agent"
```

### How catalog and schema names are resolved

| Key | Source |
|-----|--------|
| `catalog_name` | Built at runtime by `build_user_catalog()` as `labuser_<username>` (or `dbacademy` on Vocareum) |
| `schema_name` | YAML config field `catalog.schema_name` (default: `"airbnb_agent"`) |

### Key Python packages

```
databricks-langchain
langgraph >= 1.1.5
langgraph-prebuilt >= 1.0.9
langchain-community
mlflow-skinny[databricks]
databricks-agents
databricks-sdk
unitycatalog-langchain[databricks]
unitycatalog-ai[databricks]
```

---

## License

&copy; 2026 Databricks, Inc. All rights reserved.  
Apache, Apache Spark, Spark, and related marks are trademarks of the [Apache Software Foundation](https://www.apache.org/).
