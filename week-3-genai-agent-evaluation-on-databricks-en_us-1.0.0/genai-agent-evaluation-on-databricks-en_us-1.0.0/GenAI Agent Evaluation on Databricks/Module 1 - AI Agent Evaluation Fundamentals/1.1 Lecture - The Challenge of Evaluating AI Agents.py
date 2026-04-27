# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture - The Challenge of Evaluating AI Agents
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Traditional software testing approaches are fundamentally insufficient for AI agents due to their non-deterministic nature, emergent behaviors, and context-dependent responses. This lecture explores why conventional testing fails with AI agents and introduces the unique challenges that require specialized evaluation frameworks.
# MAGIC
# MAGIC We'll examine how AI agents break traditional testing paradigms, understand the complexity of multi-step reasoning evaluation, and learn why evaluation must be treated as a continuous process rather than a one-time validation step.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Explain why traditional software testing approaches are insufficient for AI agents
# MAGIC - Identify the key challenges unique to evaluating AI agents (non-determinism, emergent behavior, context dependency)
# MAGIC - Understand why evaluation must be treated as a continuous process
# MAGIC - Recognize the importance of proper evaluation dataset design
# MAGIC - Describe the operational setup requirements for systematic agent evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Why Traditional Testing Falls Short

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. The Deterministic Testing Paradigm
# MAGIC
# MAGIC Traditional software testing relies on deterministic inputs and expected outputs. You write unit tests, integration tests, and end-to-end tests that verify your code produces the exact same result every time given the same input. This approach works well for classical software systems where behavior is predictable and reproducible.
# MAGIC
# MAGIC **Traditional testing assumptions:**
# MAGIC - Same input always produces same output
# MAGIC - Behavior is explicitly programmed and predictable
# MAGIC - Success can be measured through exact string matching or numerical comparisons
# MAGIC - Edge cases can be anticipated and tested systematically

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. How AI Agents Break the Paradigm
# MAGIC
# MAGIC **AI agents fundamentally break this paradigm:**
# MAGIC
# MAGIC - **Non-determinism**: LLMs introduce randomness through temperature and sampling, meaning the same input can produce different outputs
# MAGIC - **Emergent behavior**: Agents make autonomous decisions about tool usage and reasoning paths that weren't explicitly programmed
# MAGIC - **Context dependency**: Agent responses depend on retrieved documents, conversation history, and external data sources
# MAGIC - **Qualitative assessment**: Success often requires subjective judgment about helpfulness, tone, or appropriateness rather than exact string matching
# MAGIC
# MAGIC Consider a simple example: An agent answering "What's the weather in San Francisco?" might respond:
# MAGIC - "It's currently 65°F and sunny in San Francisco."
# MAGIC - "San Francisco weather: 65 degrees, clear skies."
# MAGIC - "The temperature in SF is 65°F with no clouds."
# MAGIC
# MAGIC All three responses are correct, helpful, and appropriate—yet none match exactly. Traditional assertion-based testing (`assert output == "expected_response"`) would fail on all three.

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. The Agent Evaluation Challenge

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Multi-Dimensional Complexity
# MAGIC
# MAGIC ![the-agent-evaluation-challenge.png](../Includes/images/Evaluation with MLflow/the-agent-evaluation-challenge.png)
# MAGIC
# MAGIC Evaluating AI agents introduces unique complexities beyond traditional software:
# MAGIC
# MAGIC **Multi-step reasoning**: Agents may invoke multiple tools, retrieve various documents, and build complex reasoning chains. Evaluation must assess not just the final answer but the quality of intermediate steps.
# MAGIC
# MAGIC **Tool calling accuracy**: Did the agent select the right tools? Did it pass appropriate parameters? Did it correctly interpret tool results?
# MAGIC
# MAGIC **Retrieval quality**: For RAG-based agents, evaluation must verify that retrieved documents contain relevant information and that the agent correctly synthesizes information from multiple sources.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Safety and Real-World Variability
# MAGIC
# MAGIC **Safety and alignment**: Agents must avoid harmful outputs, respect user boundaries, and decline inappropriate requests—qualities that require sophisticated evaluation beyond simple pass/fail tests.
# MAGIC
# MAGIC **Real-world variability**: Production agents encounter diverse user queries, unexpected phrasings, and edge cases that are difficult to anticipate during development.
# MAGIC
# MAGIC These challenges demand a more sophisticated evaluation framework specifically designed for the probabilistic, contextual nature of AI agents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Evaluation as a Continuous Process

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. The Continuous Evaluation Cycle
# MAGIC
# MAGIC ![evaluation-as-continuous-process.png](../Includes/images/Evaluation with MLflow/evaluation-as-continuous-process.png "evaluation-as-continuous-process.png")
# MAGIC
# MAGIC Unlike traditional software where comprehensive test suites provide stable quality signals, AI agent evaluation is an ongoing process:
# MAGIC
# MAGIC **Development phase**: Rapid iteration requires frequent evaluation to validate that changes improve quality without introducing regressions.
# MAGIC
# MAGIC **Pre-deployment validation**: Comprehensive evaluation across diverse test cases ensures agents meet quality bars before production release.
# MAGIC
# MAGIC **Production monitoring**: Continuous evaluation of live interactions identifies quality degradation, emerging failure patterns, and opportunities for improvement.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Why Continuous Evaluation Matters
# MAGIC
# MAGIC This continuous evaluation cycle means your evaluation infrastructure must be scalable, automated, and integrated into your development workflow. The evaluation framework must support:
# MAGIC
# MAGIC - **Rapid feedback loops** during development
# MAGIC - **Comprehensive validation** before deployment
# MAGIC - **Ongoing monitoring** in production
# MAGIC - **Dataset evolution** as usage patterns change

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Preparing for Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Why Preparation Matters
# MAGIC
# MAGIC Before using any evaluation framework, it's important to make evaluation both purposeful and reproducible. AI agents are non-deterministic and context-dependent, so traditional assertion-style tests fall short. Instead, effective evaluation requires defining quality dimensions, assembling representative datasets, and enabling tracing so judges can assess not just answers, but how those answers were produced.
# MAGIC
# MAGIC By clarifying goals, curating datasets, and selecting appropriate judges up front, you create a feedback loop where metrics reflect real user needs and failures are diagnosable through traces and rationales—not just pass/fail scores.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Designing Your Evaluation Dataset
# MAGIC
# MAGIC ![designing-evaluation-datasets.png](../Includes/images/Evaluation with MLflow/designing-evaluation-datasets.png "designing-evaluation-datasets.png")
# MAGIC
# MAGIC Your evaluation dataset defines what you test and how trustworthy your signals will be. At minimum, it should include inputs (queries) and, where appropriate, expected answers, per-row guidelines, and metadata that reflects retrieval and tool usage.
# MAGIC
# MAGIC **Key principles:**
# MAGIC
# MAGIC - **Representativeness:** Include common, high-impact user queries so offline results generalize to production
# MAGIC - **Edge cases:** Add ambiguous, out-of-scope, and adversarial prompts to surface failure modes early
# MAGIC - **Diversity:** Vary length, complexity, domain, and user expertise to expose blind spots in reasoning and retrieval
# MAGIC - **Ground truth and/or guidelines:** Use expected answers or fact sets for objective questions; use natural-language guidelines where style, policy, or completeness matter
# MAGIC - **Storage and versioning:** Store datasets as JSON or DataFrames (ideally in Delta/Unity Catalog) so they evolve alongside your agent

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Operational Setup Requirements
# MAGIC
# MAGIC Establish consistent scaffolding so results are comparable and auditable:
# MAGIC
# MAGIC - **MLflow experiments and runs:** Use stable experiment names; tag runs with agent version, dataset version, and parameters; compare metrics and inspect per-example artifacts in the UI
# MAGIC - **Unity Catalog integration:** Govern datasets and traces with access control, versioning, and lineage; register agents and dependencies for end-to-end traceability
# MAGIC - **Production feedback loop (plan ahead):** Once deployed, enable AI Gateway inference tables to log requests, responses, and traces for monitoring and for mining new evaluation examples

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Key Takeaways
# MAGIC
# MAGIC Traditional software testing approaches are fundamentally inadequate for AI agents due to their non-deterministic, emergent, and context-dependent nature. Effective agent evaluation requires:
# MAGIC
# MAGIC 1. **Recognition of unique challenges**: Non-determinism, emergent behavior, and context dependency require specialized evaluation approaches
# MAGIC 2. **Continuous evaluation mindset**: Evaluation is an ongoing process, not a one-time validation step
# MAGIC 3. **Proper preparation**: Success depends on thoughtful dataset design, clear quality definitions, and robust operational setup
# MAGIC 4. **Systematic approach**: Evaluation infrastructure must be scalable, automated, and integrated into development workflows
# MAGIC
# MAGIC Understanding these challenges is the foundation for implementing effective agent evaluation. The next lectures will explore the tools and techniques that address these challenges systematically.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ⚠️ Demo Checkpoint
# MAGIC <div style="border-left: 4px solid #ff9800; background: #fff3e0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <span style="font-size: 24px;"></span>
# MAGIC     <div>
# MAGIC       <strong style="color: #e65100; font-size: 1.1em;">Demo Checkpoint</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">Navigate to the first Demo titled <strong>01 Demo - Agent Setup</strong> and initiate the setup of your agent and UC assets that will be used throughout this training. When you are finished, navigate back to this lecture notebook and continue learning.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>