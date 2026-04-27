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
# MAGIC # Lecture - MLflow's Evaluation Framework
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC MLflow provides a comprehensive evaluation framework specifically designed for generative AI applications, offering automated judges, tracing capabilities, and systematic assessment tools. This lecture explores MLflow's architecture, core components, and how they work together to enable rigorous agent evaluation.
# MAGIC
# MAGIC We'll examine the three fundamental components of MLflow evaluation, understand the `mlflow.genai.evaluate()` function, and explore how tracing provides the foundation for sophisticated evaluation capabilities.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Describe MLflow's evaluation framework architecture and key components
# MAGIC - Understand the role of evaluation datasets, scorers, and predict functions
# MAGIC - Explain how the `mlflow.genai.evaluate()` function orchestrates evaluation
# MAGIC - Understand the role of tracing in agent evaluation and debugging
# MAGIC - Recognize how AI Gateway integration enables production monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. MLflow Overview and OpenTelemetry Integration

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. MLflow Platform Overview
# MAGIC
# MAGIC ![preparing-for-evaluation.png](../Includes/images/Evaluation with MLflow/preparing-for-evaluation.png "preparing-for-evaluation.png")
# MAGIC
# MAGIC MLflow is an open source platform for managing the full machine learning lifecycle, and it now natively supports OpenTelemetry for advanced observability and monitoring, allowing seamless tracing and metrics export to external observability platforms.
# MAGIC
# MAGIC **MLflow Core Features:**
# MAGIC - Experiment tracking with parameters, metrics, and model lineage
# MAGIC - Model Registry for versioning, stage transitions, and annotations (integrated with Unity Catalog)
# MAGIC - Model deployment for batch, streaming, and real-time inference
# MAGIC - Real-time tracing server (MLflow Tracing) for instant observability
# MAGIC - Production monitoring including automatic scoring of GenAI traces
# MAGIC - Deep support for prompt engineering and GenAI evaluation workflows

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. OpenTelemetry Integration
# MAGIC
# MAGIC **OpenTelemetry Integration Benefits:**
# MAGIC - OpenTelemetry is an open standard for telemetry data collection and export, widely adopted for observability across cloud-native systems
# MAGIC - MLflow traces are fully compatible with OpenTelemetry trace specifications, allowing export to popular solutions (e.g., Datadog, New Relic, Grafana, Splunk)
# MAGIC - MLflow supports three trace export modes:
# MAGIC   - MLflow tracking only (default): sends traces to MLflow Tracking Server
# MAGIC   - OpenTelemetry only: sends traces to an OpenTelemetry Collector
# MAGIC   - Dual export: sends traces to both MLflow and an OpenTelemetry Collector

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Core Architecture

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Three Fundamental Components
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC <img src="https://docs.databricks.com/aws/en/assets/images/flowchart-00c729ac75207b58d9c2243583a30d5a.png" alt="MLFlow Evaluation">
# MAGIC </div>
# MAGIC
# MAGIC MLflow provides a comprehensive evaluation framework designed specifically for generative AI applications. The architecture centers on three fundamental components:

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Component 1: Evaluation Datasets
# MAGIC
# MAGIC Your evaluation dataset defines what you're testing. At minimum, it contains inputs (queries or requests to your agent). Optionally, it can include:
# MAGIC
# MAGIC - **Outputs**: Pre-generated agent responses for faster evaluation without re-running inference
# MAGIC - **Expectations**: Ground truth information such as expected facts, expected responses, or per-row guidelines
# MAGIC - **Traces**: Complete execution traces for analyzing multi-step agent behavior
# MAGIC - **Metadata**: Additional context like user preferences, conversation history, or retrieved documents
# MAGIC
# MAGIC Datasets are typically stored as JSON files or Pandas DataFrames for easy manipulation and versioning.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Component 2: Scorers (Judges)
# MAGIC
# MAGIC Scorers evaluate your agent's outputs against defined criteria. MLflow provides multiple scorer types:
# MAGIC
# MAGIC - **Built-in judges**: Research-validated LLM-based assessments for common criteria like correctness, relevance, and safety
# MAGIC - **Guideline judges**: Custom business rules expressed in natural language
# MAGIC - **Code-based scorers**: Python functions for deterministic evaluation (length checks, format validation, etc.)
# MAGIC - **Custom LLM judges**: Your own LLM-based evaluation logic for specialized requirements

# COMMAND ----------

# MAGIC %md
# MAGIC ### B4. Component 3: Predict Function
# MAGIC
# MAGIC The predict function generates outputs for your evaluation dataset. This can be:
# MAGIC
# MAGIC - Your agent's prediction method (for evaluating on-the-fly)
# MAGIC - A lambda that transforms inputs to the format your agent expects
# MAGIC - Omitted entirely if you're evaluating pre-generated outputs
# MAGIC
# MAGIC These three components come together in `mlflow.genai.evaluate()`, which orchestrates the evaluation process, collects metrics, and logs comprehensive results for analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. The `mlflow.genai.evaluate()` Function

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Central Orchestration Point
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from mlflow.genai.scorers import Correctness
# MAGIC
# MAGIC results = mlflow.genai.evaluate(
# MAGIC     data=eval_dataset,                  # DataFrame, list[dict], or EvaluationDataset
# MAGIC     scorers=[Correctness()],            # Built-in and/or custom scorers
# MAGIC     predict_fn=my_app,                  # Optional: direct evaluation
# MAGIC     # model_id="models:/my-app/1",      # Optional: link to versioned app
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC The `mlflow.genai.evaluate()` function serves as the central orchestration point for agent evaluation. Understanding its behavior is critical for effective evaluation workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Key Parameters
# MAGIC
# MAGIC **Key parameters:**
# MAGIC - `data`: Your evaluation dataset
# MAGIC - `scorers`: List of scoring functions
# MAGIC - `predict_fn`: Your agent/app function
# MAGIC - `model_id`: Optional model reference

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Evaluation Workflow
# MAGIC
# MAGIC **Evaluation workflow:**
# MAGIC
# MAGIC 1. **Data loading**: MLflow loads your evaluation dataset and validates its structure
# MAGIC 2. **Output generation**: If `predict_fn` is provided, MLflow calls it for each input to generate outputs
# MAGIC 3. **Trace creation**: Each prediction creates an MLflow trace when your predict_fn is instrumented (e.g., with `@mlflow.trace` or `mlflow.openai.autolog`) or when evaluating endpoints with `mlflow.genai.to_predict_fn`. For "answer sheet" mode, evaluate constructs traces from inputs/outputs even without running the app
# MAGIC 4. **Scorer execution**: Each scorer evaluates the inputs/outputs/traces according to its logic
# MAGIC 5. **Result aggregation**: Individual scores are aggregated into summary metrics
# MAGIC 6. **Logging**: Results are logged to MLflow for analysis and comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ### C4. Return Value and Results Access
# MAGIC
# MAGIC **Return value:**
# MAGIC
# MAGIC The function returns an `EvaluationResult` object containing:
# MAGIC - **run_id**: Unique identifier for this evaluation run
# MAGIC - **metrics**: Aggregated metrics across all examples (e.g., average score, pass rate)
# MAGIC
# MAGIC **Accessing per-example results:**
# MAGIC
# MAGIC In MLflow 3, per-example results are accessed via traces rather than a result_df attribute:
# MAGIC ```python
# MAGIC eval_traces = mlflow.search_traces(run_id=results.run_id)
# MAGIC ```
# MAGIC
# MAGIC This structured approach enables systematic evaluation while maintaining full observability into individual agent interactions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. MLflow Tracing: The Foundation of Agent Observability

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Comprehensive Observability
# MAGIC
# MAGIC MLflow Tracing provides comprehensive observability into agent execution, capturing every step of your agent's reasoning process. Tracing is enabled when using `mlflow.genai.evaluate()` with properly instrumented predict functions and forms the foundation for many evaluation capabilities.
# MAGIC
# MAGIC **What tracing captures:**
# MAGIC
# MAGIC - **Model calls**: Every interaction with foundation models, including prompts, responses, and model parameters
# MAGIC - **Tool invocations**: Function calls with input parameters and return values
# MAGIC - **Retrieval operations**: Documents retrieved from vector stores, with content and metadata
# MAGIC - **Timing information**: Duration of each operation for performance analysis
# MAGIC - **Hierarchical structure**: Parent-child relationships showing the execution flow

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Span Types in Traces
# MAGIC
# MAGIC MLflow organizes traces into spans with specific types:
# MAGIC - **Root span**: Top-level span representing the complete agent invocation
# MAGIC - **RETRIEVER**: Spans where documents are fetched from vector search or other retrieval systems
# MAGIC - **TOOL**: Individual tool or function calls
# MAGIC - **CHAT_MODEL**: Language model interactions
# MAGIC - **CHAIN**: Sequences of operations (common in LangChain-based agents)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Why Tracing Matters for Evaluation
# MAGIC
# MAGIC **Why tracing matters for evaluation:**
# MAGIC
# MAGIC Certain evaluation judges, like `RetrievalSufficiency`, require traces to function. They analyze what was retrieved (not just the final response) to assess whether the retrieval system provided adequate information. Without traces, these sophisticated evaluations would be impossible.
# MAGIC
# MAGIC Tracing also enables debugging by allowing you to inspect exactly what happened during a failed evaluation, identifying whether issues stem from retrieval quality, tool selection, or LLM reasoning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. AI Gateway Integration and Production Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. AI Gateway Integration
# MAGIC
# MAGIC When you register agents using the Mosaic AI Agent Framework and deploy them to Model Serving, Databricks automatically enables AI Gateway-enhanced inference tables. These tables provide detailed logging of all requests and responses in production.
# MAGIC
# MAGIC **Inference table benefits:**
# MAGIC
# MAGIC - **Automatic logging**: Every request to your deployed agent is captured without additional instrumentation
# MAGIC - **Rich metadata**: Includes request/response content, timestamps, latency, model versions, and trace data
# MAGIC - **Query interface**: SQL-queryable tables in Unity Catalog for analysis and monitoring
# MAGIC - **Evaluation integration**: Inference table data can be directly used as evaluation datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. From Production to Evaluation
# MAGIC
# MAGIC This integration creates a powerful feedback loop:
# MAGIC 1. Agents run in production and log all interactions to inference tables
# MAGIC 2. You query inference tables to extract interesting examples, failure cases, or edge cases
# MAGIC 3. These real-world examples augment your evaluation dataset
# MAGIC 4. Future evaluations test against actual production scenarios
# MAGIC
# MAGIC This approach ensures your evaluation dataset evolves with real user behavior rather than remaining static and potentially stale.

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Key Takeaways
# MAGIC
# MAGIC MLflow's evaluation framework provides a comprehensive solution for agent evaluation through:
# MAGIC
# MAGIC 1. **Three-component architecture**: Evaluation datasets, scorers, and predict functions work together seamlessly
# MAGIC 2. **Central orchestration**: `mlflow.genai.evaluate()` handles the complexity of evaluation workflows
# MAGIC 3. **Comprehensive tracing**: Full observability into agent execution enables sophisticated evaluation and debugging
# MAGIC 4. **Production integration**: AI Gateway and inference tables create feedback loops between production and evaluation
# MAGIC 5. **OpenTelemetry compatibility**: Integration with industry-standard observability tools
# MAGIC
# MAGIC This foundation enables systematic, scalable evaluation that grows with your agent development needs. The next lectures will explore the specific types of judges and evaluation strategies you can implement within this framework.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>