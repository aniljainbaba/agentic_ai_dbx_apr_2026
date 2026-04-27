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
# MAGIC # Lecture - Types of Evaluation Judges
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC MLflow provides multiple types of evaluation judges, each designed for different evaluation scenarios and customization levels. This lecture explores the spectrum of available judges, from built-in research-validated assessments to fully custom evaluation logic.
# MAGIC
# MAGIC We'll examine built-in judges for common criteria, guideline judges for business rules, and custom approaches for specialized requirements. Understanding when and how to use each type is crucial for building comprehensive evaluation workflows.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Distinguish between different types of evaluation judges (built-in, guideline, custom)
# MAGIC - Identify appropriate built-in judges for common evaluation criteria
# MAGIC - Understand how to implement guideline judges for business rules
# MAGIC - Recognize when custom judges are necessary and how to implement them
# MAGIC - Understand the role of Feedback objects and rationales in evaluation results

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Judge Types Overview

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Evaluation Judge Spectrum
# MAGIC
# MAGIC | Approach              | Level of customization | Use cases |
# MAGIC |----------------------|------------------------|-----------|
# MAGIC | Built-in judges      | Minimal                | Quickly try LLM evaluation with built-in scorers such as `Correctness` and `RetrievalGroundedness`. |
# MAGIC | Guidelines judges    | Moderate               | A built-in judge that checks whether responses pass or fail custom natural-language rules, such as style or factuality guidelines. |
# MAGIC | Custom judges        | Full                   | Create fully customized LLM judges with detailed evaluation criteria and feedback optimization. Capable of returning numerical scores, categories, or boolean values. |
# MAGIC | Code-based scorers   | Full                   | Programmatic and deterministic scorers that evaluate things like exact matching, format validation, and performance metrics. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Built-In Judges for Common Criteria

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Research-Validated Judges
# MAGIC
# MAGIC MLflow provides research-validated judges for common evaluation tasks. These judges have been developed through extensive research, validated against human expert judgment, and optimized for specific evaluation criteria.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Core Built-in Judges
# MAGIC
# MAGIC **Correctness**  
# MAGIC Evaluates whether the model's response is factually correct compared to provided ground truth.  
# MAGIC Requires ground truth in the evaluation dataset via `expectations` (for example, an expected answer or expected facts).
# MAGIC
# MAGIC **RelevanceToQuery**  
# MAGIC Assesses whether the response directly and appropriately addresses the user's query.  
# MAGIC Useful for identifying off-topic, tangential, or irrelevant answers. Does **not** require ground truth.
# MAGIC
# MAGIC **RetrievalSufficiency**  
# MAGIC Determines whether the retrieved context contains all the information necessary to produce a correct response that includes the ground-truth facts.  
# MAGIC Requires ground truth (`expectations`) and evaluates retrieval quality rather than generation quality.
# MAGIC
# MAGIC **RetrievalRelevance**  
# MAGIC Evaluates whether the retrieved documents are relevant to the user's query.  
# MAGIC Does **not** require ground truth and focuses only on the retrieval step, independent of the final answer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Additional Built-in Judges
# MAGIC
# MAGIC **RetrievalGroundedness**  
# MAGIC Checks whether the model's response is grounded in the retrieved context and does not hallucinate unsupported facts.  
# MAGIC Does **not** require ground truth and evaluates alignment between the response and provided documents.
# MAGIC
# MAGIC **Safety**  
# MAGIC Assesses whether the response is free from harmful, offensive, or unsafe content.  
# MAGIC Does **not** require ground truth and is commonly used as a baseline content safety check.
# MAGIC
# MAGIC **Guidelines**  
# MAGIC Evaluates whether the response follows specified natural-language rules or constraints (for example, style, tone, or formatting requirements).  
# MAGIC Does **not** require ground truth.
# MAGIC
# MAGIC **ExpectationsGuidelines**  
# MAGIC Evaluates whether the response meets per-example natural-language criteria defined in the evaluation dataset.  
# MAGIC Does not require factual ground truth, but relies on example-specific guidelines provided in `expectations`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B4. Example Usage Pattern
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import Correctness
# MAGIC
# MAGIC correctness_eval = Correctness(
# MAGIC     model="databricks:/foundation-model-endpoint")
# MAGIC
# MAGIC correctness_results = mlflow.genai.evaluate(
# MAGIC     data=eval_dataset,
# MAGIC     predict_fn=lambda input: agent.predict({"input": input}),
# MAGIC     scorers=[correctness_eval],
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC This example demonstrates a common pattern for evaluating agents using the `Correctness` scorer with a Databricks foundation model endpoint.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ⚠️ Demo Checkpoint
# MAGIC <div style="border-left: 4px solid #ff9800; background: #fff3e0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <span style="font-size: 24px;"></span>
# MAGIC     <div>
# MAGIC       <strong style="color: #e65100; font-size: 1.1em;">Demo Checkpoint</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">Navigate to <strong>2.2 Demo - Using MLflow Built-In Judges</strong> to see these built-in judges in action. When you are finished, navigate back to this lecture notebook and continue learning.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Guideline Judges

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Two Types of Guideline Judges
# MAGIC
# MAGIC **1. Global Guidelines (`Guidelines` class)**
# MAGIC
# MAGIC Apply uniform criteria to all evaluations in your dataset. Global guidelines are ideal when you want to enforce consistent standards across all test cases, such as tone, style, or formatting requirements.
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import Guidelines
# MAGIC
# MAGIC tone_guidelines = Guidelines(
# MAGIC     name="professional_tone",
# MAGIC     guidelines=[
# MAGIC         "The response must use professional, business-appropriate language",
# MAGIC         "The response should avoid slang, colloquialisms, or overly casual phrasing",
# MAGIC         "The response must address the user respectfully"
# MAGIC     ],
# MAGIC     model="databricks:/foundation-model-endpoint"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Per-Row Guidelines
# MAGIC
# MAGIC **2. Per-Row Guidelines (`ExpectationsGuidelines` class)**
# MAGIC
# MAGIC Apply different criteria to each example, useful when different scenarios require different standards. Each row in your dataset includes its own specific guidelines in an `expectations` field.
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import ExpectationsGuidelines
# MAGIC
# MAGIC # Dataset contains per-row guidelines in expectations
# MAGIC # Example row:
# MAGIC # {
# MAGIC #   "input": "What's the refund policy?",
# MAGIC #   "output": "Our refund policy allows...",
# MAGIC #   "expectations": {
# MAGIC #     "guidelines": ["Must mention 30-day timeframe", "Must include receipt requirement"]
# MAGIC #   }
# MAGIC # }
# MAGIC
# MAGIC expected_guidelines = ExpectationsGuidelines(
# MAGIC     name="policy_requirements",
# MAGIC     model="databricks:/foundation-model-endpoint"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC This approach is ideal for testing diverse use cases where each input requires unique validation criteria.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Guideline Judge Advantages and Best Practices
# MAGIC
# MAGIC **Guideline judge advantages:**
# MAGIC
# MAGIC - **Domain expert accessibility**: Business stakeholders can write guidelines without coding
# MAGIC - **Rapid iteration**: Update evaluation criteria without code changes
# MAGIC - **Interpretability**: Natural language guidelines are self-documenting
# MAGIC - **Flexibility**: Express complex, context-dependent requirements that would be difficult to code
# MAGIC
# MAGIC **Tips for writing guidelines:**
# MAGIC
# MAGIC - Be specific and concrete rather than vague ("Response must cite the source document" vs. "Response should be credible")
# MAGIC - Write guidelines that can be objectively verified
# MAGIC - Focus on observable attributes in the response
# MAGIC - Test guidelines on multiple examples to ensure they work as intended

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ⚠️ Demo Checkpoint
# MAGIC <div style="border-left: 4px solid #ff9800; background: #fff3e0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <span style="font-size: 24px;"></span>
# MAGIC     <div>
# MAGIC       <strong style="color: #e65100; font-size: 1.1em;">Demo Checkpoint</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">Navigate to <strong>2.3 Demo - Guideline Judges with MLflow</strong> to explore guideline judges in practice. When you are finished, navigate back to this lecture notebook and continue learning.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ⚠️ Lab Checkpoint
# MAGIC <div style="border-left: 4px solid #ff9800; background: #fff3e0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <span style="font-size: 24px;"></span>
# MAGIC     <div>
# MAGIC       <strong style="color: #e65100; font-size: 1.1em;">Lab Checkpoint</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">Navigate to <strong>2.4 Lab - Applying Agent Evaluation</strong> to practice implementing comprehensive evaluation workflows. When you are finished, navigate back to this lecture notebook and continue learning.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Custom Judges and Code-Based Scorers

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Code-Based Scorers for Deterministic Evaluation
# MAGIC
# MAGIC When built-in judges don't meet your needs, MLflow supports custom evaluation logic through code-based scorers:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import scorer
# MAGIC from mlflow.entities import Feedback
# MAGIC
# MAGIC @scorer
# MAGIC def response_length(outputs):
# MAGIC     """Verify response length is appropriate."""
# MAGIC     word_count = len(str(outputs.get("response", "")).split())
# MAGIC
# MAGIC     if 20 <= word_count <= 100:
# MAGIC         return Feedback(
# MAGIC             value="yes",
# MAGIC             rationale=f"Response length ({word_count} words) is appropriate"
# MAGIC         )
# MAGIC     else:
# MAGIC         return Feedback(
# MAGIC             value="no",
# MAGIC             rationale=f"Response is too {'short' if word_count < 20 else 'long'} ({word_count} words)"
# MAGIC         )
# MAGIC ```
# MAGIC
# MAGIC Custom scorers allow you to implement domain-specific validation logic that goes beyond what built-in judges provide. The `@scorer` decorator converts your function into a reusable evaluation component.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Alternative Approach with Primitive Return
# MAGIC
# MAGIC For simpler use cases, scorers can return primitive values directly:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import scorer
# MAGIC
# MAGIC @scorer
# MAGIC def response_length(outputs):
# MAGIC     wc = len(str(outputs.get("response", "")).split())
# MAGIC     return "yes" if 20 <= wc <= 100 else "no"
# MAGIC ```
# MAGIC
# MAGIC This approach is more concise but doesn't provide rationale for the scoring decision, making it better suited for straightforward pass/fail checks.
# MAGIC
# MAGIC `@scorer` turns your plain Python function into an MLflow GenAI Scorer, which is a first-class, pluggable metric that `mlflow.genai.evaluate()` can run offline and that you can register for production monitoring later.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Custom LLM Judges
# MAGIC
# MAGIC **Custom LLM judges** for sophisticated evaluation:
# MAGIC
# MAGIC You can implement your own LLM-based judges for specialized evaluation criteria not covered by built-in judges. This involves:
# MAGIC 1. Designing evaluation prompts that clearly specify your criteria
# MAGIC 2. Calling an LLM to make judgments based on those prompts
# MAGIC 3. Parsing LLM responses into structured `Feedback` objects
# MAGIC 4. Wrapping this logic in a function compatible with `mlflow.genai.evaluate()`
# MAGIC
# MAGIC **When to use custom judges:**
# MAGIC
# MAGIC - You need domain-specific evaluation criteria unique to your application
# MAGIC - Built-in judges don't capture nuances important to your use case
# MAGIC - You're evaluating against proprietary standards or regulations
# MAGIC - You need evaluation logic that combines multiple signals or data sources
# MAGIC
# MAGIC Custom judges provide maximum flexibility while maintaining integration with MLflow's evaluation framework, tracing, and logging infrastructure.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ⚠️ Demo Checkpoint
# MAGIC <div style="border-left: 4px solid #ff9800; background: #fff3e0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <span style="font-size: 24px;"></span>
# MAGIC     <div>
# MAGIC       <strong style="color: #e65100; font-size: 1.1em;">Demo Checkpoint</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">Navigate to <strong>2.5 Demo - Custom Judges with MLflow</strong> to learn how to create custom judges for specialized evaluation needs. When you are finished, navigate back to this lecture notebook and continue learning.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Feedback Objects and Rationales

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. Structured Feedback Objects
# MAGIC
# MAGIC ```python
# MAGIC Feedback(
# MAGIC     value="yes",        # Binary pass/fail or numerical score
# MAGIC     rationale="The response correctly identifies the capital as Sacramento...",  # Explanation
# MAGIC     metadata={...}      # Additional structured information
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC MLflow judges return structured `Feedback` objects rather than simple scalar scores. This structure provides interpretability critical for understanding evaluation results.
# MAGIC
# MAGIC **Key components:**
# MAGIC - `value`: The actual score or judgment
# MAGIC - `rationale`: Human-readable explanation
# MAGIC - `metadata`: Additional context

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Why Rationales Matter
# MAGIC
# MAGIC - **Debugging**: Understand why an example failed, not just that it failed
# MAGIC - **Judge validation**: Verify that judges are reasoning correctly
# MAGIC - **Pattern identification**: Common rationale themes reveal systemic issues
# MAGIC - **Stakeholder communication**: Explain evaluation results to non-technical audiences
# MAGIC
# MAGIC **Using rationales effectively:**
# MAGIC
# MAGIC 1. Read rationales for all failures to identify patterns
# MAGIC 2. Spot-check rationales for passes to ensure judge is reasoning correctly
# MAGIC 3. Extract common rationale phrases to categorize failure types
# MAGIC 4. Share representative rationales when discussing evaluation with teams
# MAGIC
# MAGIC The combination of pass/fail scores with detailed rationales makes MLflow evaluation both quantitative (trackable metrics) and qualitative (understandable reasoning).

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Key Takeaways
# MAGIC
# MAGIC MLflow provides a comprehensive suite of evaluation judges to meet diverse evaluation needs:
# MAGIC
# MAGIC 1. **Built-in judges**: Research-validated assessments for common criteria like correctness, relevance, and safety
# MAGIC 2. **Guideline judges**: Business-rule evaluation through natural language guidelines, both global and per-example
# MAGIC 3. **Custom code-based scorers**: Deterministic evaluation logic for specific requirements
# MAGIC 4. **Custom LLM judges**: Sophisticated evaluation for specialized domain requirements
# MAGIC 5. **Structured feedback**: Rationales provide interpretability and debugging capabilities
# MAGIC
# MAGIC Choosing the right combination of judges depends on your specific evaluation requirements, available ground truth, and the level of customization needed. The next lectures will explore evaluation strategies and practical implementation approaches.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>