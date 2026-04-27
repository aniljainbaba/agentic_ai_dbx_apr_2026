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

# MAGIC %md-sandbox
# MAGIC # Demo - Using MLflow Built-In Judges
# MAGIC **Overview** 
# MAGIC
# MAGIC This demonstration explores MLflow's built-in judges for evaluating AI agents. Built-in judges provide research-validated, automated evaluation capabilities that help assess the quality and correctness of AI agent responses without requiring manual review.
# MAGIC
# MAGIC In this demo, you will learn how to leverage MLflow's evaluation framework to systematically assess your agent's performance using standardized metrics. We'll focus on two key evaluation dimensions: correctness and safety, demonstrating how these judges can provide objective quality assessments for your AI applications.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this demonstration, you will be able to:
# MAGIC
# MAGIC 1. **Understand LLM judges** and their role in automated agent evaluation
# MAGIC 2. **Configure and use MLflow's built-in judges** for correctness and safety assessment
# MAGIC 3. **Execute comprehensive evaluations** using `mlflow.genai.evaluate()` with multiple scorers
# MAGIC 4. **Interpret evaluation results** and leverage MLflow's tracing capabilities for detailed analysis
# MAGIC 5. **Apply best practices** for separating evaluation models and configuration management
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Prerequisites</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"> This demo uses the agent created in <strong>01 - Agent Setup</strong>. Please ensure you have completed that notebook before proceeding.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT SERVERLESS COMPUTE
# MAGIC
# MAGIC Before executing cells in this notebook, attach the notebook to **Serverless compute**.
# MAGIC
# MAGIC **NOTE:** This demo was tested on **Serverless (version 5)**.  
# MAGIC To confirm or change your Serverless version, see the Databricks documentation on Serverless dependencies.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Compute Requirements
# MAGIC
# MAGIC This course has been configured to run on Serverless compute. While classic compute may also work, testing has been performed on serverless.
# MAGIC
# MAGIC **Serverless compute should be on version 5 for this demo.** To ensure you are using the correct version, please [see this documentation on viewing and changing your notebook's Serverless version](https://docs.databricks.com/aws/en/compute/serverless/dependencies).
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Required - Select Serverless Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">You must attach this notebook to a Serverless compute resource before proceeding.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classroom Setup
# MAGIC
# MAGIC Run the following cell to configure your working environment for this course.
# MAGIC
# MAGIC This setup will:
# MAGIC - Initialize the `DA` object (Databricks Academy helper)
# MAGIC - Configure your **default catalog** and **schema**
# MAGIC - Provision any supporting configuration needed for this demo
# MAGIC
# MAGIC **NOTE:** The `DA` object is only available in Databricks Academy courses

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Configure Dataset Access
# MAGIC
# MAGIC This demonstration relies on the Airbnb dataset from Databricks Marketplace.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Dataset Information</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">All Unity Catalog and Workspace configurations have been properly setup and tested as a part of running <strong>01 Demo - Agent Setup </strong>.</code></p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1. Understanding LLM Judges

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1. What is an LLM Judge?
# MAGIC
# MAGIC LLM judges are a type of MLflow `Scorer` that uses LLMs for quality assessment. A `scorer` is a key component of MLflow's agent evaluation framework. It provides a unified interface to define evaluation criteria for models, agents, and applications.
# MAGIC
# MAGIC **Key characteristics of LLM judges:**
# MAGIC - **Automated evaluation** without human intervention
# MAGIC - **Research-validated** assessment criteria
# MAGIC - **Structured feedback** with both scores and rationales
# MAGIC - **Scalable** for large-scale evaluation workflows

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2. How is a scorer different than traditional ML metrics?
# MAGIC
# MAGIC A scorer is more flexible and can return more structured quality feedback in addition to scalar values that are typically represented as metrics in the classic ML sense.
# MAGIC
# MAGIC **Traditional ML Metrics:**
# MAGIC - Return single numerical values (accuracy, F1-score, RMSE)
# MAGIC - Focus on statistical performance
# MAGIC - Limited context about why a prediction failed
# MAGIC
# MAGIC **MLflow Scorers:**
# MAGIC - Return structured `Feedback` objects with values and rationales
# MAGIC - Assess qualitative aspects like relevance, correctness, safety
# MAGIC - Provide detailed explanations for evaluation decisions
# MAGIC - Support both binary ("yes"/"no") and continuous scoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3. How is quality measured?
# MAGIC
# MAGIC Databricks continuously improves judge quality through:
# MAGIC - **Research validation** against human expert judgment
# MAGIC - **Metrics tracking** using [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa), accuracy, F1 score
# MAGIC - **Diverse testing** on academic and real-world datasets
# MAGIC
# MAGIC **Quality assurance process:**
# MAGIC 1. **Human baseline establishment** - Expert annotators create ground truth evaluations
# MAGIC 2. **Judge performance measurement** - Compare judge outputs against human evaluations
# MAGIC 3. **Continuous improvement** - Regular updates based on performance metrics and new research
# MAGIC 4. **Cross-validation** - Testing across different domains and use cases

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1.4. What's the difference between `mlflow.genai.judges` and `mlflow.genai.scorers`?
# MAGIC
# MAGIC You can think of the difference between these two Python modules as `judges` are a subset of scorers, where LLM judges act as specialized scorers focused on LLM-based quality assessments. In contrast, a scorer orchestrates evaluation by handling data extraction and routing to the appropriate judge or algorithm. It's often the case that you wrap a judge inside a scorer for automated, trace-level evaluation.
# MAGIC
# MAGIC **Module hierarchy:**
# MAGIC - **`mlflow.genai.scorers`** - High-level evaluation interface that includes both LLM judges and other scoring methods
# MAGIC - **`mlflow.genai.judges`** - Specific LLM-based evaluation functions (subset of scorers)
# MAGIC
# MAGIC **When to use each:**
# MAGIC - Use **scorers** for integration with `mlflow.genai.evaluate()` and automated workflows
# MAGIC - Use **judges** directly for single-instance evaluation or custom scoring logic
# MAGIC
# MAGIC Since this demo will concentrate on `mlflow.genai.scorers`, it's important to note that each evaluation (for the `scorers` module) is defined by 3 components:
# MAGIC - **Dataset**: Inputs and expectations (and optionally pre-generated outputs and traces)
# MAGIC - **Scorer**: Evaluation criteria
# MAGIC - **Predict Function**: Generates outputs for the dataset
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC         You can read more about MLflow's GenAI evaluation system
# MAGIC         <a
# MAGIC           href="https://mlflow.org/docs/latest/genai/eval-monitor/#running-an-evaluation"
# MAGIC           target="_blank"
# MAGIC           rel="noopener noreferrer"
# MAGIC           style="color: #1976d2; text-decoration: underline;"
# MAGIC         >
# MAGIC           here
# MAGIC         </a>.
# MAGIC       </p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2. Built-In Judges Overview
# MAGIC MLflow provides research-validated judges for common use cases like input/output guidelines and correctness. In this demonstration, we will cover a few examples. For completeness, we provide a list for available scorers based on type. For a complete list of built-in judges, please see [this documentation](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers#built-in-llm-judges).
# MAGIC
# MAGIC ### Single-Turn Scorers
# MAGIC A single-turn scorer evaluates the quality, correctness, or relevance of the model's response to a single input or prompt from a user. Each input-response pair (or turn) is scored independently; there is no consideration of previous context or subsequent conversation turns.
# MAGIC | Judge | Arguments | Requires Ground Truth | What It Evaluates |
# MAGIC |------|-----------|-----------------------|------------------|
# MAGIC | RelevanceToQuery | `inputs`, `outputs` | No | Determines whether the response is directly relevant to the user's request. |
# MAGIC | RetrievalRelevance | `inputs`, `outputs` | No | Evaluates whether the retrieved context is directly relevant to the user's request. |
# MAGIC | Safety | `inputs`, `outputs` | No | Checks whether the content is free from harmful, offensive, or toxic material. |
# MAGIC | RetrievalGroundedness | `inputs`, `outputs` | No | Assesses whether the response is grounded in the provided context or if the agent is hallucinating. |
# MAGIC | Correctness | `inputs`, `outputs`, `expectations` | Yes | Determines whether the response is correct compared to the provided ground truth. |
# MAGIC | RetrievalSufficiency | `inputs`, `outputs`, `expectations` | Yes | Evaluates whether the retrieved context contains all necessary information to produce a response that includes the ground truth facts. |
# MAGIC | Guidelines | `inputs`, `outputs` | No | Checks whether the response meets specified natural-language guidelines. |
# MAGIC | ExpectationsGuidelines | `inputs`, `outputs`, `expectations` | No (guidelines required in `expectations`) | Evaluates whether the response meets per-example natural-language criteria defined in the expectations. |
# MAGIC
# MAGIC
# MAGIC ### Multi-Turn Scorers
# MAGIC
# MAGIC Multi-turn scorers evaluate entire conversation sessions rather than individual turns. They require traces with session IDs and are experimental in MLflow 3.7.0.
# MAGIC
# MAGIC | Scorer | What It Evaluates | Requires Session |
# MAGIC |-------|------------------|------------------|
# MAGIC | ConversationCompleteness | Determines whether the agent addresses all user questions throughout the conversation. | Yes |
# MAGIC | ConversationalRoleAdherence | Evaluates whether the assistant maintains its assigned role consistently throughout the conversation. | Yes |
# MAGIC | ConversationalSafety | Checks whether the assistant's responses are safe and free from harmful or inappropriate content. | Yes |
# MAGIC | ConversationalToolCallEfficiency | Assesses whether tool usage across the conversation was efficient, appropriate, and non-redundant. | Yes |
# MAGIC | KnowledgeRetention | Evaluates whether the assistant correctly retains and applies information from earlier user inputs. | Yes |
# MAGIC | UserFrustration | Determines whether the user exhibits frustration during the conversation and whether that frustration was effectively resolved. | Yes |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3. Examples with Built-In Judges

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Example 1: Correctness Evaluation
# MAGIC
# MAGIC Here we will look at using the `Correctness` judge. It will assess whether your application's response is factually correct by comparing it against provided ground truth information, which is defined as `expected_facts` or `expected_response`. Note that we will be evaluating the output from our custom agent built in the previous demo and not pass pre-generated outputs.
# MAGIC
# MAGIC **The Correctness judge evaluates:**
# MAGIC - Whether the response contains all expected facts
# MAGIC - If the response contradicts any provided ground truth
# MAGIC - The overall factual accuracy of the generated content
# MAGIC
# MAGIC **Input requirements:**
# MAGIC - **Ground truth data** - Either `expected_facts` (list of factual statements) or `expected_response` (complete expected answer)
# MAGIC - **Response data** - The actual output from your agent or model
# MAGIC - **Optional context** - Additional information that might inform the evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.1 Create Correctness Judge Instance
# MAGIC
# MAGIC Create an instance of the `Correctness` class called `correctness_eval`. By default, built-in judges use a Databricks-hosted LLM optimized for evaluation tasks, but you can specify a different model using the `model` parameter.
# MAGIC
# MAGIC **Model specification format:** `<provider>:/<model-name>`
# MAGIC - For Databricks models: `databricks:/model-endpoint-name`
# MAGIC - For OpenAI models: `openai/gpt-4o`
# MAGIC - For other LiteLLM-compatible providers: `provider/model-name`

# COMMAND ----------

from mlflow.genai.scorers import Correctness

correctness_eval = Correctness(
    model=correctness_eval_endpoint)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### 3.1.2 Load Evaluation Dataset
# MAGIC
# MAGIC Next, read in the evaluation dataset, which is stored in our volume `agent_vol` as `correctness_eval`. Note that this will have two different data points for evaluation. Of course, you should use more in your own projects.
# MAGIC
# MAGIC **Dataset structure for correctness evaluation:**
# MAGIC - **inputs** - The query or request sent to your agent
# MAGIC - **outputs** - The response generated by your agent (optional for some evaluation modes)
# MAGIC - **expectations** - Ground truth information containing either:
# MAGIC   - `expected_facts`: List of factual statements that should be present
# MAGIC   - `expected_response`: Complete expected answer for comparison
# MAGIC
# MAGIC Running the next cell will display the evaluation dataset.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #0d47a1; font-size: 1.1em;">
# MAGIC         Understanding Other Evaluation Data Types
# MAGIC       </strong>
# MAGIC       <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC         <li>
# MAGIC           For a list of data formats for direct evaluation, see
# MAGIC           <a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-harness#data-formats-for-direct-evaluation">
# MAGIC             this documentation
# MAGIC           </a>.
# MAGIC         </li>
# MAGIC         <li>
# MAGIC           For a list of data formats for answer sheet evaluation, see
# MAGIC           <a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-harness#data-formats-for-answer-sheet-evaluation">
# MAGIC             this documentation
# MAGIC           </a>.
# MAGIC         </li>
# MAGIC         <li>
# MAGIC           For further reading on using an MLflow evaluation dataset with the SDK, see
# MAGIC           <a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-datasets#mlflow-evaluation-dataset-sdk-reference">
# MAGIC             this documentation
# MAGIC           </a>.
# MAGIC         </li>
# MAGIC       </ul>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

import json 
from pprint import pprint
from pathlib import Path

path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/correctness_eval.json")

with path.open("r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

pprint(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.3 Execute Correctness Evaluation
# MAGIC
# MAGIC Now evaluate the dataset `eval_dataset`. Let's break down the input parameters for `mlflow.genai.evaluate()` as shown in the next cell:
# MAGIC - **data**: This is the evaluation dataset you are exposing to your scorer.
# MAGIC - **predict_fn**: This uses a lambda function to call the loaded agent from above (`agent`) using its predefined `predict` method.
# MAGIC - **scorers**: This is the list of evaluations you wish to assess. In this case, we are only interested in `correctness_eval`, which was instantiated above using the `Correctness` class from `mlflow.genai.scorers`.
# MAGIC
# MAGIC **Evaluation process:**
# MAGIC 1. **Data iteration** - MLflow processes each item in the evaluation dataset
# MAGIC 2. **Prediction generation** - Your agent generates responses for each input
# MAGIC 3. **Trace creation** - MLflow automatically creates traces for each prediction
# MAGIC 4. **Judge evaluation** - The Correctness judge compares responses against expected facts, which you can find in `<catalog_name>.<schema_name>.agent_vol`
# MAGIC 5. **Result aggregation** - Individual scores are compiled into summary metrics
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - After running the next cell, your output will include a button to view the evaluation results with MLflow. Click on it.
# MAGIC - You will be taken to your **MLflow Experiments** with a screen that resembles the following.
# MAGIC
# MAGIC ![mlflow-evaluation-runs.png](../Includes/images/built-in agents with mlflow/mlflow-evaluation-runs.png "mlflow-evaluation-runs.png")
# MAGIC - If you click on the latest run, the trace will appear with more details. Below is an example. You can see the feedback and expected output to help explain why the evaluation was returned as **Pass** or **Fail**. In general, this occurs with all metrics.
# MAGIC
# MAGIC ![mlflow-evaluation-runs2.png](../Includes/images/built-in agents with mlflow/mlflow-evaluation-runs2.png "mlflow-evaluation-runs2.png")
# MAGIC - You will also see a trace showing the request from the query

# COMMAND ----------

correctness_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=[correctness_eval],
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.4 Inspecting Correctness Results
# MAGIC
# MAGIC Results is an **EvaluationResult** object, which contains the run ID, aggregated metrics, and a dataframe called `results_df`, which is a per-row summary Pandas DataFrame for further inspection.
# MAGIC
# MAGIC **EvaluationResult components:**
# MAGIC - **run_id** - Unique identifier for this evaluation run in MLflow
# MAGIC - **metrics** - Aggregated performance metrics (e.g., average correctness score)
# MAGIC - **result_df** - Detailed per-example results with individual scores and rationales
# MAGIC
# MAGIC **Understanding the results:**
# MAGIC - **Value** - "yes" indicates correct response, "no" indicates incorrect
# MAGIC - **Rationale** - Detailed explanation of the judge's decision
# MAGIC - **Trace links** - Click on any row to view the complete MLflow trace
# MAGIC
# MAGIC **Instructions:**
# MAGIC After running the next cell, click on any of the rows and you will see the MLflow trace appear for further inspection.

# COMMAND ----------

print(f"The run ID is: {correctness_results.run_id}")
print(f"The aggregated metrics are: {correctness_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(getattr(correctness_results, "result_df", None))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 3.2. Example 2: Safety Evaluation
# MAGIC
# MAGIC Safety evaluation measures whether the content is free from harmful, offensive, or toxic material. This judge is particularly valuable for applications where you need to ensure your responses meet safety guidelines and don't contain inappropriate content.
# MAGIC
# MAGIC **The Safety judge evaluates:**
# MAGIC - Whether content contains harmful, offensive, or inappropriate material
# MAGIC - If responses meet safety guidelines and standards
# MAGIC - The overall safety and appropriateness of generated content
# MAGIC
# MAGIC **Requirements for Safety:**
# MAGIC - **Input and output content** - The judge examines both user inputs and agent responses
# MAGIC - **No ground truth required** - Safety evaluation doesn't require expected responses
# MAGIC - **Content analysis** - Judge examines text for safety violations and inappropriate material
# MAGIC
# MAGIC We can apply the same type of workflow as above to this metric by importing the `Safety` class and again using `mlflow.genai.evaluate()`
# MAGIC
# MAGIC **Instructions:**
# MAGIC Run the next two cells and again inspect the output by clicking on **View evaluation results in MLflow**. Which outputs failed the Safety metric? Read the feedback to understand the reasoning.

# COMMAND ----------

from mlflow.genai.scorers import Safety

safety_eval = Safety(
    model=safety_eval_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.1 Load Safety Evaluation Dataset
# MAGIC
# MAGIC Next, we'll load the safety evaluation dataset which contains examples that test various safety scenarios. This dataset is stored in our volume as `safety_eval` and includes different types of content to evaluate safety responses.

# COMMAND ----------

path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/safety_eval.json")

with path.open("r", encoding="utf-8") as f:
    safety_eval_dataset = json.load(f)

pprint(safety_eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.2 Execute Safety Evaluation
# MAGIC
# MAGIC Now we'll evaluate the safety dataset using the same evaluation framework. The Safety judge will assess whether the agent's responses contain any harmful, offensive, or inappropriate content.

# COMMAND ----------

safety_results = mlflow.genai.evaluate(
    data=safety_eval_dataset,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=[safety_eval],
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.3 Understanding `Safety` Results
# MAGIC
# MAGIC The Safety scorer evaluates content for harmful or inappropriate material and provides insights into the safety of your agent's responses:
# MAGIC
# MAGIC **Result interpretation:**
# MAGIC - **"yes"** - Content is safe and appropriate
# MAGIC - **"no"** - Content contains harmful, offensive, or inappropriate material
# MAGIC - **Rationale** - Detailed explanation of what safety concerns were identified
# MAGIC
# MAGIC **Common safety concerns:**
# MAGIC - **Harmful content** - Violence, hate speech, or dangerous instructions
# MAGIC - **Inappropriate material** - Adult content or offensive language
# MAGIC - **Toxic behavior** - Harassment, bullying, or discriminatory content
# MAGIC
# MAGIC This evaluation helps ensure your agent maintains appropriate safety standards across different types of user interactions.

# COMMAND ----------

print(f"The run ID is: {safety_results.run_id}")
print(f"The aggregated metrics are: {safety_results.metrics}")
print("\nThe results from the safety evaluation:")
display(getattr(safety_results, "result_df", None))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4. Evaluating Multiple Metrics at Once
# MAGIC
# MAGIC With `mlflow.genai.evaluate()`, you can run multiple evaluation scorers in one call and log all resulting metrics in a single evaluation run by passing a list to the scorers parameter. To do this, you can simply add more to the list of scorers by, for example, setting `scorers=[safety_eval, correctness_eval]`.
# MAGIC
# MAGIC **Instructions:**
# MAGIC After running the next cell, inspect the evaluation like before. When inspecting the trace, you will see both `correctness` and `Safety` present.
# MAGIC
# MAGIC ![mlflow-evaluation-runs3.png](../Includes/images/built-in agents with mlflow/mlflow-evaluation-runs3.png "mlflow-evaluation-runs3.png")

# COMMAND ----------

scorers = [safety_eval, correctness_eval]
mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=scorers
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demonstration, you have successfully learned how to leverage MLflow's built-in judges for automated agent evaluation. You've explored two critical evaluation dimensions:
# MAGIC
# MAGIC 1. **Correctness evaluation** - Assessing factual accuracy against ground truth
# MAGIC 2. **Safety evaluation** - Evaluating whether content is free from harmful or inappropriate material
# MAGIC
# MAGIC These built-in judges are useful for OOB LLM evaluation, but there are other approaches to explore such as [guideline judges, custom judges, and code-based scorers](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers#built-in-llm-judges).

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>