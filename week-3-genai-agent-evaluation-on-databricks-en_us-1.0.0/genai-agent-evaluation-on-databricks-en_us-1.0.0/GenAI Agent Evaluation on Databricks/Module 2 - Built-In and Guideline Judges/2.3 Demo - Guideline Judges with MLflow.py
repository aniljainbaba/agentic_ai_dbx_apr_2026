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
# MAGIC # Demo - Guideline Judges with MLflow
# MAGIC
# MAGIC **Overview** 
# MAGIC
# MAGIC This demonstration explores how to implement and use guideline judges in MLflow for evaluating generative AI applications. Guideline judges provide a powerful way to evaluate AI outputs against custom business rules and quality standards using natural language criteria.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this demonstration, you will be able to:
# MAGIC
# MAGIC - Understand the difference between global and per-row guideline judges in MLflow
# MAGIC - Implement the built-in `Guidelines()` judge for uniform evaluation criteria
# MAGIC - Apply the `ExpectationsGuidelines()` judge for scenario-specific evaluations
# MAGIC - Write effective natural language guidelines that reference context variables
# MAGIC - Distinguish between offline and online evaluation approaches for AI systems
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

# MAGIC %run ../Includes/Classroom-Setup-3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1. Understanding Guideline Judges
# MAGIC
# MAGIC A guideline judge is a built-in MLflow component that evaluates whether AI responses pass or fail custom natural-language rules. These judges excel at evaluating business-critical aspects like compliance requirements, style guidelines, factual accuracy, and content appropriateness.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1. Types of Guideline Judges Available
# MAGIC
# MAGIC MLflow provides two primary guideline judge implementations:
# MAGIC
# MAGIC 1. **Built-in `Guidelines()` judge**: Applies global guidelines uniformly to all rows in your evaluation dataset. This judge evaluates application inputs and outputs and works in both offline evaluation and production monitoring scenarios.
# MAGIC
# MAGIC 2. **Built-in `ExpectationsGuidelines()` judge**: Applies per-row guidelines that have been labeled by domain experts in an evaluation dataset. This approach evaluates application inputs and outputs but is designed specifically for offline evaluation workflows.
# MAGIC
# MAGIC Both judges use specially-tuned large language models to make pass/fail determinations based on your specified criteria.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1.2. Offline vs. Online Evaluation
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #0d47a1; font-size: 1.1em;">Offline vs. Online Evaluation</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;"><strong>Offline evaluation</strong> focuses on testing AI systems before deployment using benchmark datasets and reference metrics, while <strong>online evaluation</strong> gathers real-world feedback from actual users after deployment, tracking live usage and performance. Offline methods validate that a system works as intended; online methods reveal how well it's working in practice and drive continuous improvements.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3. How Guideline Judges Work
# MAGIC
# MAGIC Guidelines judges operate by using a specially-tuned LLM to evaluate whether text meets your specified criteria. The evaluation process follows these key steps:
# MAGIC
# MAGIC 1. **Receives context**: The judge accepts any JSON dictionary containing the data to evaluate, such as request, response, retrieved_documents, or user_preferences
# MAGIC 2. **Applies guidelines**: Your natural language rules are used to define pass/fail conditions
# MAGIC 3. **Makes judgment**: The judge returns a binary pass/fail score with detailed rationale explaining the decision
# MAGIC
# MAGIC The judge automatically extracts request and response data from your application traces, making it easy to evaluate real-world AI interactions without complex data preprocessing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4. Load Evaluation Datasets
# MAGIC
# MAGIC Run the next cell to create a helper function that will read our evaluation datasets from `agent_vol` in Unity Catalog. We will use these datasets later. Take a moment to inspect these datasets and compare and contrast the different fields being used in each one, as they each cover a different use case shown below.

# COMMAND ----------

import json 

def read_eval_from_vol(file_name:str):
    path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/{file_name}")

    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    return dataset

guidelines_dataset = read_eval_from_vol("guidelines_eval.json")
guidelines_dataset_pre_gen = read_eval_from_vol("guidelines_eval_pre_gen.json")
guidelines_dataset_row_level = read_eval_from_vol("guidelines_eval_row_level.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5. Inspect Sample Dataset
# MAGIC
# MAGIC Run the next cell to print a formatted view of `guidelines_dataset`.

# COMMAND ----------

from pprint import pprint

pprint(guidelines_dataset)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Part 2. Implementing Global Guidelines
# MAGIC
# MAGIC Next, we will initialize some guidelines. The `Guidelines()` judge applies uniform guidelines across _all rows_ in your evaluation dataset (defined by `guidelines_dataset`). This approach is ideal when you have consistent quality standards that should apply to every AI interaction. Run the next cell to define a guideline scorer related to the agent's tone. 
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

from mlflow.genai.scorers import Guidelines
language_guideline = Guidelines(
    name="spanish",
    guidelines=["The response should be in Spanish"],
    model_name = guidelines_endpoint
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Basic Global Guidelines Implementation
# MAGIC
# MAGIC Global guidelines work by defining rules that apply to all evaluations uniformly. The judge automatically extracts the `request` (from your inputs) and `response` (from your outputs) to create the evaluation context. Run the next cell to evaluate with our global guideline defined by `language_guideline`. Note that both the inputs will fail since the response will be in English and not Spanish.

# COMMAND ----------

guidelines_dataset_results = mlflow.genai.evaluate(
    data=guidelines_dataset,
    predict_fn= lambda input: agent.predict({"input": input}),
    scorers=[language_guideline]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Inspect Global Guidelines Results
# MAGIC
# MAGIC We can inspect the `guidelines_results` object as shown below. Note that you can click on each row and the MLflow interface will populate for that particular trace. Also note that the assessment will show that both inputs failed because the result was not translated to Spanish (we did not have this as a part of our system prompt or injected as a part of the input).

# COMMAND ----------

print(f"The run ID is: {guidelines_dataset_results.run_id}")
print(f"The aggregated metrics are: {guidelines_dataset_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(guidelines_dataset_results.result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3. Evaluating Pre-Generated Inputs/Outputs
# MAGIC
# MAGIC Now let's consider the case when we have a dataset of pregenerated inputs and outputs that we wish to evaluate. This has been stored in the `guidelines_eval_pre_gen.json` file in `agent_vol` if you wish to view the dataset itself. Essentially, we are testing evaluation on a Spanish response based on some input. The first datapoint correctly returned a Spanish translation while the second did not. 
# MAGIC
# MAGIC Run the next two cells and view the evaluation by clicking **View evaluation results in MLflow** like before. Note that we are not passing our agent with `predict_fn` in `mlflow.genai.evaluate()` since we already have the agent's response (Inputs and Outputs) stored in the eval dataset `guidelines_eval_pre_gen.json`.

# COMMAND ----------

guidelines_dataset_pre_gen_results = mlflow.genai.evaluate(
    data=guidelines_dataset_pre_gen,
    scorers=[language_guideline]
)

# COMMAND ----------

print(f"The run ID is: {guidelines_dataset_pre_gen_results.run_id}")
print(f"The aggregated metrics are: {guidelines_dataset_pre_gen_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(guidelines_dataset_pre_gen_results.result_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Part 3. Implementing Per-Row Guidelines for Edge Cases
# MAGIC
# MAGIC The `ExpectationsGuidelines` judge enables scenario-specific evaluation by applying different guidelines to each row in your dataset. Like we did above, run the next two cells to perform the evaluation using `mlflow.genai.evaluate()` and view the metadata from the evaluation. For this particular dataset, we see that one row will pass the evaluation and one will fail. After running the next cell, go and view the trace like before to view and understand the reasoning. 
# MAGIC
# MAGIC This approach is particularly valuable when different types of interactions require different evaluation criteria and you have granular requirements. 
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">`ExpectationsGuidelines` scorer requires an `outputs` field.</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"> They can be passed directly or as a trace containing them. We will pass them directly.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

from mlflow.genai.scorers import ExpectationsGuidelines

expected_guidelines = ExpectationsGuidelines(
    name="expected_guidelines",
    model_name = guidelines_endpoint
)

guidelines_dataset_row_level_results = mlflow.genai.evaluate(
    data = guidelines_dataset_row_level,
    scorers=[expected_guidelines]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Inspect Per-Row Guidelines Results
# MAGIC
# MAGIC Note you can also view the reasoning using `per_row_results.results_df` and viewing the `assessment` column or clicking on the row of interest like before.

# COMMAND ----------

print(f"The run ID is: {guidelines_dataset_row_level_results.run_id}")
print(f"The aggregated metrics are: {guidelines_dataset_row_level_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(guidelines_dataset_row_level_results.result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. When to Use Per-Row Guidelines
# MAGIC
# MAGIC Per-row guidelines are most effective when:
# MAGIC - You have domain experts who have labeled specific examples with custom guidelines
# MAGIC - Different rows in your dataset require different evaluation criteria
# MAGIC - You need to test how your AI system handles various edge cases or specialized scenarios
# MAGIC - Your evaluation dataset contains diverse interaction types that each have unique quality requirements

# COMMAND ----------

# MAGIC %md
# MAGIC ##Conclusion
# MAGIC
# MAGIC Guideline judges provide a powerful and intuitive way to evaluate generative AI applications using natural language criteria. By implementing both global and per-row guidelines, you can create comprehensive evaluation frameworks that align with your business requirements and quality standards.
# MAGIC
# MAGIC The key advantages of guideline judges include their business-friendly approach (domain experts can write criteria without coding), flexibility to update criteria without code changes, clear interpretability of results, and support for rapid iteration on evaluation criteria.
# MAGIC
# MAGIC Whether you're implementing uniform quality standards across all interactions or need scenario-specific evaluation criteria, MLflow's guideline judges offer the tools necessary to ensure your AI applications meet your organization's standards for compliance, style, accuracy, and overall quality.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>