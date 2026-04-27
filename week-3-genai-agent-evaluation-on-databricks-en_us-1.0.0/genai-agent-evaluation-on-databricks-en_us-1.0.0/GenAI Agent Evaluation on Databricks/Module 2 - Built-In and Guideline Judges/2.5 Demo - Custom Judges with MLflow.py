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
# MAGIC # Demo - Custom Judges with MLflow
# MAGIC
# MAGIC **Overview** 
# MAGIC
# MAGIC This demonstration explores how to create and implement custom LLM judges in MLflow for evaluating generative AI applications. Custom judges provide the flexibility to define complex, nuanced scoring guidelines using natural language instructions tailored to your specific business requirements and evaluation criteria.
# MAGIC
# MAGIC Custom judges built with `make_judge()` offer complete control over evaluation logic, allowing you to assess quality dimensions that may not be covered by built-in judges. This includes domain-specific requirements, complex multi-step evaluations, and trace-based analysis of agent execution patterns.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this demonstration, you will be able to:
# MAGIC
# MAGIC 1. **Create custom judges** using `mlflow.genai.judges.make_judge()` with natural language instructions
# MAGIC 2. **Implement template variables** to access inputs, outputs, expectations, and execution traces
# MAGIC 3. **Design trace-based judges** that analyze complete agent execution workflows
# MAGIC 4. **Configure feedback value types** for categorical, boolean, and numeric scoring systems
# MAGIC 5. **Apply best practices** for writing effective judge instructions and model selection
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

# MAGIC %run ../Includes/Classroom-Setup-5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1. Understanding Custom Judges

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1.1. What are Custom Judges?
# MAGIC
# MAGIC Custom LLM judges are specialized evaluation functions created using `mlflow.genai.judges.make_judge()` that allow you to define complex and nuanced scoring guidelines for GenAI applications using natural language instructions. Unlike built-in judges that cover common quality dimensions, custom judges give you complete control over evaluation criteria. Note that this is a different imported class than previous demos, where we relied upon `mlflow.genai.judges.scorers()`. 
# MAGIC
# MAGIC **Key characteristics of custom judges:**
# MAGIC - **Natural language instructions** - Define evaluation criteria in plain English
# MAGIC - **Template variable access** - Use inputs, outputs, expectations, and traces in evaluation
# MAGIC - **Flexible feedback types** - Return categorical, boolean, or numeric scores
# MAGIC - **Domain-specific evaluation** - Address unique business requirements and quality standards
# MAGIC
# MAGIC The `make_judge` function in MLflow creates a custom LLM-based judge (scorer) for evaluating GenAI outputs according to your own instructions and criteria. You specify the judge's name, natural language instructions (with template variables like (`{inputs)}`, `{{outputs}}`, etc., which we will cover down below), and optionally the model and output type; the function returns a callable judge object that can be used to assess responses, conversations, or traces. 
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #0d47a1; font-size: 1.1em;">
# MAGIC         More on <code>mlflow.genai.judges.make_judge()</code>
# MAGIC       </strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC         You can read more about the <code>make_judges</code> class
# MAGIC         <a href="https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/make-judge/" target="_blank">
# MAGIC           here
# MAGIC         </a>.
# MAGIC       </p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2. Template Variables in Custom Judges
# MAGIC
# MAGIC Custom judges use **template variables** to access different aspects of your agent's execution. These variables provide the context needed for comprehensive evaluation:
# MAGIC
# MAGIC - **`{{inputs}}`** - Input data provided to the agent
# MAGIC - **`{{outputs}}`** - Output data generated by your agent  
# MAGIC - **`{{expectations}}`** - Ground truths or expected outcomes
# MAGIC - **`{{trace}}`** - The complete execution trace of your agent
# MAGIC - **`{{conversation}},`** - The complete execution of your agent
# MAGIC
# MAGIC **Important constraints:**
# MAGIC - Your instructions must include at least one template variable
# MAGIC - Only these four variables are allowed (custom variables like `{{question}}` will cause validation errors)
# MAGIC - This ensures consistent behavior and prevents template injection issues
# MAGIC - When using `make_judge` (see below), it must contain at least one of the 4 template variables mentioned above.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3. Types of Custom Judges
# MAGIC
# MAGIC Custom judges can be categorized into two main types based on their evaluation approach:
# MAGIC
# MAGIC **Standard Custom Judges:**
# MAGIC - Evaluate inputs, outputs, and expectations
# MAGIC - Focus on content quality and correctness
# MAGIC - Suitable for response validation and content assessment
# MAGIC
# MAGIC **Trace-Based Judges:**
# MAGIC - Analyze complete execution traces using Model Context Protocol (MCP) tools
# MAGIC - Validate tool usage patterns and execution workflows
# MAGIC - Identify performance bottlenecks and investigate failures
# MAGIC - Verify multi-step agent processes
# MAGIC
# MAGIC For trace-based judges, the `model` parameter must be specified in `make_judge()` to enable trace analysis capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4. Load Evaluation Datasets
# MAGIC
# MAGIC Run the next cell to create a helper function that will read our evaluation datasets from `agent_vol` in Unity Catalog. We will use these datasets to demonstrate different types of custom judges.

# COMMAND ----------

# MAGIC %md
# MAGIC Run the next cell to view the evaluation dataset we will be using with our custom judge.

# COMMAND ----------

import json 
from pathlib import Path

path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/custom_eval.json")
    
with path.open("r", encoding="utf-8") as f:
    custom_eval = json.load(f)

print("✅ Loaded dataset custom_eval as `custom_eval`")
pprint(custom_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2. Creating Standard Custom Judges

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Basic Custom Judge Implementation
# MAGIC
# MAGIC Let's create our first custom judge that evaluates response completeness. This judge will assess whether the agent's response fully addresses all aspects of the user's question.
# MAGIC
# MAGIC **Key components of a custom judge:**
# MAGIC - **name** - Identifier for the judge in evaluation results
# MAGIC - **instructions** - Natural language evaluation criteria using template variables
# MAGIC - **feedback_value_type** - Expected return type (Literal for categorical responses)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC First, let's define our **feedback type**. `feedback_value_type` is one of the supported types for serialization and the judge will use structured outputs to enforce the type specified (which is recommended). 
# MAGIC
# MAGIC Currently, the supportes types are
# MAGIC - **PbValueType**: `PbValueType` in MLflow represents the set of primitive types allowed for feedback or expectation values in assessments: `float`, `int`, `str`, or `bool`. It is used as a type alias to define what kinds of values can be stored in feedback or expectation fields. 
# MAGIC - **Literal** types with PbValueType values: We will use this one down below. 
# MAGIC - `dict[str, PbValueType]`
# MAGIC - `list[PbValueType]`
# MAGIC
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #0d47a1; font-size: 1.1em;">
# MAGIC         <code>make_judge</code> Source Code
# MAGIC       </strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC         You can read the source code for <code>make_judge</code>
# MAGIC         <a href="https://mlflow.org/docs/latest/api_reference/_modules/mlflow/genai/judges/make_judge.html" target="_blank">
# MAGIC           here
# MAGIC         </a>.
# MAGIC       </p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

from typing import Literal

response_completeness_feedback_value_type = Literal["complete", "partial", "incomplete"]

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have our feedback type, in the code below we will pass:
# MAGIC - The name of the judge, `response_completeness`
# MAGIC - Instructions, which is stored in our `coherent_instructions` object
# MAGIC - Feedback value type, `feedback_value_type`

# COMMAND ----------

from mlflow.genai.judges import make_judge

# Create a custom judge for response completeness
completeness_judge = make_judge(
    name="response_completeness",
    instructions=(
        coherent_instructions
    ),
    feedback_value_type=response_completeness_feedback_value_type
)

# COMMAND ----------

# MAGIC %md
# MAGIC Below is a summary of different arguments you can pass with `make_judge`, some of which will not be covered in this demo.
# MAGIC
# MAGIC
# MAGIC
# MAGIC | Field Name            | Description |
# MAGIC |----------------------|-------------|
# MAGIC | `name` | The name of the judge. |
# MAGIC | `instructions` | Natural language instructions for evaluation. Must contain **at least one** of the following template variables: `{{inputs}}`, `{{outputs}}`, `{{expectations}}`, `{{conversation}}`, or `{{trace}}`. Custom variables are not supported. <br><br> **Note:** `{{conversation}}` may only be used together with `{{expectations}}` and cannot be combined with `{{inputs}}`, `{{outputs}}`, or `{{trace}}`. |
# MAGIC | `model` | The model identifier used for evaluation (for example, `"openai:/gpt-4"`). |
# MAGIC | `description` | A description of what the judge evaluates. |
# MAGIC | `feedback_value_type` | Type specification for the `value` field in the Feedback object. The judge uses structured outputs to enforce this type. If unspecified, the type is inferred by the judge. It is **recommended** to explicitly specify this field. <br><br> **Supported types (matching `FeedbackValueType`):** <br> • `int` — Integer ratings (e.g., 1–5 scale) <br> • `float` — Floating-point scores (e.g., 0.0–1.0) <br> • `str` — Text responses <br> • `bool` — Yes/No evaluations <br> • `Literal[values]` — Enum-like choices (e.g., `Literal["good", "bad"]`) <br> • `dict[str, int \| float \| str \| bool]` — Dictionary with string keys and primitive values <br> • `list[int \| float \| str \| bool]` — List of primitive values <br><br> **Note:** Pydantic `BaseModel` types are not supported. |
# MAGIC | `inference_params` | Optional dictionary of inference parameters passed to the model (for example, `temperature`, `top_p`, `max_tokens`). These parameters allow fine-grained control over evaluation behavior. Lower temperatures typically yield more deterministic and reproducible results. |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Inspect Custom Judge Dataset
# MAGIC
# MAGIC Now that we have our cust judge instance, `completeness_judge`, let's examine the evaluation dataset we'll use to test our custom judge. This dataset contains various types of queries and responses to evaluate completeness.

# COMMAND ----------

print("Custom Judge Evaluation Dataset:")
pprint(custom_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3. Execute Custom Judge Evaluation
# MAGIC
# MAGIC Now let's run our custom judge against the evaluation dataset. We'll use `mlflow.genai.evaluate()` to process each example and generate completeness assessments.

# COMMAND ----------

completeness_results = mlflow.genai.evaluate(
    data=custom_eval,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=[completeness_judge]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4. Inspect Custom Judge Results
# MAGIC
# MAGIC Examine the results from our completeness evaluation. The judge provides both categorical scores and detailed rationales explaining each decision.

# COMMAND ----------

print(f"The run ID is: {completeness_results.run_id}")
print(f"The aggregated metrics are: {completeness_results.metrics}")
print("\nThe results from the completeness evaluation:")
display(completeness_results.result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3. Creating Trace-Based Custom Judges

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 3.1. Understanding Trace-Based Evaluation
# MAGIC
# MAGIC Trace-based judges analyze the complete execution trace of your agent to understand what happened during agent execution. They can autonomously explore traces using Model Context Protocol (MCP) tools and provide insights into:
# MAGIC
# MAGIC - **Tool usage patterns** - Whether appropriate tools were selected and used correctly
# MAGIC - **Performance bottlenecks** - Identifying slow or inefficient execution steps  
# MAGIC - **Execution failures** - Understanding why certain operations failed
# MAGIC - **Multi-step workflows** - Verifying complex agent reasoning chains
# MAGIC
# MAGIC <div style="
# MAGIC   border-left: 4px solid #ff9800;
# MAGIC   background: #fff3e0;
# MAGIC   padding: 14px 18px;
# MAGIC   border-radius: 4px;
# MAGIC   margin: 16px 0;
# MAGIC ">
# MAGIC   <strong style="display:block; color:#e65100; margin-bottom:6px; font-size: 1.1em;">
# MAGIC     Warning
# MAGIC   </strong>
# MAGIC   <div style="color:#333;">
# MAGIC Trace-based judges require the <code>model</code> parameter to be specified in <code>make_judge()</code>.
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Create a Tool Usage Judge
# MAGIC
# MAGIC Let's create a trace-based judge that validates whether the agent used tools appropriately for the given request. As before, we will create our `feedback_value_type` and use imported instructions (`tool_usage_instructions`) from our agent's configuration file. This time we will set the type to `bool` since we're really just asking whether a tool was used or not.

# COMMAND ----------

tool_feedback_value_type = bool

# COMMAND ----------

# Create a trace-based judge for tool usage validation
tool_usage_judge = make_judge(
    name="tool_usage_validator",
    instructions=(
        tool_usage_instructions
    ),
    feedback_value_type=tool_feedback_value_type,
    model=custom_eval_endpoint  # Required for trace-based judges
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Create Trace Dataset
# MAGIC In order to demonstrate how to pass a trace to a custom judge, we need to first build a trace with a session ID as shown in the next step with the helper function `gen_trace_data()`. This will create 3 different traces under the session ID passed with the function, e.g. `session-demo4-001` is the default, along with your username (these were defined in **A7. Load Agent Evaluation Configuration**). We then search for the trace using these values.

# COMMAND ----------

from typing import Tuple, Dict, Any
from mlflow.entities import SpanType

def gen_trace_data(query: str, model :str =custom_eval_endpoint, user_id :str =username, session_id :str =session_id) -> Tuple[Dict[str, Any]]:
    with mlflow.start_span(name = "populate_agent_trace",span_type=SpanType.AGENT) as span:
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": session_id,
                "mlflow.trace.user": user_id
            },
            tags={
                "training_type": "agent_eval_training",
                "model": model,
                "agent_type": "TOOL-CALLING"
            }
        )

        query_payload = [
            {"input": [{"role": "user", "content": query}]}]

        response = agent.predict(query_payload)

        # log input and output at the span-level for visability
        span.set_inputs({"query": query})
        span.set_outputs({"response": response})
        trace_id = span.trace_id

    return query, trace_id

# COMMAND ----------

# Some sample quries to test tool calling
queries = [
    "How many Entire home/apt listings are in the Mission neighborhood?",
    "Count the number of Private room listings in Nob Hill.",
    "What is the average listing price in Haight Ashbury?"
    ]

# Generate traces for the queries
for query in queries:
    gen_trace_data(query=query)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we load the trace using the `experiment_id` and `session_id` as shown in the next cell. `session_traces` returns a list of traces that we can also inspect (see the output from the next cell).

# COMMAND ----------

session_traces = mlflow.search_traces(
    locations=[experiment_id],
    filter_string=f"metadata.`mlflow.trace.session` = '{session_id}'",
    return_type="list")
session_traces[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Execute the single agent's response.

# COMMAND ----------

trace = session_traces[0]

# Evaluate the entire conversation session
feedback = tool_usage_judge(trace=trace)

# COMMAND ----------

print(f"Assessment: {feedback.value}")
print(f"Rationale: {feedback.rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4. Perform Batch Trace-Based Judge Evaluation
# MAGIC
# MAGIC Run the trace-based judge to analyze how well the agent uses tools during execution.

# COMMAND ----------

trace_judge_results = mlflow.genai.evaluate(
    data=session_traces,
    scorers=[tool_usage_judge]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5. Inspect Trace Judge Results
# MAGIC
# MAGIC Review the trace-based evaluation results. The judge provides insights into tool usage patterns and execution quality.

# COMMAND ----------

print(f"The run ID is: {trace_judge_results.run_id}")
print(f"The aggregated metrics are: {trace_judge_results.metrics}")
print("\nThe results from trace-based evaluation:")
display(trace_judge_results.result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demonstration, you learned how to create and implement custom LLM judges in MLflow for comprehensive evaluation of generative AI applications. You explored both standard custom judges for evaluating response quality and trace-based judges for analyzing agent execution patterns.
# MAGIC
# MAGIC **Key takeaways:**
# MAGIC 1. Custom judges provide flexible evaluation capabilities beyond built-in scorers
# MAGIC 2. Template variables enable access to inputs, outputs, expectations, and execution traces
# MAGIC 3. Trace-based judges offer deep insights into agent tool usage and workflow patterns
# MAGIC 4. Proper feedback value type configuration ensures consistent evaluation results
# MAGIC 5. MLflow's evaluation framework seamlessly integrates custom judges with existing workflows
# MAGIC
# MAGIC These custom evaluation capabilities enable you to build robust quality assurance processes tailored to your specific GenAI application requirements.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>