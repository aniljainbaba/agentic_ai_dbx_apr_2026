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
# MAGIC # Demo - Agent Setup
# MAGIC **Overview**
# MAGIC
# MAGIC This demo focuses on the setup of a custom agent using environment variables specific to the user and Python code located in the folder `./artifacts`. At this point in the lifecycle of your agent application, you are ready for evaluation and this notebook acts as a checkpoint where you have an agent already registered to Unity Catalog.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC - Configure the Databricks environment and classroom assets needed for agent evaluation
# MAGIC - Load a Unity Catalog-registered agent by name and alias
# MAGIC - Generate traces by interacting with an agent and locate those traces in Unity Catalog

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

# MAGIC %run ../Includes/Classroom-Setup-1

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

# MAGIC %md
# MAGIC ## Part 1 - Import The Agent and View MLflow Trace
# MAGIC
# MAGIC In this section, we'll import and test our agent that's been instrumented with MLflow tracing. The agent combines SQL functions to provide comprehensive responses about San Francisco Airbnb listings.
# MAGIC
# MAGIC MLflow tracing automatically captures the execution flow of our agent, including retrieval operations, function calls, and LLM interactions.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Import the Agent
# MAGIC
# MAGIC Here we will load the model using the Unity Catalog path and alias that was created as a part of the setup script for this demo using `demo_setup.load_agent()`.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">We Are Using A Simple Agent</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">The custom agent built in this demo and used in other demos/labs is really a <strong>tool-calling agent</strong> using UDFs registered to Unity Catalog. Your agent may have a varienty tools, but this is not the focus of our current objective, which is to understand the basics of evaluation.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

agent = demo_setup.load_agent("airbnb_eval_agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Questions
# MAGIC
# MAGIC Now that we have our agent working, let's pass some questions using the `predict()` method defined in `airbnb_agent.py`. Let's first create a helper function we can use to send the expected payload to the agent.

# COMMAND ----------

def agent_payload(question: str):
    return [{"input": [{"role": "user", "content": question}]}]

# COMMAND ----------

agent.predict(agent_payload("What tools do you have available?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect the Trace
# MAGIC
# MAGIC Based on the output above, we can see the tools the agent has access to. Let's next ask a question that will invoke a couple tool calls. Namely, the next question will nudge the agent to use both:
# MAGIC
# MAGIC 1. `avg_neigh_price`
# MAGIC 2. `cnt_by_room_type`
# MAGIC
# MAGIC The screenshot below is an example of the output when running the next cell and what you should expect. Notice the tools defined above are being called.
# MAGIC
# MAGIC ![mlflow-toolcall.png](../Includes/images/built-in agents with mlflow/mlflow-toolcall.png "mlflow-toolcall.png")

# COMMAND ----------

agent.predict(agent_payload("Can you tell me what the average price is in Mission? Also, what's the number of listings for that neighborhood that have private rooms?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2 - Inspect the Trace in Unity Catalog
# MAGIC
# MAGIC Navigate to the **Catalog Explorer** and search for **airbnb_eval_agent**. There you will find a model (at least **Version 1**) with the alias **@champion**. Click on it and navigate to the **Traces** tab. There you will find the same two requests we sent above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC - Your agent is now registered in Unity Catalog and can be loaded using `mlflow.pyfunc.load_model()`
# MAGIC - MLflow tracing automatically captures all agent interactions, providing valuable data for evaluation
# MAGIC - The agent successfully combines multiple tools (SQL functions) to provide comprehensive responses
# MAGIC - All traces are accessible through the Unity Catalog interface for monitoring and debugging
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC With your agent now properly configured and registered, you're ready to move on to the next notebooks in this series where you'll learn how to evaluate agent performance, create evaluation datasets from traces, and implement comprehensive testing strategies.
# MAGIC
# MAGIC Remember that in future notebooks, you'll load your agent directly using `mlflow.pyfunc.load_model()` rather than the `demo_setup()` helper function used in this demo.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>