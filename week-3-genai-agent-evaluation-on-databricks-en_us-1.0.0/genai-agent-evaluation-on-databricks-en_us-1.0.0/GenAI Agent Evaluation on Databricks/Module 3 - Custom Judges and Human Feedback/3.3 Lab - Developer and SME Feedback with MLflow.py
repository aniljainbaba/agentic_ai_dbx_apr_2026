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
# MAGIC # Lab - Developer and SME Feedback with MLflow
# MAGIC
# MAGIC **Overview**
# MAGIC
# MAGIC This lab provides hands-on experience with MLflow's human feedback capabilities, focusing on collecting and managing assessments from different reviewer personas. You'll learn to implement feedback workflows that support development iteration, domain expert evaluation, and end-user input collection. The lab covers MLflow Assessments, which allow you to attach structured feedback, scores, and ground truth to traces and spans for quality evaluation and improvement.
# MAGIC
# MAGIC Human feedback is critical for improving AI applications, as it provides qualitative insights that complement automated metrics. This lab demonstrates how to systematically collect, organize, and analyze feedback from multiple stakeholder types to drive continuous improvement of your AI systems.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC - Implement developer feedback workflows using MLflow Assessments
# MAGIC - Configure SME (Subject Matter Expert) feedback collection through the Chat UI
# MAGIC - Distinguish between feedback and expectation assessment types
# MAGIC - Navigate MLflow's tracing interface to add and review assessments
# MAGIC - Review SME feedback programmatically in a Databricks Notebook
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Prerequisites</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"> This lab uses the agent created in <strong>01 - Agent Setup</strong>. Please ensure you have completed that notebook before proceeding.</p>
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

# MAGIC %run ../Includes/Classroom-Setup-6

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1. Assessment Types
# MAGIC
# MAGIC MLflow supports two distinct types of assessments, each serving different evaluation purposes. Understanding these types is crucial for implementing effective feedback workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1. Feedback Assessments
# MAGIC
# MAGIC **Feedback** evaluates your app's actual outputs or intermediate steps. It answers questions like "Was the agent's response good?" Feedback assesses what the app produced, such as ratings or comments, and provides qualitative insights into the generated content.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2. Expectation Assessments
# MAGIC
# MAGIC **Expectation** defines the desired or correct outcome (ground truth) that your app should have produced. For example, this could be "The ideal response" to a user's query. For a given input, the Expectation is always the same. Expectations define what the app should generate and are useful for creating evaluation datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2. Reviewer Personas and MLflow Support
# MAGIC
# MAGIC There are three main categories for which feedback is gathered, each with specific access patterns and use cases within MLflow's ecosystem.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Developer Reviewers
# MAGIC
# MAGIC **Developer** reviewers can directly annotate traces within the MLflow UI. These reviewers have full workspace access and can add assessments during development and testing phases.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Domain Expert Reviewers
# MAGIC
# MAGIC **Domain Expert** reviewers are SMEs that have been identified to provide structured feedback on your application's outputs and define **expectations** for correct responses. These reviewers set the standard for answering the question _What do high-quality responses look like?_ There are two approaches for collecting domain expert feedback:
# MAGIC - Interactive testing with the [Chat UI](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/expert-feedback/live-app-testing)
# MAGIC - [Labeling existing traces](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/expert-feedback/label-existing-traces). This is ideal for **structured** evaluation sessions.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 2.3. End User Reviewers
# MAGIC
# MAGIC **End User** reviewers are users that are interacting with your live application. These users have unique insight into real-world performance, helping identify problematic queries that need fixing, while highlighting successful interactions to preserve during future updates.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC In this lab, you will only focus on developer and SME feedback scenarios. For more information on user feedback within backend and frontend applications like <code>FastAPI</code> and <code>React</code>, please see 
# MAGIC <a href="https://docs.databricks.com/aws/en/mlflow3/genai/tracing/collect-user-feedback/?language=Development">
# MAGIC       this documentation
# MAGIC     </a>
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3. Developer Feedback Implementation
# MAGIC
# MAGIC First, you will explore how developers can provide feedback during the development phase. MLflow Tracing allows you to add feedback or expectations directly to traces during development, giving you a quick way to record quality issues, mark successful examples, or add notes for future reference.
# MAGIC
# MAGIC The lab initialization script has created traces for you to evaluate. When running `demo_lab.run()`, we stressed our agent with 3 queries that are now available for assessment.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Navigate to Feedback Experiment
# MAGIC
# MAGIC Follow these steps to access the feedback experiment and provide developer assessments:
# MAGIC
# MAGIC 1. Navigate to and click on **Workspace** on the left side menu
# MAGIC 2. Locate the experiment called `feedback_experiment` and click on it
# MAGIC 3. We have created a session called **feedback-session-001** - click on **Sessions** to see this session, then click on it
# MAGIC 4. On the left, you will see **Turn 1** through **Turn 3** (see screenshot below)
# MAGIC
# MAGIC ![mlflow-assessment3.png](../Includes/images/feedback evaluation/mlflow-assessment3.png "mlflow-assessment3.png")
# MAGIC
# MAGIC Here we can add assessments at both the session-level and trace-level.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Viewing Trace-Level Assessment
# MAGIC
# MAGIC Now we'll add a trace-level assessment to evaluate tool usage:
# MAGIC
# MAGIC 1. Click on **Turn 1** and select **View full trace** or **Evaluate trace**
# MAGIC 2. You will see **Assessment** on the right of the trace - click on it (see screenshot)
# MAGIC
# MAGIC ![mlflow-assessment1.png](../Includes/images/feedback evaluation/mlflow-assessment1.png "mlflow-assessment1.png")
# MAGIC
# MAGIC 3. Click on **Add Feedback** or **Add Expectation** on the right of the screen
# MAGIC 4. This will create a submenu with options to select **Feedback** or **Expectations** from **Assessment Type**, enter the **Assessment Name**, and select **String**, **Boolean**, or **Number** from **Data Type** (see screenshot below)
# MAGIC
# MAGIC ![mlflow-assessment2.png](../Includes/images/feedback evaluation/mlflow-assessment2.png "mlflow-assessment2.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Configure Tool Usage Assessment
# MAGIC
# MAGIC Based on the lab setup, you will see that a tool was used and it was indeed the correct one. Fill out the assessment with these values:
# MAGIC
# MAGIC - **Assessment Type**: _Expectation_
# MAGIC - **Assessment Name**: `tool_usage`
# MAGIC - **Data Type**: _Boolean_
# MAGIC - **Value**: _True_
# MAGIC - **Rationale**: _The tool was used correctly._
# MAGIC
# MAGIC Click **Create** and close the trace.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4. Add Session-Level Assessment
# MAGIC
# MAGIC Now let's add a session-level assessment:
# MAGIC
# MAGIC 1. You can evaluate the other 2 turns using the same process as Turn 1
# MAGIC 2. Return to the main session menu and click on **Add Feedback** under **Session scorers** > **Feedback**
# MAGIC 3. Fill out the following values:
# MAGIC    - **Assessment Type**: Feedback
# MAGIC    - **Assessment Name**: _ready_for_sme_feedback_
# MAGIC    - **Data Type**: _String_
# MAGIC    - **Value**: _True_
# MAGIC    - **Rationale**: _Based on the review of the outputs from each turn, the results are correct and testing is complete and ready for SME review._
# MAGIC 4. Click **Create**
# MAGIC
# MAGIC **NOTE:** You are now ready to send to your SMEs for review, which we will cover in the next section.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4. SME Feedback Implementation
# MAGIC
# MAGIC Next, we'll implement SME (Subject Matter Expert) feedback. Suppose you are now acting as a SME for review. There are two approaches for SME feedback collection:
# MAGIC
# MAGIC 1. **Interactive testing with the Chat UI**
# MAGIC 2. **Labeling existing traces**
# MAGIC
# MAGIC In this lab, you will use the Chat UI approach.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 4.1. Deploy Agent Using Agent Framework
# MAGIC
# MAGIC Next, we need a deployed model to a model serving endpoint. We will use the `databricks` SDK class `agent.deploy()` method. All we need is the `model_name` and `uc_model_info`, both of which were created as part of the classroom setup.
# MAGIC
# MAGIC **NOTE:** In the interest of time, this has been performed for you as part of running **01 Demo - Agent Setup**.
# MAGIC
# MAGIC Here is a code snippet showing how to deploy the agent:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC from databricks import agents
# MAGIC
# MAGIC model_name = f"{catalog_name}.{schema_name}.{agent_name}"  # UC FQN
# MAGIC alias = "Champion"
# MAGIC
# MAGIC client = MlflowClient()
# MAGIC mv = client.get_model_version_by_alias(model_name, alias)
# MAGIC
# MAGIC # Pass the version, not the ModelVersion object
# MAGIC deployment = agents.deploy(
# MAGIC     model_name=model_name,
# MAGIC     model_version=int(mv.version),  # str also works; cast to int for clarity
# MAGIC     scale_to_zero=True
# MAGIC )
# MAGIC print("Endpoint:", deployment.endpoint_name)
# MAGIC ```
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
# MAGIC <code>scale_to_zero=True</code> means that your agent endpoint might be scaled down to workspace resources. This means that performing your initial query can take some time.
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2. Interactive Testing with Chat UI
# MAGIC
# MAGIC We will use the [`get_review_app()`](https://mlflow.org/docs/latest/api_reference/_modules/mlflow/genai/labeling.html#get_review_app) method to get or create the review app.
# MAGIC
# MAGIC **Warning:** **Domain experts** need the following permissions to use the Review App's Chat UI:
# MAGIC
# MAGIC - **Account access**: Must be provisioned in your Databricks account, but does *not* require workspace access
# MAGIC - **Endpoint access**: `CAN_QUERY` permission on the model serving endpoint
# MAGIC - **MLflow access**: Workspace access with `CAN_EDIT` permissions on the MLflow Experiment
# MAGIC
# MAGIC For users without workspace access, account admins can:
# MAGIC - Use account-level SCIM provisioning to sync users from your identity provider
# MAGIC - Manually register users and groups in Databricks
# MAGIC
# MAGIC See [User and group management](https://docs.databricks.com/aws/en/admin/users-groups/scim/) for details.

# COMMAND ----------

from mlflow.genai.labeling import get_review_app

review_app = get_review_app()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3. Configure Review App
# MAGIC
# MAGIC Let's connect our review app to our agent. The output from the next cell will contain the Review App URL. If you click on it, you will be able to:
# MAGIC
# MAGIC - Access the chat interface through their web browser
# MAGIC - Interact with your application by typing questions
# MAGIC - Provide feedback after each response using the built-in feedback controls
# MAGIC - Continue the conversation to test multiple interactions
# MAGIC
# MAGIC **Alternative UI approach:** You can also navigate to **Serving** on the left side menu, click on your deployed agent, and select the dropdown menu next to **Use** and select **Open review app**.

# COMMAND ----------

# MAGIC %md
# MAGIC In the next code snippet, we are using `agent_name` and `model_username_string` string variables that were created as a part of our lab setup.

# COMMAND ----------

review_app.add_agent(
    agent_name = agent_name,
    model_serving_endpoint = model_username_string
)

print(f"Share this URL: {review_app.url}/chat")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4. Test Query and Provide Feedback
# MAGIC
# MAGIC Next, let's pass a query and provide feedback. For your convenience, a **Copy to clipboard** button has been added - click it to copy the query. Paste it in the review app. Below is a screenshot of what your screen should look like:
# MAGIC
# MAGIC ![feedback-app.png](../Includes/images/feedback evaluation/feedback-app.png "feedback-app.png")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <button onclick="copyBlock()">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-block" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC How many rooms in Mission have a private room?
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyBlock() {
# MAGIC   const el = document.getElementById("copy-block");
# MAGIC   if (!el) return;
# MAGIC
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   // Preferred modern API
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackCopy(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackCopy(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackCopy(text) {
# MAGIC   const textarea = document.createElement("textarea");
# MAGIC   textarea.value = text;
# MAGIC   textarea.style.position = "fixed";
# MAGIC   textarea.style.left = "-9999px";
# MAGIC   document.body.appendChild(textarea);
# MAGIC   textarea.select();
# MAGIC   try {
# MAGIC     document.execCommand("copy");
# MAGIC     alert("Copied to clipboard");
# MAGIC   } catch (err) {
# MAGIC     console.error("Fallback copy failed:", err);
# MAGIC     alert("Could not copy to clipboard. Please copy manually.");
# MAGIC   } finally {
# MAGIC     document.body.removeChild(textarea);
# MAGIC   }
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5. Submit Feedback
# MAGIC
# MAGIC Let's add some feedback. The result looks good and you should clearly see that a tool was used. Copy and paste the following into the feedback after selecting **Yes** in the **Awaiting Feedback** window as shown in the previous screenshot. After pasting, click **Done**. You will see a message that the feedback has been submitted.
# MAGIC
# MAGIC ![feedback-app2.png](../Includes/images/feedback evaluation/feedback-app2.png "feedback-app2.png")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <button onclick="copyBlock()">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-block" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC The agent should end with "Is there anything else I can help with?"
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyBlock() {
# MAGIC   const el = document.getElementById("copy-block");
# MAGIC   if (!el) return;
# MAGIC
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   // Preferred modern API
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackCopy(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackCopy(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackCopy(text) {
# MAGIC   const textarea = document.createElement("textarea");
# MAGIC   textarea.value = text;
# MAGIC   textarea.style.position = "fixed";
# MAGIC   textarea.style.left = "-9999px";
# MAGIC   document.body.appendChild(textarea);
# MAGIC   textarea.select();
# MAGIC   try {
# MAGIC     document.execCommand("copy");
# MAGIC     alert("Copied to clipboard");
# MAGIC   } catch (err) {
# MAGIC     console.error("Fallback copy failed:", err);
# MAGIC     alert("Could not copy to clipboard. Please copy manually.");
# MAGIC   } finally {
# MAGIC     document.body.removeChild(textarea);
# MAGIC   }
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.6. View Submitted Feedback
# MAGIC
# MAGIC Navigate back to the model serving endpoint in **Serving** and click on **Traces** and select the latest trace (you might need to click the reload icon if you don't see your trace immediately). In the trace screen, you will see the familiar output, but you will also see your feedback submitted (see screenshot below).
# MAGIC
# MAGIC ![feedback-app3.png](../Includes/images/feedback evaluation/feedback-app3.png "feedback-app3.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.7. Viewing SME Feedback in a Notebook
# MAGIC
# MAGIC As noted above, the MLflow trace is available using the UI, but we can also view the assessment from the SME programmatically as shown in the next cell. This allows for reporting capabilities downstream if needed. We first get the mlflow experiment with `get_experiment_by_name()` and then search traces by inserting the experiment's name with `search_traces()`. Keep in mind that in our example we're using the `eval_demo_experiment` since this is the trace linked to our deployed model. We are also using the stored valued for `deployed_model_experiment_loc`, which was created as a part of the lab setup. This is the location of the experiment `eval_demo_experiment`.
# MAGIC
# MAGIC After running the next cell, scroll and expand the assessment for the 1 record returned. This will show the latest assessment (since we used `head(1)` on the generated Pandas dataframe).

# COMMAND ----------

deployed_model_experiment_loc

# COMMAND ----------

import pandas 

mlflow.set_tracking_uri("databricks")

## Resolve experiment ID
exp = mlflow.get_experiment_by_name(deployed_model_experiment_loc)
exp_id = exp.experiment_id
print("Experiment ID:", exp_id)

## Query traces using string experiment ID; limit to top-level fields for safe display
traces_df = mlflow.search_traces(
    locations=[str(exp_id)],            ## must be str
    include_spans=True,                ## exclude heavy nested span trees
    return_type="pandas"
)
traces_df.head(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you have successfully implemented human feedback workflows using MLflow across different reviewer personas:
# MAGIC
# MAGIC - **Developer feedback** through direct trace annotation in the MLflow UI
# MAGIC - **SME feedback** using the interactive Chat UI for real-time evaluation
# MAGIC
# MAGIC You've learned to distinguish between feedback and expectation assessment types, navigate MLflow's tracing interface, and understand the permissions and access patterns required for different reviewer types. These skills are essential for building robust AI applications that continuously improve through structured human feedback collection.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>