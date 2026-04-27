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
# MAGIC # Lecture - Offline vs. Online Evaluation Strategies
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Effective agent evaluation requires both offline validation using curated datasets and online monitoring of production performance. This lecture explores the complementary roles of offline and online evaluation, their respective strengths and limitations, and how to create feedback loops between them.
# MAGIC
# MAGIC We'll examine when to use each approach, understand their implementation patterns, and learn how to build evaluation systems that evolve with real-world usage while maintaining rigorous quality standards.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Identify when to use offline versus online evaluation strategies
# MAGIC - Understand the strengths and limitations of each approach
# MAGIC - Describe how offline and online evaluation complement each other
# MAGIC - Explain how to create feedback loops between production and evaluation
# MAGIC - Recognize best practices for comprehensive evaluation strategies

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Evaluation Strategy Overview

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Two Complementary Approaches
# MAGIC
# MAGIC ![offline-vs-online-evaluation.png](../Includes/images/Evaluation with MLflow/offline-vs-online-evaluation.png "offline-vs-online-evaluation.png")
# MAGIC
# MAGIC Agent evaluation requires both controlled validation and real-world monitoring. Offline and online evaluation serve different but complementary purposes in ensuring agent quality throughout the development and deployment lifecycle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Offline Evaluation: Pre-Deployment Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Controlled Environment Testing
# MAGIC
# MAGIC Offline evaluation tests your agent using curated datasets before deployment. This approach provides controlled, reproducible validation of agent quality.
# MAGIC
# MAGIC **Characteristics of offline evaluation:**
# MAGIC
# MAGIC - **Controlled environment**: You control exactly what inputs the agent receives
# MAGIC - **Reproducibility**: Same dataset yields consistent evaluation results (modulo LLM non-determinism)
# MAGIC - **Ground truth availability**: You can include expected outputs, facts, or expert-labeled guidelines
# MAGIC - **Comprehensive coverage**: Intentionally test edge cases, failure modes, and diverse scenarios
# MAGIC - **Pre-deployment gate**: Prevents deploying agents that don't meet quality standards

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Offline Evaluation Workflow
# MAGIC
# MAGIC **Offline evaluation workflow:**
# MAGIC
# MAGIC 1. **Dataset creation**: Curate representative examples covering expected use cases, edge cases, and known failure patterns
# MAGIC 2. **Ground truth labeling**: Add expected outputs, facts, or guidelines (often with domain expert involvement)
# MAGIC 3. **Agent execution**: Run your agent on the dataset to generate responses
# MAGIC 4. **Scorer application**: Apply multiple judges to assess various quality dimensions
# MAGIC 5. **Analysis**: Review aggregated metrics and individual failures to identify improvements
# MAGIC 6. **Iteration**: Update agent, re-evaluate, compare results

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Offline Evaluation Strengths and Limitations
# MAGIC
# MAGIC **Strengths:**
# MAGIC - Enables rigorous validation before users encounter your agent
# MAGIC - Supports A/B testing different agent configurations
# MAGIC - Provides baseline metrics for production monitoring
# MAGIC
# MAGIC **Limitations:**
# MAGIC - Dataset may not fully represent real user behavior
# MAGIC - Static datasets become stale as usage patterns evolve
# MAGIC - Cannot capture issues that only emerge with scale or diverse users

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Online Evaluation: Production Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Real-World Performance Analysis
# MAGIC
# MAGIC Online evaluation analyzes agent performance using real user interactions in production. This approach captures how your agent performs with actual usage patterns, unexpected inputs, and diverse user populations.
# MAGIC
# MAGIC **Characteristics of online evaluation:**
# MAGIC
# MAGIC - **Real-world data**: Evaluates against actual user queries and behaviors
# MAGIC - **Scale validation**: Tests how your agent handles production load and request diversity
# MAGIC - **Drift detection**: Identifies when performance degrades over time
# MAGIC - **Continuous monitoring**: Ongoing quality signals rather than one-time assessment
# MAGIC - **User feedback integration**: Combines automated evaluation with actual user satisfaction

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Online Evaluation Workflow
# MAGIC
# MAGIC **Online evaluation workflow:**
# MAGIC
# MAGIC 1. **Inference logging**: Deploy agent with AI Gateway-enhanced inference tables enabled
# MAGIC 2. **Automatic tracing**: Every production request creates traces captured in inference tables
# MAGIC 3. **Scheduled evaluation**: Periodically run scorers on recent production traces
# MAGIC 4. **Alert configuration**: Set up monitoring alerts for quality degradation
# MAGIC 5. **Feedback collection**: Gather user thumbs up/down or explicit feedback
# MAGIC 6. **Dataset augmentation**: Extract interesting production examples for offline evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. MLflow's Production Monitoring Capabilities
# MAGIC
# MAGIC **MLflow's production monitoring capabilities:**
# MAGIC
# MAGIC Databricks provides built-in production monitoring for agents deployed via Model Serving:
# MAGIC - **Automated scorer execution**: Run MLflow scorers on production traces automatically
# MAGIC - **Quality dashboards**: Visualize evaluation metrics over time
# MAGIC - **Alerting**: Get notified when quality metrics drop below thresholds
# MAGIC - **Trace inspection**: Deep dive into individual failures or edge cases

# COMMAND ----------

# MAGIC %md
# MAGIC ### C4. Online Evaluation Strengths and Limitations
# MAGIC
# MAGIC **Strengths:**
# MAGIC - Reflects actual user experience and real-world performance
# MAGIC - Detects issues that weren't anticipated in offline testing
# MAGIC - Provides data for continuous improvement
# MAGIC
# MAGIC **Limitations:**
# MAGIC - Users may experience failures before you detect them
# MAGIC - More difficult to attribute causes (confounding from diverse inputs)
# MAGIC - Requires production infrastructure and monitoring systems

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Complementary Strategies

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. When to Use Each Approach
# MAGIC
# MAGIC Effective agent evaluation combines both offline and online approaches:
# MAGIC
# MAGIC **Offline evaluation for:**
# MAGIC - Pre-deployment validation and quality gates
# MAGIC - Comparing alternative agent implementations
# MAGIC - Testing specific hypotheses about agent behavior
# MAGIC - Rapid iteration during development
# MAGIC
# MAGIC **Online evaluation for:**
# MAGIC - Validating that offline results generalize to production
# MAGIC - Detecting performance degradation or drift
# MAGIC - Understanding real user needs and pain points
# MAGIC - Building evaluation datasets from actual usage

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Creating a Feedback Loop
# MAGIC
# MAGIC **Creating a feedback loop:**
# MAGIC
# MAGIC 1. Start with offline evaluation on carefully curated datasets
# MAGIC 2. Deploy to production with monitoring enabled
# MAGIC 3. Analyze production traces to identify failures and edge cases
# MAGIC 4. Add production examples to your offline evaluation dataset
# MAGIC 5. Use enhanced dataset to validate improvements before re-deploying
# MAGIC
# MAGIC This cycle ensures your evaluation evolves with your understanding of real-world usage while maintaining rigorous pre-deployment validation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Building Comprehensive Evaluation Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. Dataset Composition Principles
# MAGIC
# MAGIC The quality of your evaluation depends critically on your dataset. A well-designed evaluation dataset should be representative, diverse, and include appropriate ground truth.
# MAGIC
# MAGIC **Dataset composition principles:**
# MAGIC
# MAGIC **1. Representativeness**: Include examples reflecting typical user queries and use cases
# MAGIC - Analyze production logs or user research to identify common patterns
# MAGIC - Ensure distribution of query types matches expected usage
# MAGIC - Include varying query complexity levels
# MAGIC
# MAGIC **2. Edge case coverage**: Deliberately test boundary conditions and potential failures
# MAGIC - Ambiguous queries that could be interpreted multiple ways
# MAGIC - Queries requiring information not available in retrieval corpus
# MAGIC - Adversarial examples designed to trigger failure modes
# MAGIC - Queries in unexpected formats or phrasings

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Diversity and Ground Truth Approaches
# MAGIC
# MAGIC **3. Diversity dimensions**: Vary multiple aspects of inputs
# MAGIC - Query length (terse vs. verbose)
# MAGIC - Query complexity (simple fact lookup vs. multi-step reasoning)
# MAGIC - Domain coverage (different topics your agent should handle)
# MAGIC - User expertise levels (novice vs. expert domain knowledge)
# MAGIC
# MAGIC **Ground truth approaches:**
# MAGIC
# MAGIC **Expert labeling**: Domain experts create expected outputs or fact sets
# MAGIC - Most accurate but resource-intensive
# MAGIC - Essential for high-stakes applications
# MAGIC - Creates reusable gold standard dataset
# MAGIC
# MAGIC **Synthetic generation**: Use LLMs to generate expected responses
# MAGIC - Scalable but requires validation
# MAGIC - Useful for bootstrapping evaluation
# MAGIC - Should be spot-checked by humans
# MAGIC
# MAGIC **Production mining**: Extract examples from inference tables
# MAGIC - Reflects real usage patterns
# MAGIC - May require filtering or cleaning
# MAGIC - Can identify emergent failure modes

# COMMAND ----------

# MAGIC %md
# MAGIC ### E3. Dataset Maintenance
# MAGIC
# MAGIC **Dataset maintenance:**
# MAGIC
# MAGIC Evaluation datasets require ongoing curation:
# MAGIC - Regularly add new examples from production
# MAGIC - Remove or update stale examples
# MAGIC - Rebalance categories as usage patterns shift
# MAGIC - Version datasets alongside agent versions

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Integration with Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ### F1. Governance Layer for Evaluation
# MAGIC
# MAGIC Unity Catalog provides the governance layer for your evaluation workflow, ensuring traceability, access control, and lineage tracking.
# MAGIC
# MAGIC **Agent registration**: When you register agents in Unity Catalog:
# MAGIC - All agent dependencies (vector indexes, UC functions, model endpoints) are tracked
# MAGIC - Access controls determine who can evaluate or modify agents
# MAGIC - Lineage shows which evaluation datasets and metrics produced each version
# MAGIC - Aliases (like `@champion`) enable clean promotion workflows

# COMMAND ----------

# MAGIC %md
# MAGIC ### F2. Dataset and Trace Storage
# MAGIC
# MAGIC **Evaluation dataset storage**: Store datasets as Delta tables or volumes:
# MAGIC - Version datasets alongside agent versions
# MAGIC - Apply access controls to sensitive evaluation data
# MAGIC - Enable SQL-based dataset analysis and augmentation
# MAGIC - Support collaborative dataset curation
# MAGIC
# MAGIC **Trace storage**: MLflow traces are stored in Unity Catalog:
# MAGIC - Query traces using SQL for analysis
# MAGIC - Join traces with evaluation results for deeper insights
# MAGIC - Apply data governance policies to production trace data
# MAGIC - Enable cross-team trace sharing (where appropriate)
# MAGIC
# MAGIC This integration ensures your evaluation infrastructure benefits from the same governance, security, and collaboration capabilities as the rest of your data platform.

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Key Takeaways
# MAGIC
# MAGIC Effective agent evaluation requires both offline and online strategies:
# MAGIC
# MAGIC 1. **Complementary approaches**: Offline evaluation provides controlled validation while online evaluation captures real-world performance
# MAGIC 2. **Feedback loops**: Production data should inform offline evaluation datasets, creating continuous improvement cycles
# MAGIC 3. **Dataset quality**: Representative, diverse, and well-maintained datasets are crucial for meaningful evaluation
# MAGIC 4. **Governance integration**: Unity Catalog provides the infrastructure for scalable, governed evaluation workflows
# MAGIC 5. **Continuous evolution**: Evaluation strategies must adapt as agents, usage patterns, and requirements evolve
# MAGIC
# MAGIC The combination of rigorous offline validation and comprehensive online monitoring creates a robust evaluation framework that ensures agent quality throughout the development and deployment lifecycle.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>