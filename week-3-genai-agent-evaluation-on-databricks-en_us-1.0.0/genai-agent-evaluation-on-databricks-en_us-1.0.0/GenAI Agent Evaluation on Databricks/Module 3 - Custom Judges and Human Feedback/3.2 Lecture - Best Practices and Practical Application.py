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
# MAGIC # Lecture - Best Practices and Practical Application
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This final lecture consolidates best practices for agent evaluation and provides guidance for practical implementation. We'll explore tips for comprehensive evaluation, understand the MLflow ecosystem integration, and discuss how to apply these concepts to real-world scenarios.
# MAGIC
# MAGIC The lecture emphasizes that evaluation is deeply context-dependent and provides a framework for thinking about evaluation requirements specific to your applications, users, and business needs.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Apply best practices for comprehensive agent evaluation
# MAGIC - Understand how MLflow integrates with the broader evaluation ecosystem
# MAGIC - Identify key questions for designing evaluation strategies for your use cases
# MAGIC - Recognize evaluation as an ongoing discipline rather than a one-time activity
# MAGIC - Plan evaluation infrastructure that supports continuous improvement

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Tips for Agent Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Comprehensive Evaluation Strategy
# MAGIC
# MAGIC ![tips-for-agent-evaluation.png](../Includes/images/Evaluation with MLflow/tips-for-agent-evaluation.png "tips-for-agent-evaluation.png")
# MAGIC
# MAGIC Effective agent evaluation requires a holistic approach that considers multiple dimensions of quality, diverse evaluation scenarios, and systematic processes for continuous improvement.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Multi-Dimensional Quality Assessment
# MAGIC
# MAGIC **Quality dimensions to consider:**
# MAGIC
# MAGIC - **Correctness**: Are the factual claims in responses accurate?
# MAGIC - **Relevance**: Do responses address the user's actual question?
# MAGIC - **Completeness**: Are responses thorough without being verbose?
# MAGIC - **Safety**: Are responses free from harmful or inappropriate content?
# MAGIC - **Consistency**: Do similar queries receive similar quality responses?
# MAGIC - **Efficiency**: Do agents use tools and resources appropriately?
# MAGIC - **User experience**: Are responses helpful, clear, and well-formatted?

# COMMAND ----------

# MAGIC %md
# MAGIC ### A3. Evaluation Dataset Best Practices
# MAGIC
# MAGIC **Building robust evaluation datasets:**
# MAGIC
# MAGIC 1. **Start with user research**: Analyze actual user queries and pain points
# MAGIC 2. **Include failure cases**: Test scenarios where your agent should decline or redirect
# MAGIC 3. **Vary complexity**: Include simple lookups and complex multi-step reasoning
# MAGIC 4. **Test boundaries**: Include edge cases and ambiguous queries
# MAGIC 5. **Maintain diversity**: Cover different domains, query styles, and user types
# MAGIC 6. **Version systematically**: Track dataset changes alongside agent versions
# MAGIC 7. **Validate regularly**: Ensure ground truth remains accurate and relevant

# COMMAND ----------

# MAGIC %md
# MAGIC ### A4. Evaluation Workflow Best Practices
# MAGIC
# MAGIC **Systematic evaluation processes:**
# MAGIC
# MAGIC - **Automate where possible**: Use CI/CD pipelines to run evaluations on code changes
# MAGIC - **Set quality gates**: Define minimum thresholds for deployment
# MAGIC - **Compare systematically**: Always evaluate changes against baseline performance
# MAGIC - **Document rationales**: Capture reasoning behind evaluation decisions
# MAGIC - **Share results**: Make evaluation results accessible to stakeholders
# MAGIC - **Act on insights**: Use evaluation results to drive concrete improvements

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. MLflow Evaluation Ecosystem

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Ecosystem Integration
# MAGIC
# MAGIC ![mlflow-ecosystem.png](../Includes/images/Evaluation with MLflow/mlflow-ecosystem.png "mlflow-ecosystem.png")
# MAGIC
# MAGIC MLflow evaluation integrates with the broader Databricks ecosystem to provide comprehensive evaluation capabilities from development through production monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. MLflow Experiments and Runs
# MAGIC
# MAGIC MLflow Experiments organize your evaluation results, enabling comparison and analysis across agent iterations.
# MAGIC
# MAGIC **Experiment structure:**
# MAGIC
# MAGIC Each evaluation call to `mlflow.genai.evaluate()` creates a run within an experiment:
# MAGIC - **Runs** represent individual evaluation executions with specific configurations
# MAGIC - **Experiments** group related evaluation runs for comparison
# MAGIC - **Metrics** are logged at the run level (aggregated scores)
# MAGIC - **Artifacts** include detailed per-example results and evaluation datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Comparison and Analysis Best Practices
# MAGIC
# MAGIC **Comparison capabilities:**
# MAGIC
# MAGIC MLflow UI enables systematic comparison:
# MAGIC - Compare metrics across runs to quantify improvements
# MAGIC - Visualize metric trends over time
# MAGIC - Filter runs by tags, parameters, or metrics
# MAGIC - Export results for external analysis
# MAGIC
# MAGIC **Best practices:**
# MAGIC
# MAGIC - Use consistent experiment naming conventions
# MAGIC - Tag runs with meaningful metadata (agent version, configuration, dataset version)
# MAGIC - Log hyperparameters and configuration as run parameters
# MAGIC - Add descriptive notes documenting run context and findings

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Practical Application Framework

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Key Questions for Your Use Case
# MAGIC
# MAGIC As you work through the demonstrations, consider how these concepts apply to your own use cases:
# MAGIC
# MAGIC **Questions to guide your thinking:**
# MAGIC - What quality dimensions matter most for your application?
# MAGIC - Which failure modes would be most problematic in production?
# MAGIC - How will you curate evaluation datasets that reflect real usage?
# MAGIC - What existing systems or processes could inform your evaluation criteria?
# MAGIC - How will you integrate evaluation into your development workflow?
# MAGIC - What metrics will you track in production monitoring?

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Context-Dependent Evaluation
# MAGIC
# MAGIC Effective evaluation is deeply context-dependent. While MLflow provides powerful tools, you must define what "good" means for your specific application, users, and business requirements.
# MAGIC
# MAGIC **Consider your context:**
# MAGIC
# MAGIC - **Domain requirements**: Medical, legal, or financial applications may require specialized evaluation criteria
# MAGIC - **User expectations**: Expert users may expect different response styles than general consumers
# MAGIC - **Risk tolerance**: High-stakes applications require more rigorous evaluation than experimental tools
# MAGIC - **Resource constraints**: Balance evaluation thoroughness with development velocity
# MAGIC - **Regulatory requirements**: Some domains require specific compliance or audit capabilities

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Implementation Planning
# MAGIC
# MAGIC **Planning your evaluation implementation:**
# MAGIC
# MAGIC 1. **Start simple**: Begin with basic built-in judges and expand gradually
# MAGIC 2. **Identify stakeholders**: Include domain experts, users, and business stakeholders
# MAGIC 3. **Define success metrics**: Establish clear, measurable quality standards
# MAGIC 4. **Plan infrastructure**: Consider storage, compute, and governance requirements
# MAGIC 5. **Design feedback loops**: Connect evaluation results to development processes
# MAGIC 6. **Prepare for scale**: Ensure evaluation infrastructure can grow with your needs

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Evaluation as an Ongoing Discipline

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Continuous Improvement Mindset
# MAGIC
# MAGIC Remember that evaluation is not a one-time activity but an ongoing discipline. As your agents evolve, as usage patterns shift, and as your understanding deepens, your evaluation approach should evolve too. MLflow's flexible framework supports this continuous improvement while maintaining rigor and reproducibility.
# MAGIC
# MAGIC **Evolution drivers:**
# MAGIC
# MAGIC - **Agent capabilities**: New features require new evaluation criteria
# MAGIC - **User behavior**: Changing usage patterns demand dataset updates
# MAGIC - **Business requirements**: Shifting priorities may change quality definitions
# MAGIC - **Technical advances**: New evaluation techniques become available
# MAGIC - **Lessons learned**: Production experience informs evaluation improvements

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Building Evaluation Culture
# MAGIC
# MAGIC **Fostering evaluation excellence:**
# MAGIC
# MAGIC - **Make evaluation visible**: Share results and insights across teams
# MAGIC - **Celebrate improvements**: Recognize quality improvements driven by evaluation
# MAGIC - **Learn from failures**: Use evaluation failures as learning opportunities
# MAGIC - **Invest in tooling**: Provide teams with excellent evaluation infrastructure
# MAGIC - **Train team members**: Ensure everyone understands evaluation principles and practices

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Looking Forward

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. The Foundation for Quality
# MAGIC
# MAGIC The quality of your AI agents ultimately depends on the quality of your evaluation. Invest in building robust evaluation infrastructure early, and you'll reap benefits throughout your agent's lifecycle—from development through production and beyond.
# MAGIC
# MAGIC **Long-term benefits of good evaluation:**
# MAGIC
# MAGIC - **Faster development**: Quick feedback enables rapid iteration
# MAGIC - **Higher quality**: Systematic evaluation catches issues early
# MAGIC - **User trust**: Consistent quality builds user confidence
# MAGIC - **Operational efficiency**: Automated evaluation reduces manual testing
# MAGIC - **Continuous improvement**: Data-driven optimization becomes possible

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Next Steps
# MAGIC
# MAGIC You now have the conceptual foundation to implement systematic agent evaluation. The upcoming demonstrations will transform these concepts into practical skills, showing you exactly how to evaluate real agents, interpret results, and drive quality improvements.
# MAGIC
# MAGIC **Your evaluation journey:**
# MAGIC
# MAGIC 1. **Apply the concepts**: Work through the hands-on demonstrations
# MAGIC 2. **Adapt to your context**: Consider your specific requirements and constraints
# MAGIC 3. **Start simple**: Implement basic evaluation and expand gradually
# MAGIC 4. **Measure and improve**: Use evaluation results to drive concrete improvements
# MAGIC 5. **Share and learn**: Collaborate with others building evaluation capabilities

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ⚠️ Lab Checkpoint
# MAGIC <div style="border-left: 4px solid #ff9800; background: #fff3e0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <span style="font-size: 24px;"></span>
# MAGIC     <div>
# MAGIC       <strong style="color: #e65100; font-size: 1.1em;">Lab Checkpoint</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">Navigate to <strong>06 Lab - Developer and SME Feedback with MLflow</strong> to explore human feedback capabilities and complete your evaluation learning journey. When you are finished, navigate back to this lecture notebook for the conclusion.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC Effective agent evaluation is both an art and a science. It requires technical understanding of evaluation frameworks, domain expertise to define quality criteria, and systematic processes to ensure continuous improvement.
# MAGIC
# MAGIC MLflow provides the technical foundation, but success depends on thoughtful application of these tools to your specific context. By combining rigorous offline evaluation with comprehensive online monitoring, you can build agents that consistently deliver high-quality experiences to your users.
# MAGIC
# MAGIC The investment in evaluation infrastructure pays dividends throughout the agent lifecycle. Start building your evaluation capabilities today, and you'll be well-positioned to develop, deploy, and maintain high-quality AI agents that meet your users' needs and exceed their expectations.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>