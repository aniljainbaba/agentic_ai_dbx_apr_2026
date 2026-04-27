# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks] databricks_langchain
# MAGIC %pip install -U databricks-langchain langchain-community langchain "langgraph>=1.1.5" "langgraph-prebuilt>=1.0.9"
# MAGIC %pip install unitycatalog-langchain[databricks]
# MAGIC %pip install unitycatalog-ai[databricks]
# MAGIC %pip install -U databricks-sdk
# MAGIC %pip install reportlab
# MAGIC %restart_python

# COMMAND ----------

#test code for initial setup
#create lab schema and volume
'''
spark.sql("CREATE SCHEMA IF NOT EXISTS labuser_ajain2.agent_poc_data")
spark.sql("CREATE VOLUME IF NOT EXISTS labuser_ajain2.agent_poc_data.`sf-listings`")
print("Schema and volume created successfully")
'''
# Verify

vols = spark.sql("SHOW VOLUMES IN labuser_ajain2.agent_poc_data").collect()
for v in vols:
    print(f"  Volume: {v['volume_name']}")

# COMMAND ----------

#test code for initial setup
# Create the v01 schema and sf-listings volume in labuser_ajain2
'''
spark.sql("CREATE SCHEMA IF NOT EXISTS labuser_ajain2.v01")
spark.sql("USE CATALOG labuser_ajain2")
spark.sql("USE SCHEMA v01")
spark.sql("CREATE VOLUME IF NOT EXISTS `sf-listings`")
print("Created labuser_ajain2.v01.sf-listings volume")
'''
# Copy CSV from where we already have it
'''
src = "/Volumes/labuser_ajain2/agent_poc_data/sf-listings/sf-airbnb.csv"
dst = "/Volumes/labuser_ajain2/v01/sf-listings/sf-airbnb.csv"
dbutils.fs.cp(src, dst)
print(f"Copied CSV to {dst}")
'''

# Verify
files = dbutils.fs.ls("/Volumes/labuser_ajain2/v01/sf-listings/")
for f in files:
    print(f"  {f.name} ({f.size} bytes)")

# COMMAND ----------

import sys, os

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
includes_dir = "../Includes"
course_dir = os.path.dirname(includes_dir)
sys.path.insert(0, course_dir)


config_path = f"{includes_dir}/config/setup-common-with-py-tools-lab.yaml"

# COMMAND ----------

from Includes import setup_demo_environment

env = setup_demo_environment(
    config_path=config_path,
    catalog_name="labuser_ajain2",
    databricks_share_name="labuser_ajain2",
)
catalog_name = env["catalog_name"]
schema_name = env["schema_name"]

# COMMAND ----------

import json

config = {
    "llm_endpoint": "databricks-gpt-oss-120b",
    "llm_temperature": 0.1,
    "system_prompt": "You are a helpful assistant. Make sure to use tools for additional functionality.",
    "tool_list": [
        f"{catalog_name}.{schema_name}.avg_neigh_price",
        f"{catalog_name}.{schema_name}.airbnb_posting_info",
    ],
}

with open("./lab_agent_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Configuration file created: ./lab_agent_config.json")

# COMMAND ----------


