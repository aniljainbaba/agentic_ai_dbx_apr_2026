# Databricks notebook source
# MAGIC %pip install --quiet --upgrade \
# MAGIC   databricks-sdk \
# MAGIC   databricks-vectorsearch \
# MAGIC   langchain \
# MAGIC   langchain-text-splitters \
# MAGIC   "protobuf<5" \
# MAGIC   "databricks-langchain==0.18"
# MAGIC
# MAGIC dbutils.library.restartPython()