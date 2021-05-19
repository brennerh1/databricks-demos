# Databricks notebook source
#demo sql analytics reading raw (bronze) and (gold tables)

# COMMAND ----------

# MAGIC %md
# MAGIC Azure Databricks SQL Analytics Docs https://docs.microsoft.com/en-us/azure/databricks/scenarios/sql/
# MAGIC 
# MAGIC Connecting with Power BI and other BI Tools https://docs.microsoft.com/en-us/azure/databricks/integrations/bi/index-sqla

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM iot_demo.turbine_gold
