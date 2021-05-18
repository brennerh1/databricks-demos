# Databricks notebook source
# MAGIC %run "/Users/william.braccialli@databricks.com/IoT-Demo/part0-create-variables"

# COMMAND ----------

# MAGIC %md 
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Azure Demos/Azure Databricks Lakehouse + with Azure Data Services.jpg" alt="" width="1400" height="1080">

# COMMAND ----------

# MAGIC %md 
# MAGIC #####Goals for this demo</br>
# MAGIC 1) Azure Data Factory connects to source system and lands data into parquet file in Data Lake Landing Zone</br>
# MAGIC 2) Azure Data Factory orchestrates the execution of a Databricks Notebook</br>
# MAGIC 3) Databricks reads new and updated files (changed data) using Auto Loader and lands it into Delta Lake table</br>
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Azure Demos/Data Factory to Azure Databricks.jpg" alt="" width="600" height="600">

# COMMAND ----------

# MAGIC %md
# MAGIC Auto Loader is a feature in Databricks that allows Data Engineers to read new/changed files from a landing zone as a data stream without having to manage CDC.
# MAGIC 
# MAGIC https://docs.microsoft.com/en-us/azure/databricks/spark/latest/structured-streaming/auto-loader

# COMMAND ----------

# Auto Loader configurations
cloudfile = {
  "cloudFiles.subscriptionId": varSubscriptionId,
#   "cloudFiles.connectionString": queuesas,
  "cloudFiles.format": "parquet",
  "cloudFiles.tenantId": varTenantId,
  "cloudFiles.clientId": varApplicationId,
  "cloudFiles.clientSecret": varAuthenticationKey,
  "cloudFiles.resourceGroup": varResourceGroupName,
  "cloudFiles.maxFilesPerTrigger": 1
}

# COMMAND ----------

# As of 4/13/21, Auto Loader requires a schema to be passed to the read, even when reading from parquet and JSON.  A feature to eliminate this requirement is in the roadmap
maintenanceheader_Schema_df = spark.read.parquet(maintenanceheaderPath)
parquetSchemaMaintenanceheader = maintenanceheader_Schema_df.schema

# COMMAND ----------

# As of 4/13/21, Auto Loader requires a schema to be passed to the read, even when reading from parquet and JSON.  A feature to eliminate this requirement is in the roadmap
poweroutput_Schema_df = spark.read.parquet(poweroutputPath)
parquetSchemaPoweroutput = poweroutput_Schema_df.schema

# COMMAND ----------

# read the maintenance_header records from the landing zone using Auto Loader
maintenanceheader_df = (
  spark.readStream.format("cloudFiles")
  .options(**cloudfile)
  .option("cloudFiles.useNotifications", "true")
  .schema(parquetSchemaMaintenanceheader)
  .load("/mnt/landingzone/fleetmaintenance/maintenanceheader/")
)

# COMMAND ----------

# read the power_output records from the landing zone using Auto Loader
poweroutput_df = (
  spark.readStream.format("cloudFiles")
  .options(**cloudfile)
  .option("cloudFiles.useNotifications", "true")
  .schema(parquetSchemaPoweroutput)
  .load("/mnt/landingzone/fleetmaintenance/poweroutput/")
)

# COMMAND ----------

(maintenanceheader_df.writeStream
  .format("delta")
  .outputMode("append")
  .trigger(once=True)
  .option("checkpointLocation", "/mnt/landingzone/checkPoint/fleetmaintenance/maintenanceheader/")
  .start("/mnt/demo/iot/silver/maintenanceheader")
)

# COMMAND ----------

(poweroutput_df.writeStream
  .format("delta")
  .outputMode("append")
  .trigger(once=True)
  .option("checkpointLocation", "/mnt/landingzone/checkPoint/fleetmaintenance/poweroutput/")
  .start("/mnt/demo/iot/silver/poweroutput")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Create tables on top of delta storage so that tables can be easily found and managed for end users

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS iot_demo.maintenance_header
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/demo/iot/silver/maintenanceheader'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS iot_demo.power_output
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/demo/iot/silver/poweroutput'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(1) AS maintenance_header_count FROM iot_demo.maintenance_header

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(1) AS power_output_count FROM iot_demo.power_output
