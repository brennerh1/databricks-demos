# Databricks notebook source
# MAGIC %run "./part0-create-variables"

# COMMAND ----------

# MAGIC %md
# MAGIC Auto Loader is a feature in Databricks that allows Data Engineers to read new/changed files from a landing zone as a data stream without having to manage deltas.
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

try:
  dbutils.fs.ls(SILVER_PATH + "maintenanceheader")
except:
  maintenanceheader_Schema_df.filter("1=0").write.mode("overwrite").format("delta").save(SILVER_PATH + "maintenanceheader")

# COMMAND ----------

try:
  dbutils.fs.ls(SILVER_PATH + "poweroutput")
except:
  poweroutput_Schema_df.filter("1=0").write.mode("overwrite").format("delta").save(SILVER_PATH + "poweroutput")

# COMMAND ----------

# read the maintenance_header records from the landing zone using Auto Loader
maintenanceheader_df = spark.readStream.format("cloudFiles") \
  .options(**cloudfile) \
  .option("cloudFiles.useNotifications", "true") \
  .schema(parquetSchemaMaintenanceheader) \
  .load(maintenanceheaderPath)

# COMMAND ----------

# read the power_output records from the landing zone using Auto Loader
poweroutput_df = spark.readStream.format("cloudFiles") \
  .options(**cloudfile) \
  .option("cloudFiles.useNotifications", "true") \
  .schema(parquetSchemaPoweroutput) \
  .load(poweroutputPath)

# COMMAND ----------

#function to trim strings to cleanliness and consistency
def trimDFStrings(microDF):
  for columnName, columnType in microDF.dtypes:
    if columnType == "string":
      microDF = microDF.withColumn(columnName, trim(col(columnName)))
  return microDF

# COMMAND ----------

from delta.tables import *
from pyspark.sql.functions import *
import time

# UDF to perform an UPSERT/MERGE routine to Silver Delta Table
def mergeToDF(microDF, batchId, tableName, joinList):
  start_time = time.time()
  
  #delta table
  deltaTable = DeltaTable.forPath(spark, tableName)
  
  # create my join condition for the merge
  joinCond = ""
  for col in joinList:
    joinCond += "s." + col + " = t." + col + " AND " 
  joinCond = joinCond[:-4]
  
  # trim strings and drop any possible duplicates
  microDF = trimDFStrings(microDF)
  microDF = microDF.dropDuplicates(joinList)
  
  # perform Python based merge (upsert)
  (deltaTable.alias("t")
   .merge(
   microDF.alias("s"),
   joinCond)
   .whenMatchedUpdateAll()
   .whenNotMatchedInsertAll()
   .execute()
  )
  
  end_time = time.time()
  elapsed_time = end_time - start_time

  print(f"inside forEatchBatch for batchId:{batchId}, Table:{tableName}. Rows in passed dataframe:{microDF.count()}. Ellapsed time:{elapsed_time} seconds." )

# COMMAND ----------

maintenanceheader_df.writeStream \
  .format("delta") \
  .foreachBatch(lambda df, epochId: mergeToDF(df, epochId, SILVER_PATH + "maintenanceheader", ["deviceId","date"])) \
  .outputMode("update") \
  .trigger(once=True) \
  .option("checkpointLocation", maintenanceheaderCheckpointPath) \
  .start()

# COMMAND ----------

poweroutput_df.writeStream \
  .format("delta") \
  .foreachBatch(lambda df, epochId: mergeToDF(df, epochId, SILVER_PATH + "poweroutput", ["deviceId","date","window"])) \
  .outputMode("update") \
  .trigger(once=True) \
  .option("checkpointLocation", poweroutputCheckpointPath) \
  .start()

# COMMAND ----------

# MAGIC %md
# MAGIC Create tables on top of delta storage so that tables can be easily found and managed for end users

# COMMAND ----------

# maintenanceheaderquery = """CREATE TABLE IF NOT EXISTS {0}
# USING DELTA
# LOCATION '{1}'""".format(maintenanceheaderSilverTable, SILVER_PATH + "maintenanceheader")

# print(maintenanceheaderquery)

# spark.sql(maintenanceheaderquery)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS iot_demo.maintenance_header
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/demo/iot/silver/maintenanceheader'

# COMMAND ----------

# poweroutputquery = """CREATE TABLE IF NOT EXISTS {0}
# USING DELTA
# LOCATION '{1}'""".format(poweroutputSilverTable, SILVER_PATH + "poweroutput")

# print(poweroutputquery)

# spark.sql(poweroutputquery)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS iot_demo.power_output
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/demo/iot/silver/poweroutput'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(1) FROM iot_demo.maintenance_header

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(1) FROM iot_demo.power_output
