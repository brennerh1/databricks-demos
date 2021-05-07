# Databricks notebook source
# MAGIC %md > **Note:** Auto Loader is not currently available on Databricks Community Edition. Please contact your Databricks representative to upgrade.
# MAGIC 
# MAGIC ### Setup
# MAGIC Open the companion `streamfiles.py` notebook in another browser tab, then click "Run All" in that tab right before running this notebook. `streamfiles.py` kicks off writes to the target directory every several seconds that we will use to demonstrate Auto Loader.

# COMMAND ----------

# clean up the workspace
dbutils.fs.rm("/tmp/iot_stream/", recurse=True)
dbutils.fs.rm("/tmp/iot_stream_chkpts/", recurse=True)
spark.sql(f"DROP TABLE IF EXISTS iot_stream")
spark.sql(f"DROP TABLE IF EXISTS iot_devices")
spark.sql("CREATE TABLE iot_devices USING DELTA AS SELECT * FROM json.`/databricks-datasets/iot/` WHERE 2=1")
spark.sql("SET spark.databricks.cloudFiles.schemaInference.enabled=true")
dbutils.fs.cp("/databricks-datasets/iot-stream/data-device/part-00000.json.gz",
              "/tmp/iot_stream/part-00000.json.gz", recurse=True)

# COMMAND ----------

input_data_path = "/tmp/iot_stream/"
chkpt_path = "/tmp/iot_stream_chkpts/"

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Getting your data into Delta Lake with Auto Loader and COPY INTO
# MAGIC Incrementally and efficiently load new data files into Delta Lake tables as soon as they arrive in your data lake (S3/Azure Data Lake/Google Cloud Storage).
# MAGIC 
# MAGIC <!-- <img src="https://databricks.com/wp-content/uploads/2021/02/telco-accel-blog-2-new.png" width=800/> -->
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/delta-data-ingestion.png" width=1000/>
# MAGIC 
# MAGIC <!-- <img src="https://databricks.com/wp-content/uploads/2020/02/dl-workflow2.png" width=750/> -->

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Auto Loader

# COMMAND ----------

df = (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "json")
      .option("cloudFiles.schemaLocation", chkpt_path)
      .load(input_data_path))

(df.writeStream.format("delta")
   .option("checkpointLocation", chkpt_path)
   .table("iot_stream"))

# COMMAND ----------

display(df.selectExpr("COUNT(*) AS record_count"))

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> Auto Loader with `triggerOnce`

# COMMAND ----------

df = (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "json")
      .option("cloudFiles.schemaLocation", chkpt_path)
      .load(input_data_path))

(df.writeStream.format("delta")
   .trigger(once=True)
   .option("checkpointLocation", chkpt_path)
   .table("iot_stream"))

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM iot_stream

# COMMAND ----------

# MAGIC %md ### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> SQL `COPY INTO` command
# MAGIC Retriable, idempotent, simple.

# COMMAND ----------

# MAGIC %sql
# MAGIC COPY INTO iot_devices
# MAGIC FROM "/databricks-datasets/iot/"
# MAGIC FILEFORMAT = JSON

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM iot_devices

# COMMAND ----------

# MAGIC %sql DESCRIBE HISTORY iot_devices

# COMMAND ----------

# MAGIC %md #### <img src="https://pages.databricks.com/rs/094-YMS-629/images/dbsquare.png" width=30/> View the documentation for [Auto Loader](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html) and [COPY INTO](https://docs.databricks.com/spark/2.x/spark-sql/language-manual/copy-into.html).

# COMMAND ----------

# clean up workspace
dbutils.fs.rm("/tmp/iot_stream/", recurse=True)
dbutils.fs.rm("/tmp/iot_stream_chkpts/", recurse=True)
spark.sql(f"DROP TABLE IF EXISTS iot_stream")
spark.sql(f"DROP TABLE IF EXISTS iot_devices")
