# Databricks notebook source
# MAGIC %md 
# MAGIC #####Goals for this demo</br>
# MAGIC 1) Databricks reads IoT Telemetry + Maintanance data from Delta Lake (Silver Layer - enriched), joins data and stores on Delta Lake (Gold Layer - aggreaged) </br>
# MAGIC 2) Databricks writes Gold Layer data to Synapse Analytics Dedicated SQL Pool </br>
# MAGIC 3) Data is ready to be consumed by BI/Analytics and Data Science / Machine Learning</br>
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Azure Demos/azure-integration-demo-part3-v2.png" alt="" width="600">

# COMMAND ----------

import time
from pyspark.sql import functions as F


ROOT_PATH = f"/mnt/demo/iot/"
CHECKPOINT_PATH = ROOT_PATH + "checkpoint/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"

spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type",  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id", "bf2d513a-68bc-4d21-80ed-d0c2e9cdba17")
spark.conf.set("fs.azure.account.oauth2.client.secret", dbutils.secrets.get(scope="gui-keyvalt",key="gui-service-principal"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint", "https://login.microsoftonline.com/9f37a392-f0ae-4280-9796-f1864a10effc/oauth2/token")
SYNAPSE_PATH = "abfss://iot-demo@guitestdata.dfs.core.windows.net/synapse-tmp/"
CHECKPOINT_PATH_SYNAPSE = "/mnt/iot-demo/synapse-checkpoint/"
JDBC_URL = "jdbc:sqlserver://fieldengdeveastus2syn.sql.azuresynapse.net:1433;database=dalepool;encrypt=true;trustServerCertificate=true;hostNameInCertificate=*.sql.azuresynapse.net;loginTimeout=30;Authentication=ActiveDirectoryIntegrated"

# COMMAND ----------

from pyspark.sql.functions import *
df_turbine_sensor = spark.readStream.format('delta').option('ignoreChanges',True).table('iot_demo.turbine_sensor_agg') \
  .withColumn("hour", date_format(col("window"), 'H:mm'))
df_weather = spark.readStream.format('delta').option('ignoreChanges',True).table('iot_demo.weather_agg')
df_maintenanceheader = spark.readStream.format('delta').option('ignoreChanges',True).table('iot_demo.maintenance_header')
df_poweroutput = spark.readStream.format('delta').option('ignoreChanges',True).table('iot_demo.power_output')

# COMMAND ----------

df_gold_turbine = spark.sql("""
SELECT 
  sensor.*,
  weather.temperature, weather.humidity, weather.windspeed, weather.winddirection,
  maint.maintenance,
  power.power
FROM iot_demo.turbine_sensor_agg sensor
INNER JOIN iot_demo.weather_agg weather
   ON sensor.date = weather.date 
   AND sensor.window = weather.date
INNER JOIN iot_demo.maintenance_header maint
   ON sensor.date = maint.date
   and sensor.deviceid = maint.deviceid
INNER JOIN iot_demo.power_output power
   ON sensor.date = power.date
   AND sensor.window = power.window
   AND sensor.deviceid = power.deviceid
""")

# COMMAND ----------

display(df_gold_turbine)

# COMMAND ----------

def merge_delta(incremental, target): 
  incremental.dropDuplicates(['date','window','deviceid']).createOrReplaceTempView("incremental")
  
  try:
    # MERGE records into the target table using the specified join key
    incremental._jdf.sparkSession().sql(f"""
      MERGE INTO delta.`{target}` t
      USING incremental i
      ON i.date=t.date AND i.window = t.window AND i.deviceId = t.deviceid
      WHEN MATCHED THEN UPDATE SET *
      WHEN NOT MATCHED THEN INSERT *
    """)
  except:
    # If the †arget table does not exist, create one
    incremental.write.format("delta").partitionBy("date").save(target)
    
turbine_gold = (
  df_gold_turbine
    .writeStream                                                               # Write the resulting stream
    .foreachBatch(lambda i, b: merge_delta(i, GOLD_PATH + "turbine_gold"))    # Pass each micro-batch to a function
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH + "turbine_gold")             # Checkpoint so we can restart streams gracefully
    .start()
)

# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS iot_demo.turbine_gold")
# Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS iot_demo.turbine_gold USING DELTA LOCATION "{GOLD_PATH + "turbine_gold"}"')
    break
  except Exception as e:
    print("error, trying agian in 3 seconds")
    time.sleep(3)
    pass

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from iot_demo.turbine_gold

# COMMAND ----------

spark.conf.set("spark.databricks.sqldw.writeSemantics", "copy")                           # Use COPY INTO for faster loads to Synapse from Databricks

write_to_synapse = (
  spark.readStream
    .format('delta')
    .option('ignoreChanges',True)
    .table('iot_demo.turbine_gold') # Read in Gold turbine readings from Delta as a stream
    .writeStream
    .format("com.databricks.spark.sqldw")                                     # Write to Synapse (SQL DW connector)
    .option("url",JDBC_URL)                                # SQL Pool JDBC connection (with SQL Auth) string
    .option("tempDir", SYNAPSE_PATH)                                                      # Temporary ADLS path to stage the data (with forwarded permissions)
    .option("enableServicePrincipalAuth", "true")
    .option("dbTable", "iot_demo.turbine_gold")                                                # Table in Synapse to write to
    .option("checkpointLocation", CHECKPOINT_PATH_SYNAPSE + "turbine_gold")                              # Checkpoint for resilient streaming
    .start()
)

# COMMAND ----------

 
