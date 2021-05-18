# Databricks notebook source
# MAGIC %md 
# MAGIC #####Goals for this demo</br>
# MAGIC 1) [IoT Simulator](https://azure-samples.github.io/raspberry-pi-web-simulator/) + [script](https://github.com/tomatoTomahto/azure_databricks_iot/blob/master/azure_iot_simulator.js) and configured for your IoT Hub writes events to IoT Hub</br>
# MAGIC 2) Databricks reads IoT Hub and writes (streaming) files to Delta Lake (Bronze Layer - raw)</br>
# MAGIC 3) Databricks reads Delta Lake (Bronze Layer - raw) and writes data to Delta Lake (Silver Layer - enriched)</br>
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Azure Demos/azure-integration-demo-part2-v2.png" alt="" width="600">

# COMMAND ----------

# Pyspark and ML Imports
import time, os, json, requests
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

ROOT_PATH = f"/mnt/demo/iot/"
CHECKPOINT_PATH = ROOT_PATH + "checkpoint/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"
EVENT_HUB_NAME = "iothub-ehub-gui-test-i-9443584-b75fbb1d77"
IOT_ENDPOINT = dbutils.secrets.get(scope="gui-keyvalt", key="iot-endpoint")
ehConf = { 
  'eventhubs.connectionString':sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(IOT_ENDPOINT),
  'ehName':EVENT_HUB_NAME
}


# COMMAND ----------

# Schema of incoming data from IoT hub
schema = "timestamp timestamp, deviceId string, temperature double, humidity double, windspeed double, winddirection string, rpm double, angle double"

# Read directly from IoT Hub using the EventHubs library for Databricks
iot_stream = (
  spark.readStream.format("eventhubs")                                               # Read from IoT Hubs directly
    .options(**ehConf)                                                               # Use the Event-Hub-enabled connect string
    .load()                                                                          # Load the data
    .withColumn('reading', F.from_json(F.col('body').cast('string'), schema))        # Extract the "body" payload from the messages
    .select('reading.*', F.to_date('reading.timestamp').alias('date'))               # Create a "date" field for partitioning
)

# Split our IoT Hub stream into separate streams and write them both into their own Delta locations
write_turbine_to_delta = (
  iot_stream.filter('temperature is null')                                           # Filter out turbine telemetry from other data streams
    .select('date','timestamp','deviceId','rpm','angle')                             # Extract the fields of interest
    .writeStream.format('delta')                                                     # Write our stream to the Delta format
    .partitionBy('date')                                                             # Partition our data by Date for performance
    .option("checkpointLocation", CHECKPOINT_PATH + "turbine_raw2")                  # Checkpoint so we can restart streams gracefully
    .start(BRONZE_PATH + "turbine_sensor_raw")                                       # Stream the data into an ADLS Path
)

write_weather_to_delta = (
  iot_stream.filter(iot_stream.temperature.isNotNull())                              # Filter out weather telemetry only
    .select('date','deviceid','timestamp','temperature','humidity','windspeed','winddirection') 
    .writeStream.format('delta')                                                     # Write our stream to the Delta format
    .partitionBy('date')                                                             # Partition our data by Date for performance
    .option("checkpointLocation", CHECKPOINT_PATH + "weather_raw2")                  # Checkpoint so we can restart streams gracefully
    .start(BRONZE_PATH + "weather_raw")                                              # Stream the data into an ADLS Path
)


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE DATABASE IF NOT EXISTS iot_demo

# COMMAND ----------

display(iot_stream)

# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS iot_demo.turbine_sensor_raw")
spark.sql(f"CREATE TABLE IF NOT EXISTS iot_demo.turbine_sensor_raw USING DELTA LOCATION '{BRONZE_PATH}turbine_sensor_raw'")
# spark.sql(f"DROP TABLE IF EXISTS iot_demo.weather_raw")
spark.sql(f"CREATE TABLE IF NOT EXISTS iot_demo.weather_raw USING DELTA LOCATION '{BRONZE_PATH}weather_raw'")

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Azure Demos/azure-integration-demo-part2-v2.png" alt="" width="600">

# COMMAND ----------

# Create functions to merge turbine and weather data into their target Delta tables
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
    
turbine_bronze_to_silver = (
  spark.readStream.format('delta').table("iot_demo.turbine_sensor_raw")
    .withColumn("window", F.window("timestamp", "1 hour")["start"])
    .groupby("date", "window", "deviceId")
    .agg(
      F.avg("rpm").alias("rpm"), 
      F.avg("angle").alias("angle")
    )  
    .writeStream
    .foreachBatch(lambda i, b: merge_delta(i, SILVER_PATH + "turbine_sensor_agg"))
    .outputMode("update")
    .option("checkpointLocation", CHECKPOINT_PATH + "turbine_sensor_agg")
    .start()
)

weather_bronze_to_silver = (
  spark.readStream.format('delta').table("iot_demo.weather_raw")
  .withColumn("window", F.window("timestamp", "1 hour")["start"])
  .groupby("date", "window", "deviceId")
  .agg(
      F.avg("temperature").alias("temperature"), 
      F.avg("humidity").alias("humidity"),
      F.avg("windspeed").alias("windspeed"),
      F.last("winddirection").alias("winddirection")
  )
  .writeStream
  .foreachBatch(lambda i, b: merge_delta(i, SILVER_PATH + "weather_agg"))
  .outputMode("update")
  .option("checkpointLocation", CHECKPOINT_PATH + "weather_agg")
  .start()
)


# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS iot_demo.turbine_sensor_agg")
# spark.sql(f"DROP TABLE IF EXISTS iot_demo.weather_agg")
# Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS iot_demo.turbine_sensor_agg USING DELTA LOCATION "{SILVER_PATH + "turbine_sensor_agg"}"')
    spark.sql(f'CREATE TABLE IF NOT EXISTS iot_demo.weather_agg USING DELTA LOCATION "{SILVER_PATH + "weather_agg"}"')
    break
  except Exception as e:
    print("error, trying agian in 3 seconds")
    time.sleep(3)
    pass

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from iot_demo.turbine_sensor_agg

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from iot_demo.weather_agg

# COMMAND ----------


