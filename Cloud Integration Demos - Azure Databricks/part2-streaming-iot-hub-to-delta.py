# Databricks notebook source
# MAGIC %run "./part0-create-variables"

# COMMAND ----------

# MAGIC %md 
# MAGIC #Goals for this demo
# MAGIC 1. [IoT Simulator](https://azure-samples.github.io/raspberry-pi-web-simulator/) + [script](https://github.com/tomatoTomahto/azure_databricks_iot/blob/master/azure_iot_simulator.js) writes live events to your IoT Hub</br>
# MAGIC 2. Databricks reads IoT Hub and writes (streaming) files to Delta Lake (Bronze Layer - raw)</br>
# MAGIC 3. Databricks reads Delta Lake (Bronze Layer - raw) and writes data to Delta Lake (Silver Layer - enriched)</br>
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Azure Demos/azure-integration-demo-part2-v2.png" alt="" width="600">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #References
# MAGIC This demo is a short version of IoT Tutorial, check links below for detailed instructions:
# MAGIC 1. [IoT Demo - Part1](https://databricks.com/blog/2020/08/03/modern-industrial-iot-analytics-on-azure-part-1.html)
# MAGIC 2. [IoT Demo - Part2](https://databricks.com/blog/2020/08/11/modern-industrial-iot-analytics-on-azure-part-2.html)
# MAGIC 3. [IoT Demo - Part3](https://databricks.com/blog/2020/08/20/modern-industrial-iot-analytics-on-azure-part-3.html)
# MAGIC <br/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Run IoT Simulator
# MAGIC 1. Open [IoT Simulator](https://azure-samples.github.io/raspberry-pi-web-simulator/)
# MAGIC 2. Erase code on right-hand side of page
# MAGIC 3. Load new code base form this [script](https://github.com/tomatoTomahto/azure_databricks_iot/blob/master/azure_iot_simulator.js) 
# MAGIC 4. Replace connection string (line 15) with your IoT Hub details<br/>
# MAGIC ```const connectionString = 'HostName=<hub_hostname>;DeviceId=<hub_deviceid;SharedAccessKey=<access_key>';```
# MAGIC 5. Click on "Run" to start producing events

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Read streaming data from IoT Hub using Spark Streaming

# COMMAND ----------

# Pyspark and ML Imports
import time, os, json, requests
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

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

# MAGIC %md
# MAGIC 
# MAGIC #Check Raw Data

# COMMAND ----------

display(iot_stream)

# COMMAND ----------

# spark.sql(f"DROP TABLE IF EXISTS iot_demo.turbine_sensor_raw")
spark.sql(f"CREATE TABLE IF NOT EXISTS iot_demo.turbine_sensor_raw USING DELTA LOCATION '{BRONZE_PATH}turbine_sensor_raw'")
# spark.sql(f"DROP TABLE IF EXISTS iot_demo.weather_raw")
spark.sql(f"CREATE TABLE IF NOT EXISTS iot_demo.weather_raw USING DELTA LOCATION '{BRONZE_PATH}weather_raw'")

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Save data into Delta tables
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
    print(e)
    time.sleep(3)
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC #Check Silver data

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from iot_demo.turbine_sensor_agg

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from iot_demo.weather_agg

# COMMAND ----------


