# Databricks notebook source
ROOT_PATH = f"/mnt/demo/iot/"
CHECKPOINT_PATH = ROOT_PATH + "checkpoint/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"
CHECKPOINT_PATH_SYNAPSE = "/mnt/iot-demo/synapse-checkpoint/"
# Delta Tables
maintenanceheaderSilverTable = "iot_demo.maintenance_header"
poweroutputSilverTable = "iot_demo.power_output"
# Landing Locations
maintenanceheaderPath = "/mnt/landingzone/fleetmaintenance/maintenanceheader/"
poweroutputPath = "/mnt/landingzone/fleetmaintenance/poweroutput/"
# Checkpoint Locations
maintenanceheaderCheckpointPath = "/mnt/landingzone/checkPoint/fleetmaintenance/maintenanceheader/"
poweroutputCheckpointPath = "/mnt/landingzone/checkPoint/fleetmaintenance/poweroutput/"

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS iot_demo.turbine_sensor_agg")
spark.sql(f"DROP TABLE IF EXISTS iot_demo.weather_agg")
spark.sql(f"DROP TABLE IF EXISTS iot_demo.turbine_sensor_raw")
spark.sql(f"DROP TABLE IF EXISTS iot_demo.weather_raw")
spark.sql("DROP TABLE IF EXISTS {0}".format(maintenanceheaderSilverTable))
spark.sql("DROP TABLE IF EXISTS {0}".format(poweroutputSilverTable))
spark.sql(f"DROP TABLE IF EXISTS iot_demo.turbine_gold")

# COMMAND ----------

dbutils.fs.rm(f"{BRONZE_PATH}turbine_power_raw", recurse=True)
dbutils.fs.rm(f"{CHECKPOINT_PATH}", recurse=True)
dbutils.fs.rm(SILVER_PATH + "weather_agg", recurse=True)
dbutils.fs.rm(f"{GOLD_PATH}", recurse=True)
dbutils.fs.rm("/mnt/iot-demo/synapse-checkpoint/", recurse=True)

# COMMAND ----------

dbutils.fs.rm(maintenanceheaderCheckpointPath, recurse=True)
dbutils.fs.rm(SILVER_PATH + "maintenanceheader", recurse=True)
dbutils.fs.rm(poweroutputCheckpointPath, recurse=True)
dbutils.fs.rm(SILVER_PATH + "poweroutput", recurse=True)
