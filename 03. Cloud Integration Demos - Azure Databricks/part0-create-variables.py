# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

varApplicationId = dbutils.secrets.get(scope = "gui-keyvalt", key = "service-principal-client-id") #service principle id
varAuthenticationKey = dbutils.secrets.get(scope = "gui-keyvalt", key = "service-principal-secret") #service principle key
varTenantId = dbutils.secrets.get(scope = "gui-keyvalt", key = "tenant-id") #the directory id from azure active directory -> properties
varStorageAccountName = "maintlandingzone" #storage acccount name
varResourceGroupName = "oneenv" #resource group name
varSubscriptionId = dbutils.secrets.get(scope = "gui-keyvalt", key = "subscription-Id") #storage acccount name
varFileSystemName = "landing" #ADLS container name
# queuesas = dbutils.secrets.get(scope = "gui-keyvalt", key = "kv-queue-sas")

# COMMAND ----------

ROOT_PATH = f"/mnt/demo/iot/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"

# COMMAND ----------

# Delta Tables
maintenanceheaderSilverTable = "iot_demo.maintenance_header"
poweroutputSilverTable = "iot_demo.power_output"

# Landing Locations
maintenanceheaderPath = "/mnt/landingzone/fleetmaintenance/maintenanceheader/"
poweroutputPath = "/mnt/landingzone/fleetmaintenance/poweroutput/"

# Checkpoint Locations
maintenanceheaderCheckpointPath = "/mnt/landingzone/checkPoint/fleetmaintenance/maintenanceheader/"
poweroutputCheckpointPath = "/mnt/landingzone/checkPoint/fleetmaintenance/poweroutput/"
