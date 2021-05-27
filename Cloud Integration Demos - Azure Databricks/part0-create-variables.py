# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

varApplicationId = dbutils.secrets.get(scope = "xxxx-keyvalt", key = "service-principal-client-id") #service principle id
varAuthenticationKey = dbutils.secrets.get(scope = "xxxx-keyvalt", key = "service-principal-secret") #service principle key
varTenantId = dbutils.secrets.get(scope = "xxxx-keyvalt", key = "tenant-id") #the directory id from azure active directory -> properties
varStorageAccountName = "maintlandingzone" #storage acccount name
varResourceGroupName = "oneenv" #resource group name
varSubscriptionId = dbutils.secrets.get(scope = "xxxx-keyvalt", key = "subscription-Id") #storage acccount name
varFileSystemName = "landing" #ADLS container name

# COMMAND ----------

ROOT_PATH = f"/mnt/demo/iot/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"
CHECKPOINT_PATH = ROOT_PATH + "checkpoint/"
EVENT_HUB_NAME = "iothub-ehub-XXXXXXXXX"
IOT_ENDPOINT = dbutils.secrets.get(scope="xxxx-keyvalt", key="iot-endpoint")

spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type",  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id", "xxxx-xxxx-xxxx-xxxxx")
spark.conf.set("fs.azure.account.oauth2.client.secret", dbutils.secrets.get(scope="xxxx-keyvalt",key="xxxx-service-principal"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint", "https://login.microsoftonline.com/xxxx-xxxxxx-xxxxxx/oauth2/token")
SYNAPSE_PATH = "abfss://iot-demo@xxxxxxx.dfs.core.windows.net/synapse-tmp/"
CHECKPOINT_PATH_SYNAPSE = "/mnt/iot-demo/synapse-checkpoint/"
JDBC_URL = "jdbc:sqlserver://xxxxxxxxxxx.sql.azuresynapse.net:1433;database=dalepool;encrypt=true;trustServerCertificate=true;hostNameInCertificate=*.sql.azuresynapse.net;loginTimeout=30;Authentication=ActiveDirectoryIntegrated"

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

# COMMAND ----------


