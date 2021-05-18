# Databricks notebook source
# Pyspark and ML Imports
import os, json, requests
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np 
import pandas as pd
import xgboost as xgb
import mlflow.xgboost
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
import random, string

# Random String generator for ML models served in AzureML
random_string = lambda length: ''.join(random.SystemRandom().choice(string.ascii_lowercase) for _ in range(length))

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC -- Calculate the power 6 hours ahead using Spark Windowing and build a feature_table to feed into our ML models
# MAGIC CREATE OR REPLACE VIEW iot_demo.turbine_feature_table AS
# MAGIC SELECT r.*,
# MAGIC   LEAD(power, 1) OVER (PARTITION BY r.deviceid ORDER BY window) as power_1_hour_ahead
# MAGIC FROM iot_demo.turbine_gold r
# MAGIC WHERE r.date <= CURRENT_DATE()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from iot_demo.turbine_feature_table

# COMMAND ----------

features = ['rpm', 'angle', 'temperature', 'humidity', 'windspeed', 'maintenance']
target = ['power_1_hour_ahead']

# COMMAND ----------

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = spark.table("iot_demo.turbine_feature_table").filter("power_1_hour_ahead is not null").toPandas()

x = df[features]
y = df[target]
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.20, random_state=30)

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
import numpy as np

search_space = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,4)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'criterion': hp.choice('criterion', ["mse", "mae"])
}

def train_model(params):
  mlflow.sklearn.autolog()
  with mlflow.start_run(nested=True):
        
   # Fit, train, and score the model
    model = RandomForestRegressor(**params)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)

    return {'status': STATUS_OK, 'loss': mean_squared_error(test_y, preds)} #, 'params': model.get_params()}
  
#spark_trials = SparkTrials(16)

with mlflow.start_run(run_name='skl_randfor_hyperopt'):
  best_params = fmin(
    fn = train_model,
    space = search_space,
    algo = tpe.suggest,
    max_evals = 30,
    trials = SparkTrials(5),
    rstate = np.random.RandomState(123)
  )

# COMMAND ----------

best_power_model_uri = (
  mlflow
  .search_runs()
  .sort_values("metrics.loss")['artifact_uri'].iloc[0] + '/model'
)

# COMMAND ----------

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, best_power_model_uri)

# COMMAND ----------

from pyspark.sql.types import ArrayType, FloatType


#Create a Spark UDF for the MLFlow model
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, best_power_model_uri)

#Execute the UDF against some data
df_predictions = (
  spark
  .table("iot_demo.turbine_feature_table")
  .withColumn(target[0] + '_prediction', pyfunc_udf(*(features+target)))
)

display(df_predictions)

# COMMAND ----------

# AML Workspace Information - replace with your workspace info
aml_resource_group = dbutils.widgets.get("Resource Group")
aml_subscription_id = dbutils.widgets.get("Subscription ID")
aml_region = dbutils.widgets.get("Region")
aml_workspace_name = "iot"
power_model = "power_prediction"
life_model = "life_prediction"

# Connect to a workspace (replace widgets with your own workspace info)
workspace = Workspace.create(name = aml_workspace_name,
                             subscription_id = aml_subscription_id,
                             resource_group = aml_resource_group,
                             location = aml_region,
                             exist_ok=True)


scoring_uris = {}
for model, path in [('life',best_life_model),('power',best_power_model)]:
  # Build images for each of our two models in Azure Container Instances
  print(f"-----Building image for {model} model-----")
  model_image, azure_model = mlflow.azureml.build_image(model_uri=path, 
                                                        workspace=workspace, 
                                                        model_name=model,
                                                        image_name=model,
                                                        description=f"XGBoost model to predict {model} of a turbine", 
                                                        synchronous=True)
  model_image.wait_for_creation(show_output=True)

  # Deploy web services to host each model as a REST API
  print(f"-----Deploying image for {model} model-----")
  dev_webservice_name = model + random_string(10)
  dev_webservice_deployment_config = AciWebservice.deploy_configuration()
  dev_webservice = Webservice.deploy_from_image(name=dev_webservice_name, image=model_image, deployment_config=dev_webservice_deployment_config, workspace=workspace)
  dev_webservice.wait_for_deployment()

  # Get the URI for sending REST requests to
  scoring_uris[model] = dev_webservice.scoring_uri

# COMMAND ----------

# Retrieve the Scoring URL provided by AzureML
power_uri = scoring_uris['power'] 
life_uri = scoring_uris['life'] 

# Construct a payload to send with the request
payload = {
  'angle':8,
  'rpm':6,
  'temperature':25,
  'humidity':50,
  'windspeed':5,
  'power':150,
  'age':10
}

def score_data(uri, payload):
  rest_payload = json.dumps({"data": [list(payload.values())]})
  response = requests.post(uri, data=rest_payload, headers={"Content-Type": "application/json"})
  return json.loads(response.text)

print(f'Current Operating Parameters: {payload}')
print(f'Predicted power (in kwh) from model: {score_data(power_uri, payload)}')
print(f'Predicted remaining life (in days) from model: {score_data(life_uri, payload)}')

# COMMAND ----------

#publish to Azure ML

# COMMAND ----------


